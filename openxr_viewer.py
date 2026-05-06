# openxr_viewer.py
# Renders Desktop2Stereo's depth-parallax left/right eye views into a VR headset
# via the pyopenxr binding (pip install pyopenxr).
# Uses a world-space virtual screen quad with proper per-eye view/projection matrices
# derived from xr.locate_views() for full 6DoF/3DoF head tracking.
# The depth-parallax FRAGMENT_SHADER from viewer.py is reused unchanged.

import sys
import math
import time
import ctypes
import queue as _queue

import glfw
import moderngl
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from utils import get_font_type
from OpenGL.GL import (
    glGenFramebuffers, glBindFramebuffer, glFramebufferTexture2D,
    glDeleteFramebuffers, glCheckFramebufferStatus,
    GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
    GL_FRAMEBUFFER_COMPLETE,
)

try:
    import xr
    OPENXR_AVAILABLE = True
except ImportError:
    OPENXR_AVAILABLE = False
    print("[OpenXRViewer] pyopenxr not installed. Run: pip install pyopenxr")

from viewer import FRAGMENT_SHADER

# GL_RGBA8 numeric value (used for swapchain format)
_GL_RGBA8 = 0x8058

# World-space vertex shader: applies MVP to place the quad in the scene
_WORLD_VERT = """
#version 330
in vec2 in_position;
in vec2 in_uv;
out vec2 uv;
uniform mat4 u_mvp;
void main() {
    uv = in_uv;
    gl_Position = u_mvp * vec4(in_position, 0.0, 1.0);
}
"""

# Simple passthrough shaders for the status window
_STATUS_VERT = """
#version 330
in vec2 in_position;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
    v_uv = in_uv;
}
"""

_STATUS_FRAG = """
#version 330
uniform sampler2D tex;
in vec2 v_uv;
out vec4 fragColor;
void main() {
    fragColor = texture(tex, v_uv);
}
"""


# ---------------------------------------------------------------------------
# XR math helpers (module-level, pure functions)
# ---------------------------------------------------------------------------

def _xr_quat_to_mat4(q):
    """XrQuaternionf → standard 4×4 rotation matrix (numpy, math row/col convention).

    Produces the matrix that left-multiplies a column vector: v' = R @ v.
    Callers must transpose before writing to OpenGL (which reads column-major).
    """
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y),  0],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x),  0],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y),  0],
        [  0,             0,             0,              1],
    ], dtype=np.float32)


def _pose_to_view_mat4(pose):
    """XrPosef → standard 4×4 view matrix (numpy, math row/col convention).

    The view matrix is the inverse of the head-pose model matrix:
      V = [ R^T | -R^T @ pos ]
          [  0  |      1     ]
    Caller must transpose before writing to OpenGL.
    """
    R  = _xr_quat_to_mat4(pose.orientation)[:3, :3]
    Rt = R.T                                              # inverse rotation
    t  = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float32)
    V  = np.eye(4, dtype=np.float32)
    V[:3, :3] = Rt
    V[:3, 3]  = -Rt @ t                                  # translation in last column
    return V


def _fov_to_proj_mat4(fov, near=0.05, far=100.0):
    """XrFovf → standard 4×4 OpenGL asymmetric-frustum projection matrix
    (numpy, math row/col convention). Caller must transpose before writing to OpenGL.
    """
    l = math.tan(fov.angle_left)  * near
    r = math.tan(fov.angle_right) * near
    t = math.tan(fov.angle_up)    * near
    b = math.tan(fov.angle_down)  * near
    p = np.zeros((4, 4), dtype=np.float32)
    p[0, 0] =  2 * near / (r - l)
    p[0, 2] =  (r + l)  / (r - l)      # col 2 of row 0
    p[1, 1] =  2 * near / (t - b)
    p[1, 2] =  (t + b)  / (t - b)      # col 2 of row 1
    p[2, 2] = -(far + near) / (far - near)
    p[2, 3] = -2 * far * near / (far - near)  # translation in last column
    p[3, 2] = -1.0                      # w = -z (perspective divide)
    return p


class OpenXRViewer:
    """
    Renders the depth-parallax stereo views into a VR headset using OpenXR.

    A virtual flat screen is placed in world space at `screen_distance` meters.
    Head pose from xr.locate_views() provides proper 6DoF view matrices per eye.
    The screen can be repositioned/resized via keyboard (status window) or
    VR controller thumbsticks.

    Parameters mirror the relevant subset of StereoWindow.__init__.
    Call run(first_rgb, first_depth) to enter the blocking frame loop.
    """

    def __init__(
        self,
        ipd=0.064,
        depth_ratio=1.0,
        convergence=0.0,
        frame_size=(1280, 720),
        fps=60,
        depth_q=None,
        show_fps=True,
        **kwargs,
    ):
        self.ipd_uv = ipd
        self.depth_strength = 0.1   # base multiplier (same as StereoWindow)
        self.depth_ratio = depth_ratio
        self.convergence = convergence
        self.frame_size = frame_size
        self.fps = fps
        self._time_sleep = 1.0 / fps
        self.depth_q = depth_q
        self.show_fps = show_fps

        # Virtual screen transform (world space, metres / radians)
        self.screen_distance = 2.0
        self.screen_width    = 2.0
        self.screen_height   = None   # derived from frame aspect ratio on first frame
        self.screen_pan_x    = 0.0
        self.screen_pan_y    = 0.0
        self.screen_yaw      = 0.0    # rotation around Y axis

        # Interaction speeds (per second)
        self._dist_speed = 0.5
        self._size_speed = 0.5
        self._pan_speed  = 0.5
        self._rot_speed  = 0.5

        # OpenXR handles
        self._xr_instance = None
        self._xr_system_id = None
        self._xr_session = None
        self._xr_space = None
        self._xr_swapchains = {}        # {eye_index: xr.Swapchain}
        self._swapchain_images = {}     # {eye_index: [XrSwapchainImageOpenGLKHR, ...]}
        self._swapchain_sizes = {}      # {eye_index: (w, h)}
        self._fbo_cache = {}            # {(eye_index, image_index): (raw_id, mgl_fbo)}
        self._session_running = False

        # Controller action handles (set by _init_controller_actions)
        self._action_set = None
        self._act_left_stick = None
        self._act_right_stick = None

        # Status window (FPS / latency / session info / screen transform)
        self.font = None
        self.font_type = get_font_type()
        self.base_font_size = 22
        self.text_padding = 8
        self.text_spacing = 3
        self.actual_fps = 0.0
        self.total_latency = 0.0
        self._status_prog = None
        self._status_vao = None
        self._status_tex = None
        try:
            self.font = ImageFont.truetype(self.font_type, self.base_font_size)
        except Exception:
            try:
                self.font = ImageFont.load_default()
            except Exception:
                self.font = None

        # ModernGL / GL handles
        self.window = None
        self.ctx = None
        self.prog = None
        self.quad_vao = None
        self.color_tex = None
        self.depth_tex = None
        self._texture_size = None

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_glfw(self):
        if not glfw.init():
            raise RuntimeError("[OpenXRViewer] GLFW init failed")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 5)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
        self._status_win_w, self._status_win_h = 460, 260
        self.window = glfw.create_window(
            self._status_win_w, self._status_win_h,
            "Desktop2Stereo XR — Status", None, None,
        )
        if not self.window:
            glfw.terminate()
            raise RuntimeError("[OpenXRViewer] GLFW window creation failed")
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)

        # Keyboard controls — keep a reference so it isn't GC'd
        self._key_callback_ref = self._make_key_callback()
        glfw.set_key_callback(self.window, self._key_callback_ref)

    def _make_key_callback(self):
        viewer = self  # capture self for the closure
        def _cb(window, key, scancode, action, mods):
            if action not in (glfw.PRESS, glfw.REPEAT):
                return
            d = 0.1; s = 0.15; p = 0.1; r = 0.05
            if   key == glfw.KEY_W:     viewer.screen_distance = max(0.3, viewer.screen_distance - d)
            elif key == glfw.KEY_S:     viewer.screen_distance += d
            elif key == glfw.KEY_UP:    viewer.screen_pan_y += p
            elif key == glfw.KEY_DOWN:  viewer.screen_pan_y -= p
            elif key == glfw.KEY_LEFT:  viewer.screen_pan_x -= p
            elif key == glfw.KEY_RIGHT: viewer.screen_pan_x += p
            elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
                viewer.screen_width += s; viewer.screen_height = None
            elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
                viewer.screen_width = max(0.3, viewer.screen_width - s)
                viewer.screen_height = None
            elif key == glfw.KEY_Q: viewer.screen_yaw += r
            elif key == glfw.KEY_E: viewer.screen_yaw -= r
            elif key == glfw.KEY_R:
                viewer.screen_distance = 2.0; viewer.screen_pan_x = 0.0
                viewer.screen_pan_y = 0.0;    viewer.screen_yaw = 0.0
                viewer.screen_width = 2.0;    viewer.screen_height = None
        return _cb

    def _init_moderngl(self):
        self.ctx = moderngl.create_context()

        # World-space stereo rendering program (HMD eyes)
        self.prog = self.ctx.program(
            vertex_shader=_WORLD_VERT,
            fragment_shader=FRAGMENT_SHADER,
        )
        self.prog['u_convergence'].value = self.convergence
        self.prog['tex_color'].value = 0
        self.prog['tex_depth'].value = 1

        vertices = np.array([
            -1, -1, 0, 0,
             1, -1, 1, 0,
            -1,  1, 0, 1,
             1,  1, 1, 1,
        ], dtype='f4')
        vbo = self.ctx.buffer(vertices)
        self.quad_vao = self.ctx.vertex_array(
            self.prog, [(vbo, '2f 2f', 'in_position', 'in_uv')]
        )

        # Status window program (simple texture blit)
        self._status_prog = self.ctx.program(
            vertex_shader=_STATUS_VERT,
            fragment_shader=_STATUS_FRAG,
        )
        vbo2 = self.ctx.buffer(vertices.tobytes())
        self._status_vao = self.ctx.vertex_array(
            self._status_prog, [(vbo2, '2f 2f', 'in_position', 'in_uv')]
        )
        self._status_tex = self.ctx.texture(
            (self._status_win_w, self._status_win_h), 3, dtype='f1',
        )

    def _init_openxr(self):
        # 1. Instance
        app_info = xr.ApplicationInfo(
            application_name="Desktop2Stereo",
            application_version=1,
            engine_name="D2S",
            engine_version=1,
            api_version=xr.XR_CURRENT_API_VERSION,
        )
        create_info = xr.InstanceCreateInfo(
            application_info=app_info,
            enabled_extension_names=[xr.KHR_OPENGL_ENABLE_EXTENSION_NAME],
        )
        self._xr_instance = xr.create_instance(create_info)
        print("[OpenXRViewer] XrInstance created")

        # 2. System
        self._xr_system_id = xr.get_system(
            self._xr_instance,
            xr.SystemGetInfo(form_factor=xr.FormFactor.HEAD_MOUNTED_DISPLAY),
        )

        # 3. Verify GL requirements (mandatory before session creation)
        xr.get_opengl_graphics_requirements_khr(self._xr_instance, self._xr_system_id)

        # 4. Graphics binding — platform-specific
        if sys.platform == "win32":
            from OpenGL.WGL import wglGetCurrentContext, wglGetCurrentDC
            binding = xr.GraphicsBindingOpenGLWin32KHR(
                h_dc=wglGetCurrentDC(),
                h_glrc=wglGetCurrentContext(),
            )
        else:
            from OpenGL.GLX import glXGetCurrentContext, glXGetCurrentDisplay, glXGetCurrentDrawable
            binding = xr.GraphicsBindingOpenGLXlibKHR(
                x_display=glXGetCurrentDisplay(),
                glx_drawable=glXGetCurrentDrawable(),
                glx_context=glXGetCurrentContext(),
            )

        # 5. Session
        session_info = xr.SessionCreateInfo(
            system_id=self._xr_system_id,
            next=ctypes.cast(ctypes.pointer(binding), ctypes.c_void_p),
        )
        self._xr_session = xr.create_session(self._xr_instance, session_info)
        print("[OpenXRViewer] XrSession created")

        # 6. Reference space — prefer STAGE (floor origin), fall back to LOCAL
        available_spaces = xr.enumerate_reference_spaces(self._xr_session)
        ref_type = (
            xr.ReferenceSpaceType.STAGE
            if xr.ReferenceSpaceType.STAGE in available_spaces
            else xr.ReferenceSpaceType.LOCAL
        )
        self._xr_space = xr.create_reference_space(
            self._xr_session,
            xr.ReferenceSpaceCreateInfo(
                reference_space_type=ref_type,
                pose_in_reference_space=xr.Posef(),
            ),
        )

        # 7. Swapchains — one per eye
        view_configs = xr.enumerate_view_configuration_views(
            self._xr_instance,
            self._xr_system_id,
            xr.ViewConfigurationType.PRIMARY_STEREO,
        )
        for eye_index, vcv in enumerate(view_configs):
            sc_w = vcv.recommended_image_rect_width
            sc_h = vcv.recommended_image_rect_height
            print(f"[OpenXRViewer] Eye {eye_index} swapchain: {sc_w}x{sc_h}")

            sc_info = xr.SwapchainCreateInfo(
                usage_flags=(
                    xr.SwapchainUsageFlags.COLOR_ATTACHMENT_BIT |
                    xr.SwapchainUsageFlags.SAMPLED_BIT
                ),
                format=_GL_RGBA8,
                sample_count=1,
                width=sc_w,
                height=sc_h,
                face_count=1,
                array_size=1,
                mip_count=1,
            )
            swapchain = xr.create_swapchain(self._xr_session, sc_info)
            images = xr.enumerate_swapchain_images(swapchain, xr.SwapchainImageOpenGLKHR)
            self._xr_swapchains[eye_index] = swapchain
            self._swapchain_images[eye_index] = images
            self._swapchain_sizes[eye_index] = (sc_w, sc_h)

        # 8. Controller actions (optional — silently disabled if action set creation fails)
        try:
            self._init_controller_actions()
        except Exception as e:
            print(f"[OpenXRViewer] Controller actions unavailable: {e}")

    def _init_controller_actions(self):
        """Set up OpenXR action set with left/right thumbstick vec2 actions."""
        self._action_set = xr.create_action_set(
            self._xr_instance,
            xr.ActionSetCreateInfo(
                action_set_name="screen_control",
                localized_action_set_name="Screen Control",
                priority=0,
            ),
        )
        subpaths = [
            xr.string_to_path(self._xr_instance, p)
            for p in ["/user/hand/left", "/user/hand/right"]
        ]

        def make_vec2(name, label):
            return xr.create_action(
                self._action_set,
                xr.ActionCreateInfo(
                    action_type=xr.ActionType.VECTOR2F_INPUT,
                    action_name=name,
                    localized_action_name=label,
                    count_subaction_paths=len(subpaths),
                    subaction_paths=subpaths,
                ),
            )

        self._act_left_stick  = make_vec2("left_stick",  "Left Stick")
        self._act_right_stick = make_vec2("right_stick", "Right Stick")

        stick_bindings = [
            ("/user/hand/left/input/thumbstick",  self._act_left_stick),
            ("/user/hand/right/input/thumbstick", self._act_right_stick),
        ]
        for profile in [
            "/interaction_profiles/oculus/touch_controller",
            "/interaction_profiles/valve/index_controller",
        ]:
            try:
                xr.suggest_interaction_profile_bindings(
                    self._xr_instance,
                    xr.InteractionProfileSuggestedBinding(
                        interaction_profile=xr.string_to_path(self._xr_instance, profile),
                        suggested_bindings=[
                            xr.ActionSuggestedBinding(
                                action=act,
                                binding=xr.string_to_path(self._xr_instance, path),
                            )
                            for path, act in stick_bindings
                        ],
                    ),
                )
            except Exception:
                pass

        xr.attach_session_action_sets(
            self._xr_session,
            xr.SessionActionSetsAttachInfo(action_sets=[self._action_set]),
        )

    def _init_textures(self, w, h):
        if self.color_tex:
            self.color_tex.release()
        if self.depth_tex:
            self.depth_tex.release()
        self.color_tex = self.ctx.texture((w, h), 3, dtype='f1')
        self.depth_tex = self.ctx.texture((w, h), 1, dtype='f4')
        self._texture_size = (w, h)

    # ------------------------------------------------------------------
    # Per-frame helpers
    # ------------------------------------------------------------------

    def _update_frame(self, rgb, depth):
        """Upload RGB and depth to GL textures (CPU path)."""
        import torch
        if hasattr(rgb, 'detach'):
            rgb_np = (
                rgb.permute(1, 2, 0).detach().contiguous()
                .clamp(0, 255).to(torch.uint8).cpu().numpy()
            )
        else:
            rgb_np = np.asarray(rgb, dtype=np.uint8)

        if hasattr(depth, 'detach'):
            depth_np = depth.detach().cpu().contiguous().float().numpy()
        else:
            depth_np = np.asarray(depth, dtype=np.float32)

        h, w = depth_np.shape
        if self._texture_size != (w, h):
            self._init_textures(w, h)
            # Update frame_size so _build_model_mat4 uses actual aspect ratio
            self.frame_size = (w, h)
            self.screen_height = None

        self.color_tex.write(rgb_np.astype('uint8', copy=False).tobytes())
        self.depth_tex.write(depth_np.tobytes())

    def _build_model_mat4(self):
        """Construct the screen quad's world-space model matrix (math row/col convention).

        The quad's in_position spans [-1,+1] in X and Y in model space.
        We scale to physical metres, then translate to (pan_x, pan_y, -distance)
        in world space (OpenXR: right-hand Y-up, forward = -Z), then apply yaw.
        Caller must transpose before writing to OpenGL.
        """
        if self.screen_height is None:
            fw, fh = self.frame_size
            self.screen_height = self.screen_width * (fh / fw if fw > 0 else 9 / 16)

        sx  = self.screen_width  / 2.0
        sy  = self.screen_height / 2.0
        cy  = math.cos(self.screen_yaw)
        sy_ = math.sin(self.screen_yaw)

        # Yaw rotation around Y (standard mathematical form, translation in col 3)
        rot_y = np.array([
            [ cy,  0, sy_, 0],
            [  0,  1,   0, 0],
            [-sy_,  0,  cy, 0],
            [  0,  0,   0, 1],
        ], dtype=np.float32)

        # Scale + translate: translation is in the last column (col 3), row 3 = [0,0,0,1]
        model = np.array([
            [sx,  0,  0, self.screen_pan_x    ],
            [ 0, sy,  0, self.screen_pan_y    ],
            [ 0,  0,  1, -self.screen_distance],
            [ 0,  0,  0, 1                    ],
        ], dtype=np.float32)

        return rot_y @ model

    def _get_or_create_fbo(self, eye_index, image_index, texture_id):
        """Lazily create and cache a ModernGL Framebuffer wrapping the swapchain texture.

        ctx.detect_framebuffer() is used so ModernGL's internal state tracking stays
        consistent — raw glBindFramebuffer() is invisible to ModernGL and would cause
        ctx.clear() / vao.render() to target the wrong framebuffer.
        """
        key = (eye_index, image_index)
        if key in self._fbo_cache:
            return self._fbo_cache[key]

        raw_id = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, raw_id)
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture_id, 0
        )
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        if status != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(
                f"[OpenXRViewer] FBO incomplete for eye {eye_index}, "
                f"image {image_index}: {status:#x}"
            )
        mgl_fbo = self.ctx.detect_framebuffer(raw_id)
        self._fbo_cache[key] = (raw_id, mgl_fbo)
        return raw_id, mgl_fbo

    def _render_status_window(self):
        """Render FPS / latency / session info / screen transform into the status window."""
        if self.font is None:
            return

        w, h = self._status_win_w, self._status_win_h
        img = Image.new('RGB', (w, h), (25, 25, 35))
        draw = ImageDraw.Draw(img)

        lines = []
        if self.show_fps:
            lines.append((f"FPS: {self.actual_fps:.1f}", (0, 230, 0)))
            if self.total_latency > 0:
                lines.append((f"Latency: {self.total_latency:.0f} ms", (0, 220, 220)))
        state_str = "Running" if self._session_running else "Waiting"
        lines.append((f"Session: {state_str}", (200, 200, 200)))
        if self._swapchain_sizes:
            sz = self._swapchain_sizes.get(0, (0, 0))
            lines.append((f"Eye res: {sz[0]}x{sz[1]}", (160, 160, 160)))

        lines.append(("--- Screen ---", (120, 120, 180)))
        ht = self.screen_height if self.screen_height is not None else "?"
        ht_str = f"{ht:.2f}" if isinstance(ht, float) else ht
        lines.append((f"Dist: {self.screen_distance:.2f}m  W: {self.screen_width:.2f}m  H: {ht_str}m", (200, 200, 100)))
        lines.append((f"Pan X: {self.screen_pan_x:.2f}  Pan Y: {self.screen_pan_y:.2f}  Yaw: {math.degrees(self.screen_yaw):.1f}°", (200, 200, 100)))

        lines.append(("--- Keys ---", (120, 120, 180)))
        lines.append(("[W/S] dist  [+/-] size  [Q/E] yaw", (140, 140, 140)))
        lines.append(("[Arrows] pan  [R] reset", (140, 140, 140)))

        pad = self.text_padding
        y = pad
        for text, color in lines:
            draw.text((pad, y), text, font=self.font, fill=color)
            try:
                bbox = draw.textbbox((0, 0), text, font=self.font)
                y += (bbox[3] - bbox[1]) + self.text_spacing
            except Exception:
                y += self.base_font_size + self.text_spacing

        # Flip vertically: PIL row-0 is top, OpenGL texture row-0 is bottom
        status_data = np.flipud(np.array(img, dtype=np.uint8))
        self._status_tex.write(status_data.tobytes())

        self.ctx.screen.use()
        self.ctx.viewport = (0, 0, w, h)
        self._status_tex.use(location=0)
        self._status_prog['tex'].value = 0
        self._status_vao.render(moderngl.TRIANGLE_STRIP)
        glfw.swap_buffers(self.window)

    def _render_eye(self, eye_index, mgl_fbo, view_mat, proj_mat):
        """Render one eye's parallax view into the swapchain FBO using world-space MVP.

        Left eye:  u_eye_offset = -ipd/2
        Right eye: u_eye_offset = +ipd/2
        """
        sc_w, sc_h = self._swapchain_sizes[eye_index]

        mgl_fbo.use()
        self.ctx.viewport = (0, 0, sc_w, sc_h)
        mgl_fbo.clear(0.0, 0.0, 0.0, 1.0)

        if self.color_tex is None or self.depth_tex is None:
            self.ctx.screen.use()
            return

        self.color_tex.use(location=0)
        self.depth_tex.use(location=1)

        model = self._build_model_mat4()
        mvp   = proj_mat @ view_mat @ model
        self.prog['u_mvp'].write(mvp.T.astype('f4').tobytes())

        eye_sign = -1.0 if eye_index == 0 else 1.0
        self.prog['u_eye_offset'].value    = eye_sign * self.ipd_uv / 2.0
        self.prog['u_depth_strength'].value = self.depth_strength * self.depth_ratio

        # Match the desktop viewer's default (unset) uniform value of (0.0, 0.0).
        # This makes pixel_size = 1/0 = infinity, disabling the depth‑gradient
        # disocclusion test and producing identical inpainting behaviour.
        self.prog['u_resolution'].value = (0.0, 0.0)

        self.quad_vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.screen.use()

    # ------------------------------------------------------------------
    # OpenXR event loop
    # ------------------------------------------------------------------

    def _poll_xr_events(self):
        """Drain the OpenXR event queue and handle session state transitions."""
        from utils import shutdown_event
        while True:
            try:
                event_buf = xr.poll_event(self._xr_instance)
            except xr.EventUnavailable:
                break

            event_type = event_buf.type

            if event_type == xr.StructureType.EVENT_DATA_SESSION_STATE_CHANGED:
                event = ctypes.cast(
                    ctypes.byref(event_buf),
                    ctypes.POINTER(xr.EventDataSessionStateChanged),
                ).contents
                state = xr.SessionState(event.state)
                if state == xr.SessionState.READY:
                    xr.begin_session(
                        self._xr_session,
                        xr.SessionBeginInfo(
                            primary_view_configuration_type=
                                xr.ViewConfigurationType.PRIMARY_STEREO
                        ),
                    )
                    self._session_running = True
                    print("[OpenXRViewer] Session READY — rendering started")

                elif state in (
                    xr.SessionState.STOPPING,
                    xr.SessionState.LOSS_PENDING,
                    xr.SessionState.EXITING,
                ):
                    xr.end_session(self._xr_session)
                    self._session_running = False
                    print(f"[OpenXRViewer] Session state → {state.name}; rendering paused")

            elif event_type == xr.StructureType.EVENT_DATA_INSTANCE_LOSS_PENDING:
                print("[OpenXRViewer] Instance loss pending — shutting down")
                shutdown_event.set()
                break

    def _poll_controller_input(self, dt):
        """Read thumbstick state and apply deltas to the screen transform."""
        if self._action_set is None:
            return
        try:
            xr.sync_actions(
                self._xr_session,
                xr.ActionsSyncInfo(active_action_sets=[
                    xr.ActiveActionSet(
                        action_set=self._action_set,
                        subaction_path=xr.NULL_PATH,
                    )
                ]),
            )
        except Exception:
            return

        def vec2(action, hand_path_str):
            try:
                path  = xr.string_to_path(self._xr_instance, hand_path_str)
                state = xr.get_action_state_vector2f(
                    self._xr_session,
                    xr.ActionStateGetInfo(action=action, subaction_path=path),
                )
                if state.is_active:
                    return state.current_state.x, state.current_state.y
            except Exception:
                pass
            return 0.0, 0.0

        DEAD = 0.15

        lx, ly = vec2(self._act_left_stick,  "/user/hand/left")
        rx, ry = vec2(self._act_right_stick, "/user/hand/right")

        if abs(lx) > DEAD:
            self.screen_pan_x += lx * self._pan_speed * dt
        if abs(ly) > DEAD:
            self.screen_distance = max(0.3, self.screen_distance - ly * self._dist_speed * dt)
        if abs(rx) > DEAD:
            self.screen_yaw += rx * self._rot_speed * dt
        if abs(ry) > DEAD:
            self.screen_width = max(0.3, self.screen_width + ry * self._size_speed * dt)
            self.screen_height = None

    # ------------------------------------------------------------------
    # Main blocking loop
    # ------------------------------------------------------------------

    def run(self, first_rgb=None, first_depth=None):
        """
        Blocking render loop. Exits when the OpenXR session ends, the GLFW
        window is closed, or shutdown_event is set.

        Pass the first rgb/depth frames already pulled from depth_q by main.py
        so no frame is wasted.
        """
        if not OPENXR_AVAILABLE:
            raise RuntimeError("pyopenxr not available")

        from utils import shutdown_event

        self._init_glfw()
        self._init_moderngl()

        try:
            self._init_openxr()
        except Exception as e:
            print(f"[OpenXRViewer] OpenXR init failed: {e}")
            self.cleanup()
            raise

        # Upload the first frame supplied by main.py
        if first_rgb is not None and first_depth is not None:
            self._update_frame(first_rgb, first_depth)

        # Default fallback projection (used before first locate_views succeeds)
        _default_fov = xr.Fovf(
            angle_left=-0.785, angle_right=0.785,
            angle_up=0.785,   angle_down=-0.785,
        )
        _default_proj = _fov_to_proj_mat4(_default_fov)

        frame_count = 0
        last_fps_time   = time.perf_counter()
        last_frame_time = time.perf_counter()

        while (
            not glfw.window_should_close(self.window)
            and not shutdown_event.is_set()
        ):
            now = time.perf_counter()
            dt = now - last_frame_time
            last_frame_time = now

            glfw.poll_events()
            self._poll_xr_events()

            if self._session_running:
                self._poll_controller_input(dt)

            if not self._session_running:
                self._render_status_window()
                time.sleep(0.01)
                continue

            # — Wait for the runtime to signal frame timing —
            frame_state = xr.wait_frame(self._xr_session, xr.FrameWaitInfo())
            xr.begin_frame(self._xr_session, xr.FrameBeginInfo())

            composition_layers = []

            if frame_state.should_render:
                # Try to get the latest depth-estimated frame (non-blocking)
                try:
                    rgb, depth, _ = self.depth_q.get(timeout=self._time_sleep)
                    self._update_frame(rgb, depth)

                    frame_count += 1
                    now2 = time.perf_counter()
                    elapsed = now2 - last_fps_time
                    if elapsed >= 1.0:
                        self.actual_fps = frame_count / elapsed
                        frame_count = 0
                        last_fps_time = now2

                except _queue.Empty:
                    pass   # re-use textures from the previous frame

                # Head-tracking pose for this frame
                try:
                    view_state, views = xr.locate_views(
                        self._xr_session,
                        xr.ViewLocateInfo(
                            view_configuration_type=xr.ViewConfigurationType.PRIMARY_STEREO,
                            display_time=frame_state.predicted_display_time,
                            space=self._xr_space,
                        ),
                    )
                except Exception:
                    views = [None, None]

                eye_layer_views = []

                for eye_index in range(2):
                    swapchain = self._xr_swapchains[eye_index]

                    img_index = xr.acquire_swapchain_image(
                        swapchain, xr.SwapchainImageAcquireInfo()
                    )
                    xr.wait_swapchain_image(
                        swapchain,
                        xr.SwapchainImageWaitInfo(timeout=xr.INFINITE_DURATION),
                    )

                    sc_image = self._swapchain_images[eye_index][img_index]
                    _, mgl_fbo = self._get_or_create_fbo(eye_index, img_index, sc_image.image)

                    view = views[eye_index] if views and views[eye_index] else None
                    view_mat = _pose_to_view_mat4(view.pose) if view else np.eye(4, dtype=np.float32)
                    proj_mat = _fov_to_proj_mat4(view.fov)   if view else _default_proj

                    self._render_eye(eye_index, mgl_fbo, view_mat, proj_mat)

                    xr.release_swapchain_image(swapchain, xr.SwapchainImageReleaseInfo())

                    sc_w, sc_h = self._swapchain_sizes[eye_index]
                    eye_layer_views.append(xr.CompositionLayerProjectionView(
                        pose=view.pose if view else xr.Posef(),
                        fov=view.fov  if view else _default_fov,
                        sub_image=xr.SwapchainSubImage(
                            swapchain=swapchain,
                            image_rect=xr.Rect2Di(
                                offset=xr.Offset2Di(x=0, y=0),
                                extent=xr.Extent2Di(width=sc_w, height=sc_h),
                            ),
                        ),
                    ))

                proj_layer = xr.CompositionLayerProjection(
                    space=self._xr_space,
                    views=eye_layer_views,
                )
                composition_layers.append(
                    ctypes.cast(ctypes.pointer(proj_layer),
                                ctypes.POINTER(xr.CompositionLayerBaseHeader))
                )

            xr.end_frame(
                self._xr_session,
                xr.FrameEndInfo(
                    display_time=frame_state.predicted_display_time,
                    environment_blend_mode=xr.EnvironmentBlendMode.OPAQUE,
                    layers=composition_layers,
                ),
            )

            self._render_status_window()

        self.cleanup()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        """Release all OpenXR and OpenGL resources."""
        raw_ids = [raw_id for raw_id, _ in self._fbo_cache.values()]
        if raw_ids:
            try:
                glDeleteFramebuffers(len(raw_ids), raw_ids)
            except Exception:
                pass
        self._fbo_cache.clear()

        for tex in (self._status_tex, self.color_tex, self.depth_tex):
            if tex:
                try:
                    tex.release()
                except Exception:
                    pass
        self._status_tex = self.color_tex = self.depth_tex = None

        for swapchain in self._xr_swapchains.values():
            try:
                xr.destroy_swapchain(swapchain)
            except Exception:
                pass
        self._xr_swapchains.clear()

        if self._xr_space:
            try:
                xr.destroy_space(self._xr_space)
            except Exception:
                pass
            self._xr_space = None

        if self._xr_session:
            try:
                xr.destroy_session(self._xr_session)
            except Exception:
                pass
            self._xr_session = None

        if self._xr_instance:
            try:
                xr.destroy_instance(self._xr_instance)
            except Exception:
                pass
            self._xr_instance = None

        if self.window:
            try:
                glfw.terminate()
            except Exception:
                pass
            self.window = None

        print("[OpenXRViewer] Cleanup complete")
