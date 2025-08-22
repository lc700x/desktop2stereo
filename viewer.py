# viewer.py
import glfw
import moderngl
import numpy as np

# Get OS name and settings
from utils import OS_NAME, IPD, crop_icon

VERTEX_SHADER = """
    #version 330
    in vec2 in_position;
    in vec2 in_uv;
    out vec2 uv;
    void main() {
        uv = in_uv;
        gl_Position = vec4(in_position, 0.0, 1.0);
    }
    """

FRAGMENT_SHADER = """
    #version 330
    in vec2 uv;
    out vec4 frag_color;

    uniform sampler2D tex_color;
    uniform sampler2D tex_depth;
    uniform float u_eye_offset;
    uniform float u_depth_strength;

    void main() {
        vec2 flipped_uv = vec2(uv.x, 1.0 - uv.y);
        float depth = texture(tex_depth, flipped_uv).r;

        // Invert depth so near=1 shifts more, far=0 shifts less
        float depth_inv = 1.0 - depth;

        vec2 offset_uv = flipped_uv - vec2(u_eye_offset * depth_inv * u_depth_strength, 0.0);
        offset_uv = clamp(offset_uv, 0.0, 1.0);

        frag_color = texture(tex_color, offset_uv);
    }
    """
def add_logo(window):
        from PIL  import Image
        glfw_img = Image.open("icon2.png")  # Path to your icon file
        if OS_NAME != "Darwin":
            glfw_img = crop_icon(glfw_img)
            glfw.set_window_icon(window, 1, [glfw_img])
class StereoWindow:
    """A window for displaying stereo images side-by-side with depth effect."""
    def __init__(self, depth_ratio=1.0, display_mode="Half-SBS"):
        self.window_size = (1280, 720)
        self.title = "Stereo SBS Viewer"
        self.ipd_uv = IPD  # Inter-pupillary distance in UV coordinates (0.064 per eye)
        self.depth_strength = 0.1  # Strength of depth effect
        self._last_window_position = None
        self._last_window_size = None
        self._fullscreen = False
        self.depth_ratio = depth_ratio
        self.depth_ratio_original = depth_ratio
        # all available modes (order used for cycling)
        self._modes = ["Full-SBS", "Half-SBS", "TAB"]
        self.display_mode = display_mode  # Default: Side-By-Side

        # Flag to track if textures need update
        self._texture_size = None
        
        # Initialize GLFW
        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW")
        
        # Configure window
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, True)
        
        # Create window (start in windowed mode)
        self.window = glfw.create_window(*self.window_size, self.title, None, None)
        add_logo(self.window)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Could not create window")
        
        # Position window on monitor 2 if available
        self.position_on_monitor(0)  # 0=primary, 1=secondary
        
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)  # Enable VSync to limit FPS and reduce CPU load
        self.ctx = moderngl.create_context()
        
        # Setup shaders and buffers
        self.prog = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER
        )
        self.quad_vao = self.make_quad()
        self.monitor_index = 0
        self.color_tex = None
        self.depth_tex = None
        
        # Set callbacks
        glfw.set_key_callback(self.window, self.on_key_event)

    def position_on_monitor(self, monitor_index=0):
        """Position window on specified monitor (0-based index)"""
        monitors = glfw.get_monitors()
        if monitor_index < len(monitors):
            monitor = monitors[monitor_index]
            mon_x, mon_y = glfw.get_monitor_pos(monitor)
            vidmode = glfw.get_video_mode(monitor)
            mon_w, mon_h = vidmode.size.width, vidmode.size.height

            x = mon_x + (mon_w - self.window_size[0]) // 2
            y = mon_y + (mon_h - self.window_size[1]) // 2
            glfw.set_window_pos(self.window, x, y)
            self.monitor_index = monitor_index  # Track current monitor

    def get_current_monitor(self):
        """Get the monitor that contains the window center"""
        monitors = glfw.get_monitors()
        if not monitors:
            return None
            
        # Get window center
        win_x, win_y = glfw.get_window_pos(self.window)
        win_w, win_h = glfw.get_window_size(self.window)
        window_center_x = win_x + win_w // 2
        window_center_y = win_y + win_h // 2
        
        # Find monitor containing window center
        for monitor in monitors:
            monitor_x, monitor_y = glfw.get_monitor_pos(monitor)
            vidmode = glfw.get_video_mode(monitor)
            if (monitor_x <= window_center_x < monitor_x + vidmode.size.width and
                monitor_y <= window_center_y < monitor_y + vidmode.size.height):
                return monitor
        return monitors[0]  # fallback to primary monitor
    def move_to_adjacent_monitor(self, direction):
        """
        Move window to adjacent monitor.
        direction: +1 for right, -1 for left.
        """
        monitors = glfw.get_monitors()
        if not monitors or len(monitors) <= 1:
            return  # Nothing to switch

        new_index = (self.monitor_index + direction) % len(monitors)
        self.position_on_monitor(new_index)
        
    def toggle_fullscreen(self):
        """Toggle fullscreen on current monitor"""
        current_monitor = self.get_current_monitor()
        if not current_monitor:
            return

        if not self._fullscreen:
            # Enter fullscreen mode, Remember current window state
            self._last_window_position = glfw.get_window_pos(self.window)
            self._last_window_size     = glfw.get_framebuffer_size(self.window)

            # Monitor geometry
            mon_x, mon_y = glfw.get_monitor_pos(current_monitor)
            vidmode      = glfw.get_video_mode(current_monitor)
            full_w, full_h = vidmode.size.width, vidmode.size.height

            # Make the window border-less + floating
            glfw.set_window_attrib(self.window, glfw.DECORATED, glfw.FALSE)
            glfw.set_window_attrib(self.window, glfw.FLOATING,  glfw.TRUE)

            # Resize & move so it exactly covers the monitor
            glfw.set_window_size(self.window, full_w, full_h)
            glfw.set_window_pos(self.window,  mon_x,  mon_y)

            self._fullscreen = True
        else:
            # Exit fullscreeen mode, Restore decorations / floating attribute
            glfw.set_window_attrib(self.window, glfw.DECORATED, glfw.TRUE)
            glfw.set_window_attrib(self.window, glfw.FLOATING,  glfw.FALSE)

            # Restore position & size (default: 1280×720 centred on same monitor)
            restore_w, restore_h = self.window_size
            if self._last_window_size:
                restore_w, restore_h = self._last_window_size

            if self._last_window_position:
                restore_x, restore_y = self._last_window_position
            else:
                # Centre it on the current monitor if we have no previous position
                vidmode   = glfw.get_video_mode(current_monitor)
                mon_x, mon_y = glfw.get_monitor_pos(current_monitor)
                restore_x = mon_x + (vidmode.size.width  - restore_w) // 2
                restore_y = mon_y + (vidmode.size.height - restore_h) // 2

            glfw.set_window_size(self.window, restore_w, restore_h)
            glfw.set_window_pos(self.window,  restore_x, restore_y)

            self._fullscreen = False
    
    def on_key_event(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_SPACE:
                self.toggle_fullscreen()
            elif key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
            elif key == glfw.KEY_RIGHT:
                self.move_to_adjacent_monitor(+1)
            elif key == glfw.KEY_LEFT:
                self.move_to_adjacent_monitor(-1)
            elif key == glfw.KEY_DOWN:
                # Decrease depth strength by 0.1
                self.depth_ratio = max(0, self.depth_ratio - 0.1)
            elif key == glfw.KEY_UP:
                # Increase depth strength by 0.1
                self.depth_ratio = min(10, self.depth_ratio + 0.1)
            elif key == glfw.KEY_0:
                # Reset depth strength to settings
                self.depth_ratio = self.depth_ratio_original
            elif key == glfw.KEY_TAB:
                # Cycle display mode: Full-SBS -> Half-SBS -> TAB -> Full-SBS
                idx = self._modes.index(self.display_mode)
                idx = (idx + 1) % len(self._modes)
                self.display_mode = self._modes[idx]



    def make_quad(self):
        vertices = np.array([
            -1, -1, 0, 0,
             1, -1, 1, 0,
            -1,  1, 0, 1,
             1,  1, 1, 1,
        ], dtype='f4')
        vbo = self.ctx.buffer(vertices)
        return self.ctx.vertex_array(
            self.prog, [(vbo, '2f 2f', 'in_position', 'in_uv')]
        )

    def update_frame(self, rgb, depth):
        depth = depth.detach().cpu().numpy().astype('float32')
        # Normalize depth with adaptive range
        depth_sampled = depth[::8, ::8]
        depth_min = np.quantile(depth_sampled, 0.2)
        depth_max = np.quantile(depth_sampled, 0.98)
        depth = (depth - depth_min) / (depth_max - depth_min + 1e-6)
        depth = np.clip(depth, 0, 1)
        # Only recreate textures if size changed
        h, w, _ = rgb.shape
        if self._texture_size != (w, h):
            if self.color_tex:
                self.color_tex.release()
            if self.depth_tex:
                self.depth_tex.release()

            self.color_tex = self.ctx.texture((w, h), 3, dtype='f1')
            self.depth_tex = self.ctx.texture((w, h), 1, dtype='f4')

            self.prog['tex_color'].value = 0
            self.prog['tex_depth'].value = 1

            self._texture_size = (w, h)

        # Upload texture data
        self.color_tex.write(rgb.astype('uint8').tobytes())
        self.depth_tex.write((self.depth_ratio * depth).astype('float32').tobytes())

    def render(self):
        if not self.color_tex or not self.depth_tex:
            return

        self.ctx.clear(0.1, 0.1, 0.1)
        win_w, win_h = glfw.get_framebuffer_size(self.window)  # width, height

        tex_w, tex_h = self._texture_size

        def compute_render_size(max_w, max_h, src_w, src_h):
            if src_w == 0 or src_h == 0:
                return 0, 0
            scale = min(max_w / src_w, max_h / src_h)
            rw = max(1, int(round(src_w * scale)))
            rh = max(1, int(round(src_h * scale)))
            return rw, rh

        self.color_tex.use(location=0)
        self.depth_tex.use(location=1)

        if self.display_mode == "Full-SBS":
            # Each view = tex_w x tex_h
            src_w, src_h = tex_w, tex_h
            max_w, max_h = win_w / 2.0, win_h
            render_w, render_h = compute_render_size(max_w, max_h, src_w, src_h)

            center_y = win_h / 2.0
            # Left
            cx = win_w / 4.0
            x = int(cx - render_w / 2)
            y = int(center_y - render_h / 2)
            self.ctx.viewport = (x, y, render_w, render_h)
            self.prog['u_eye_offset'].value = -self.ipd_uv / 2.0
            self.prog['u_depth_strength'].value = self.depth_strength
            self.quad_vao.render(moderngl.TRIANGLE_STRIP)

            # Right
            cx = 3 * win_w / 4.0
            x = int(cx - render_w / 2)
            y = int(center_y - render_h / 2)
            self.ctx.viewport = (x, y, render_w, render_h)
            self.prog['u_eye_offset'].value = self.ipd_uv / 2.0
            self.quad_vao.render(moderngl.TRIANGLE_STRIP)

        elif self.display_mode == "Half-SBS":
            # Each view = tex_w/2 x tex_h
            src_w, src_h = tex_w / 2.0, tex_h
            max_w, max_h = win_w / 2.0, win_h
            render_w, render_h = compute_render_size(max_w, max_h, src_w, src_h)

            center_y = win_h / 2.0
            # Left
            cx = win_w / 4.0
            x = int(cx - render_w / 2)
            y = int(center_y - render_h / 2)
            self.ctx.viewport = (x, y, render_w, render_h)
            self.prog['u_eye_offset'].value = -self.ipd_uv / 2.0
            self.prog['u_depth_strength'].value = self.depth_strength
            self.quad_vao.render(moderngl.TRIANGLE_STRIP)

            # Right
            cx = 3 * win_w / 4.0
            x = int(cx - render_w / 2)
            y = int(center_y - render_h / 2)
            self.ctx.viewport = (x, y, render_w, render_h)
            self.prog['u_eye_offset'].value = self.ipd_uv / 2.0
            self.quad_vao.render(moderngl.TRIANGLE_STRIP)

        elif self.display_mode == "TAB":
            # Each view = tex_w x tex_h/2
            src_w, src_h = tex_w, tex_h / 2.0
            max_w, max_h = win_w, win_h / 2.0
            render_w, render_h = compute_render_size(max_w, max_h, src_w, src_h)

            # Top
            cx, cy = win_w / 2.0, win_h / 4.0
            x = int(cx - render_w / 2)
            y = int(cy - render_h / 2)
            self.ctx.viewport = (x, y, render_w, render_h)
            self.prog['u_eye_offset'].value = -self.ipd_uv / 2.0
            self.prog['u_depth_strength'].value = self.depth_strength
            self.quad_vao.render(moderngl.TRIANGLE_STRIP)

            # Bottom
            cx, cy = win_w / 2.0, 3 * win_h / 4.0
            x = int(cx - render_w / 2)
            y = int(cy - render_h / 2)
            self.ctx.viewport = (x, y, render_w, render_h)
            self.prog['u_eye_offset'].value = self.ipd_uv / 2.0
            self.quad_vao.render(moderngl.TRIANGLE_STRIP)
