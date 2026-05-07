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
import collections

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
    glGenBuffers, glDeleteBuffers, glBindBuffer, glBufferData,
    glBindTexture, glTexSubImage2D,
    GL_PIXEL_UNPACK_BUFFER, GL_DYNAMIC_DRAW,
    GL_RGB, GL_RED, GL_UNSIGNED_BYTE, GL_FLOAT,
)

try:
    import xr
    OPENXR_AVAILABLE = True
except ImportError:
    OPENXR_AVAILABLE = False
    print("[OpenXRViewer] pyopenxr not installed. Run: pip install pyopenxr")

from viewer import FRAGMENT_SHADER, BACKEND
try:
    from viewer import CUDART_GL
except ImportError:
    CUDART_GL = None

# GL_SRGB8_ALPHA8: desktop captures are sRGB-encoded; signalling this to the
# compositor prevents it from treating gamma values as linear (which causes pale/washed-out colours).
_GL_SRGB8_ALPHA8 = 0x8C43

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

# World-space overlay fragment shader (plain RGBA texture, no parallax)
_OVERLAY_FRAG = """
#version 330
uniform sampler2D tex;
in vec2 uv;
out vec4 fragColor;
void main() {
    fragColor = texture(tex, uv);
}
"""

# Solid-color vertex shader (no UV — avoids GLSL optimizer stripping in_uv)
_SOLID_VERT = """
#version 330
in vec2 in_position;
uniform mat4 u_mvp;
void main() {
    gl_Position = u_mvp * vec4(in_position, 0.0, 1.0);
}
"""

# Solid-color fragment shader for the screen border quad
_SOLID_FRAG = """
#version 330
uniform vec4 u_color;
out vec4 fragColor;
void main() { fragColor = u_color; }
"""

# Cinema glow fragment shader — Gaussian radial falloff for screen-projection ambience
_GLOW_FRAG = """
#version 330
in vec2 uv;
uniform vec4 u_glow_color;   // rgb = warm white, a = peak opacity
out vec4 fragColor;
void main() {
    vec2 d   = uv * 2.0 - 1.0;          // map [0,1] → [-1,+1]
    float r  = length(d);
    float g  = exp(-r * r * 0.9);       // gaussian falloff; 0.9 fills wall further
    fragColor = vec4(u_glow_color.rgb, g * u_glow_color.a);
}
"""


# ---------------------------------------------------------------------------
# Windows input helpers (no-op on non-Windows)
# ---------------------------------------------------------------------------

if sys.platform == "win32":
    _U32 = ctypes.windll.user32

    class _MOUSEINPUT(ctypes.Structure):
        _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long),
                    ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong),
                    ("time", ctypes.c_ulong), ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]

    class _INPUT(ctypes.Structure):
        class _I(ctypes.Union):
            _fields_ = [("mi", _MOUSEINPUT)]
        _anonymous_ = ("_i",)
        _fields_ = [("type", ctypes.c_ulong), ("_i", _I)]

    _MOUSEEVENTF_MOVE     = 0x0001
    _MOUSEEVENTF_LEFTDOWN = 0x0002
    _MOUSEEVENTF_LEFTUP   = 0x0004
    _MOUSEEVENTF_RIGHTDOWN= 0x0008
    _MOUSEEVENTF_RIGHTUP  = 0x0010
    _MOUSEEVENTF_ABSOLUTE = 0x8000
    _KEYEVENTF_KEYUP      = 0x0002

    def _set_cursor_pos(x, y):
        _U32.SetCursorPos(int(x), int(y))

    def _send_mouse_flags(flags):
        inp = _INPUT(type=0)
        inp.mi.dwFlags = flags
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

    def _send_key(vk, shift=False, ctrl=False, alt=False, win=False):
        kbd = ctypes.windll.user32.keybd_event
        # Press modifiers (chord support: Ctrl+C, Alt+Tab, Win+R, etc.)
        if ctrl:  kbd(0x11, 0, 0, 0)             # VK_CONTROL down
        if shift: kbd(0x10, 0, 0, 0)             # VK_SHIFT down
        if alt:   kbd(0x12, 0, 0, 0)             # VK_MENU (Alt) down
        if win:   kbd(0x5B, 0, 0, 0)             # VK_LWIN down
        kbd(vk, 0, 0, 0)                          # key down
        kbd(vk, 0, _KEYEVENTF_KEYUP, 0)           # key up
        # Release modifiers in reverse
        if win:   kbd(0x5B, 0, _KEYEVENTF_KEYUP, 0)
        if alt:   kbd(0x12, 0, _KEYEVENTF_KEYUP, 0)
        if shift: kbd(0x10, 0, _KEYEVENTF_KEYUP, 0)
        if ctrl:  kbd(0x11, 0, _KEYEVENTF_KEYUP, 0)

    def _get_desktop_size():
        return _U32.GetSystemMetrics(0), _U32.GetSystemMetrics(1)
else:
    def _set_cursor_pos(x, y): pass
    def _send_mouse_flags(flags): pass
    def _send_key(vk, shift=False, ctrl=False, alt=False, win=False): pass
    def _get_desktop_size(): return (1920, 1080)
    _MOUSEEVENTF_LEFTDOWN  = 0x0002
    _MOUSEEVENTF_LEFTUP    = 0x0004
    _MOUSEEVENTF_RIGHTDOWN = 0x0008
    _MOUSEEVENTF_RIGHTUP   = 0x0010


# ---------------------------------------------------------------------------
# Virtual keyboard layout
# ---------------------------------------------------------------------------

# (label, normal_vk, _, shifted_vk, width_units)
# VK codes: https://docs.microsoft.com/en-us/windows/win32/inputdev/virtual-key-codes
# vk == -1 marks a layout gap: the slot consumes width but renders nothing and
# generates no keystroke (used to align the navigation/arrow clusters).
_KB_UNITS_WIDE = 18   # total horizontal units per row
_KB_ROWS = [
    # Row F: Esc + F1–F12 + PrtSc/ScrLk/Pause   (1.5 + 12 + 4.5 = 18)
    [('Esc',0x1B,None,0x1B,1.5),
     ('F1',0x70,None,0x70,1),('F2',0x71,None,0x71,1),
     ('F3',0x72,None,0x72,1),('F4',0x73,None,0x73,1),
     ('F5',0x74,None,0x74,1),('F6',0x75,None,0x75,1),
     ('F7',0x76,None,0x76,1),('F8',0x77,None,0x77,1),
     ('F9',0x78,None,0x78,1),('F10',0x79,None,0x79,1),
     ('F11',0x7A,None,0x7A,1),('F12',0x7B,None,0x7B,1),
     ('PrtSc',0x2C,None,0x2C,1.5),('ScrLk',0x91,None,0x91,1.5),('Pause',0x13,None,0x13,1.5)],
    # Row 0: number row + Ins/Hom/PgUp        (13 + 2 + 3 = 18)
    [('`',0xC0,'~',0xC0,1),('1',0x31,'!',0x31,1),('2',0x32,'@',0x32,1),
     ('3',0x33,'#',0x33,1),('4',0x34,'$',0x34,1),('5',0x35,'%',0x35,1),
     ('6',0x36,'^',0x36,1),('7',0x37,'&',0x37,1),('8',0x38,'*',0x38,1),
     ('9',0x39,'(',0x39,1),('0',0x30,')',0x30,1),('-',0xBD,'_',0xBD,1),
     ('=',0xBB,'+',0xBB,1),('Bksp',0x08,None,0x08,2),
     ('Ins',0x2D,None,0x2D,1),('Hom',0x24,None,0x24,1),('PgU',0x21,None,0x21,1)],
    # Row 1: QWERTY + Del/End/PgDn            (1.5 + 12 + 1.5 + 3 = 18)
    [('Tab',0x09,None,0x09,1.5),('Q',0x51,None,0x51,1),('W',0x57,None,0x57,1),
     ('E',0x45,None,0x45,1),('R',0x52,None,0x52,1),('T',0x54,None,0x54,1),
     ('Y',0x59,None,0x59,1),('U',0x55,None,0x55,1),('I',0x49,None,0x49,1),
     ('O',0x4F,None,0x4F,1),('P',0x50,None,0x50,1),('[',0xDB,'{',0xDB,1),
     (']',0xDD,'}',0xDD,1),('\\',0xDC,'|',0xDC,1.5),
     ('Del',0x2E,None,0x2E,1),('End',0x23,None,0x23,1),('PgD',0x22,None,0x22,1)],
    # Row 2: ASDF + 3-unit gap                (1.75 + 11 + 2.25 + 3 = 18)
    [('Caps',0x14,None,0x14,1.75),('A',0x41,None,0x41,1),('S',0x53,None,0x53,1),
     ('D',0x44,None,0x44,1),('F',0x46,None,0x46,1),('G',0x47,None,0x47,1),
     ('H',0x48,None,0x48,1),('J',0x4A,None,0x4A,1),('K',0x4B,None,0x4B,1),
     ('L',0x4C,None,0x4C,1),(';',0xBA,':',0xBA,1),("'",0xDE,'"',0xDE,1),
     ('Enter',0x0D,None,0x0D,2.25),
     ('',-1,None,-1,3)],
    # Row 3: ZXCV + Up arrow centred over Down
    # 2.25 + 11 + 1.75 = 15.0   then  gap(1) | ↑(1) | gap(1)  →  ↑ occupies col 16-17
    [('Shift',0x10,None,0x10,2.25),('Z',0x5A,None,0x5A,1),('X',0x58,None,0x58,1),
     ('C',0x43,None,0x43,1),('V',0x56,None,0x56,1),('B',0x42,None,0x42,1),
     ('N',0x4E,None,0x4E,1),('M',0x4D,None,0x4D,1),(',',0xBC,'<',0xBC,1),
     ('.',0xBE,'>',0xBE,1),('/',0xBF,'?',0xBF,1),('Shift',0x10,None,0x10,1.75),
     ('',-1,None,-1,1),('↑',0x26,None,0x26,1),('',-1,None,-1,1)],
    # Row 4: bottom + arrow cluster — Down sits directly under Up at col 16-17.
    # 1.5+1+1.25+7.5+1.25+1+1.5 = 15.0   then  ←(1) | ↓(1) | →(1)
    [('Ctrl',0x11,None,0x11,1.5),('Win',0x5B,None,0x5B,1),
     ('Alt',0x12,None,0x12,1.25),
     ('Space',0x20,None,0x20,7.5),
     ('Alt',0x12,None,0x12,1.25),('Apps',0x5D,None,0x5D,1),
     ('Ctrl',0x11,None,0x11,1.5),
     ('←',0x25,None,0x25,1),('↓',0x28,None,0x28,1),('→',0x27,None,0x27,1)],
]

import collections as _collections
_KeyEntry = _collections.namedtuple('_KeyEntry', 'label vk shifted_vk rect_uv rect_local')

_KB_TEX_W, _KB_TEX_H = 1280, 384   # keyboard texture: 6 rows × 18 units


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
    The screen can be repositioned/resized/rotated via keyboard or VR controller
    thumbsticks. An in-VR FPS/latency overlay quad sits just below the screen and
    is toggled with the menu button on the left controller.

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

        # FPS display — timestamp ring: (len-1)/(last-first) is exact over the window
        self.actual_fps      = 0.0   # XR composition rate (this loop)
        self.sbs_fps         = 0.0   # SBS source rate (depth_q producer in main.py)
        self.total_latency   = 0.0
        self._frame_ts_ring  = collections.deque(maxlen=60)  # ~1 s at 60 Hz
        self._sbs_ts_ring    = collections.deque(maxlen=60)  # SBS frame arrivals

        # Add these new variables for display throttling
        self._last_overlay_update = 0.0
        self._cached_actual_fps = 0.0
        self._cached_sbs_fps = 0.0
        self._cached_latency = 0.0

        # Virtual screen transform (world space, metres / radians)
        self.screen_distance = 2.0
        self.screen_width    = 2.0
        self.screen_height   = None   # derived from frame aspect ratio on first frame
        self.screen_pan_x    = 0.0
        self.screen_pan_y    = 0.0
        self.screen_yaw      = 0.0    # rotation around Y axis
        self.screen_pitch    = 0.0    # rotation around X axis

        # Interaction speeds (per second)
        self._dist_speed  = 0.5
        self._size_speed  = 0.5
        self._pan_speed   = 0.5
        self._rot_speed   = 0.5
        self._pitch_speed = 0.5

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
        self._action_set          = None
        self._act_left_stick      = None
        self._act_right_stick     = None
        self._act_menu_btn        = None
        self._act_left_grip       = None   # grab/move mode
        self._act_right_grip      = None   # resize mode
        self._act_a_btn           = None   # right A — double press = hide all
        self._act_b_btn           = None   # right B — right mouse click
        self._act_y_btn           = None   # left  Y — reset screen position/size/rotation
        self._act_left_trigger       = None   # left trigger (float) — left mouse click
        self._act_right_trigger      = None   # right trigger (float) — right mouse click / hold
        self._act_left_stick_click   = None   # left thumbstick click — toggle keyboard
        self._act_right_stick_click  = None   # right thumbstick click — toggle cinema glow

        # Menu button debounce + FPS overlay toggle
        self._menu_pressed_last   = False
        self._fps_overlay_visible = show_fps

        # Quest-like window state
        self._screen_visible = True   # hide-all toggle (A double-press)
        self._grabbed        = False  # left grip held  → move mode
        self._resizing       = False  # right grip held → resize mode
        self._any_grip_last  = False  # any grip held last frame (grab init debounce)
        self._a_last         = False  # A-button previous frame state
        self._a_last_t       = 0.0   # timestamp of last A press (double-press detection)
        self._b_last         = False  # B-button previous frame state
        self._y_last         = False  # Y-button previous frame state (reset screen)
        # Grab tracking — anchor = laser hit point on the screen plane at grab start.
        # The anchor sits at fixed distance along the laser; screen-local offset of
        # anchor from centre is stored so the screen translates 1:1 with the laser.
        self._grab_ray_dist      = 1.5    # distance along aim ray to anchor (m) — push/pull only
        self._grab_offset_x      = 0.0    # screen-local X offset of anchor from centre (m)
        self._grab_offset_y      = 0.0    # screen-local Y offset of anchor from centre (m)
        self._grab_ctrl_pos_prev = None   # controller world pos last grab frame (delta tracking)
        self._grab_anchor_world  = None   # anchor point in world space at grab start
        self._screen_eye_init = False  # screen_pan_y aligned to headset height on first frame
        self._initial_head_y  = 0.0   # headset eye height at session start, used for B-reset
        # Border fade: shown during interaction, fades out when idle
        self._border_alpha   = 0.0    # 0.0 = invisible, 1.0 = fully opaque
        self._border_idle_t  = 0.0    # wall time when interaction last ended

        # Mouse cursor control
        self._cursor_uv_l         = None  # (u,v) where left laser hits screen, or None
        self._cursor_uv_r         = None  # (u,v) where right laser hits screen, or None
        self._cursor_ctrl         = None  # 'left' | 'right' | None — active cursor controller
        self._left_trig_prev      = 0.0
        self._right_trig_prev     = 0.0
        self._left_trig_hold_t    = None  # perf_counter() when left trigger passed threshold
        self._right_trig_hold_t   = None  # perf_counter() when right trigger passed threshold
        self._left_btn_down       = False # whether right mouse button held via left trigger
        self._right_btn_down      = False # whether right mouse button is currently held

        # Virtual keyboard
        self._keyboard_visible     = False
        self._keyboard_tex         = None  # moderngl Texture (RGBA, _KB_TEX_W × _KB_TEX_H)
        self._keyboard_vao         = None  # quad VAO using _overlay_prog
        self._keyboard_keys        = []    # list of _KeyEntry
        self._keyboard_width       = 1.5   # metres
        self._keyboard_height      = 0.5   # metres (6 rows)
        self._keyboard_distance    = 1.5   # metres in front of user origin
        self._keyboard_pan_y       = -0.2  # metres below eye level
        self._shift_active         = False
        self._caps_lock            = False
        self._ctrl_active          = False   # one-shot Ctrl modifier (Ctrl+C etc.)
        self._alt_active           = False   # one-shot Alt modifier  (Alt+Tab etc.)
        self._win_active           = False   # one-shot Win modifier  (Win+R etc.)
        self._left_stick_click_prev= False
        self._kb_trig_prev_l       = 0.0   # keyboard trigger debounce — left controller
        self._kb_trig_prev_r       = 0.0   # keyboard trigger debounce — right controller
        self._kb_hover_l           = None  # index of key under left laser, or None
        self._kb_hover_r           = None  # index of key under right laser, or None

        # GPU interop (CUDA / HIP) — initialised lazily on first frame
        self._cuda_gl         = None   # CUDART_GL instance, False = permanently failed
        self._pbo_color       = None   # GL PBO id for RGB upload
        self._pbo_depth       = None   # GL PBO id for depth upload
        self._cuda_res_color  = None   # registered resource handle
        self._cuda_res_depth  = None
        self._pbo_texture_size = None  # (w, h) at which PBOs were created

        # FPS display — timestamp ring: (len-1)/(last-first) is exact over the window
        self.actual_fps      = 0.0   # XR composition rate (this loop)
        self.sbs_fps         = 0.0   # SBS source rate (depth_q producer in main.py)
        self.total_latency   = 0.0
        self._frame_ts_ring  = collections.deque(maxlen=60)  # ~1 s at 60 Hz
        self._sbs_ts_ring    = collections.deque(maxlen=60)  # SBS frame arrivals

        # Font for in-VR overlay
        self.font = None
        self.font_type = get_font_type()
        self.base_font_size = 22
        try:
            self.font = ImageFont.truetype(self.font_type, self.base_font_size)
        except Exception:
            try:
                self.font = ImageFont.load_default()
            except Exception:
                self.font = None

        # In-VR FPS overlay GL resources
        self._overlay_prog     = None
        self._overlay_vao      = None
        self._overlay_tex      = None
        self._overlay_tex_size = (512, 80)   # two lines: FPS+res / latency

        # Screen border (slightly larger quad, solid color)
        self._border_prog = None
        self._border_vao  = None

        # Cinema glow (large Gaussian quad behind the screen, additive blend)
        self._glow_prog              = None
        self._glow_vao               = None
        self._glow_visible           = False
        self._right_stick_click_prev = False

        # Controller aim poses + laser pointer rendering
        self._act_aim_left  = None   # XrAction POSE_INPUT for left aim
        self._act_aim_right = None   # XrAction POSE_INPUT for right aim
        self._aim_space_l   = None   # XrSpace for left aim
        self._aim_space_r   = None   # XrSpace for right aim
        self._laser_vao     = None   # thin quad for laser beam
        self._dot_vao       = None   # small square for controller dot
        # Cached aim poses updated each frame (numpy 4x4 view-space matrices)
        self._aim_mat_l     = None
        self._aim_mat_r     = None

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
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)   # hidden — GL context only
        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
        self.window = glfw.create_window(1, 1, "D2S-XR", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("[OpenXRViewer] GLFW window creation failed")
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)

        # Keyboard controls — keep a reference so it isn't GC'd
        self._key_callback_ref = self._make_key_callback()
        glfw.set_key_callback(self.window, self._key_callback_ref)

    def _make_key_callback(self):
        viewer = self
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
            elif key == glfw.KEY_T: viewer.screen_pitch += r
            elif key == glfw.KEY_G: viewer.screen_pitch -= r
            elif key == glfw.KEY_F: viewer._fps_overlay_visible = not viewer._fps_overlay_visible
            elif key == glfw.KEY_R:
                viewer.screen_distance = 2.0; viewer.screen_pan_x = 0.0
                viewer.screen_pan_y = 0.0;    viewer.screen_yaw = 0.0
                viewer.screen_pitch = 0.0;    viewer.screen_width = 2.0
                viewer.screen_height = None
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

        # Screen border (solid-color quad rendered before the main screen)
        self._border_prog = self.ctx.program(
            vertex_shader=_SOLID_VERT,
            fragment_shader=_SOLID_FRAG,
        )
        vbo_border = self.ctx.buffer(vertices.tobytes())
        self._border_vao = self.ctx.vertex_array(
            self._border_prog, [(vbo_border, '2f 8x', 'in_position')]
        )

        # Laser beam: a very thin elongated quad (width=0.003 m, length=5 m)
        # in local space X=[-0.5,0.5], Y=[-1,1]; we scale X to beam_w, Y to half-length
        laser_verts = np.array([
            -1, -1, 0, 0,
             1, -1, 1, 0,
            -1,  1, 0, 1,
             1,  1, 1, 1,
        ], dtype='f4')
        self._laser_vao = self.ctx.vertex_array(
            self._border_prog,
            [(self.ctx.buffer(laser_verts.tobytes()), '2f 8x', 'in_position')],
        )
        # Controller dot: tiny square (5×5 cm)
        self._dot_vao = self.ctx.vertex_array(
            self._border_prog,
            [(self.ctx.buffer(laser_verts.tobytes()), '2f 8x', 'in_position')],
        )

        # In-VR FPS overlay program (world-space quad, plain RGBA blit)
        self._overlay_prog = self.ctx.program(
            vertex_shader=_WORLD_VERT,
            fragment_shader=_OVERLAY_FRAG,
        )
        self._overlay_prog['tex'].value = 2   # texture unit 2
        ow, oh = self._overlay_tex_size
        self._overlay_tex = self.ctx.texture((ow, oh), 4, dtype='f1')
        self._overlay_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        vbo2 = self.ctx.buffer(vertices.tobytes())
        self._overlay_vao = self.ctx.vertex_array(
            self._overlay_prog, [(vbo2, '2f 2f', 'in_position', 'in_uv')]
        )

        # Cinema glow program (WORLD_VERT passes UV; GLOW_FRAG computes Gaussian)
        self._glow_prog = self.ctx.program(
            vertex_shader=_WORLD_VERT,
            fragment_shader=_GLOW_FRAG,
        )
        self._glow_prog['u_glow_color'].value = (1.0, 0.92, 0.75, 0.70)  # warm cinema amber
        vbo_glow = self.ctx.buffer(vertices.tobytes())
        self._glow_vao = self.ctx.vertex_array(
            self._glow_prog, [(vbo_glow, '2f 2f', 'in_position', 'in_uv')]
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
        _pfn = ctypes.cast(
            xr.get_instance_proc_addr(self._xr_instance, "xrGetOpenGLGraphicsRequirementsKHR"),
            xr.PFN_xrGetOpenGLGraphicsRequirementsKHR,
        )
        _reqs = xr.GraphicsRequirementsOpenGLKHR()
        xr.check_result(xr.Result(_pfn(self._xr_instance, self._xr_system_id, ctypes.byref(_reqs))))

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
                format=_GL_SRGB8_ALPHA8,
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
        """Set up OpenXR action set with thumbstick and menu button actions."""
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

        def make_bool(name, label):
            return xr.create_action(
                self._action_set,
                xr.ActionCreateInfo(
                    action_type=xr.ActionType.BOOLEAN_INPUT,
                    action_name=name,
                    localized_action_name=label,
                    count_subaction_paths=len(subpaths),
                    subaction_paths=subpaths,
                ),
            )

        self._act_menu_btn  = make_bool("menu_btn",   "Menu Button")
        self._act_left_grip = make_bool("left_grip",  "Left Grip")
        self._act_right_grip= make_bool("right_grip", "Right Grip")
        self._act_a_btn     = make_bool("a_btn",      "A Button")
        self._act_b_btn     = make_bool("b_btn",      "B Button")
        self._act_y_btn     = make_bool("y_btn",      "Y Button")
        self._act_left_stick_click  = make_bool("left_stick_click",  "Left Stick Click")
        self._act_right_stick_click = make_bool("right_stick_click", "Right Stick Click")

        def make_float(name, label):
            return xr.create_action(
                self._action_set,
                xr.ActionCreateInfo(
                    action_type=xr.ActionType.FLOAT_INPUT,
                    action_name=name,
                    localized_action_name=label,
                    count_subaction_paths=len(subpaths),
                    subaction_paths=subpaths,
                ),
            )

        self._act_left_trigger  = make_float("left_trigger",  "Left Trigger")
        self._act_right_trigger = make_float("right_trigger", "Right Trigger")

        self._act_aim_left = xr.create_action(
            self._action_set,
            xr.ActionCreateInfo(
                action_type=xr.ActionType.POSE_INPUT,
                action_name="aim_left",
                localized_action_name="Left Aim Pose",
                count_subaction_paths=1,
                subaction_paths=[subpaths[0]],
            ),
        )
        self._act_aim_right = xr.create_action(
            self._action_set,
            xr.ActionCreateInfo(
                action_type=xr.ActionType.POSE_INPUT,
                action_name="aim_right",
                localized_action_name="Right Aim Pose",
                count_subaction_paths=1,
                subaction_paths=[subpaths[1]],
            ),
        )

        # Per-profile binding table.
        # Use squeeze/value (float path) for grip — the runtime auto-thresholds it
        # for BOOLEAN_INPUT actions, and it works on more firmware versions than
        # squeeze/click (which requires a discrete click event on some runtimes).
        _b = {
            "/interaction_profiles/oculus/touch_controller": [
                ("/user/hand/left/input/thumbstick",         self._act_left_stick),
                ("/user/hand/right/input/thumbstick",        self._act_right_stick),
                ("/user/hand/left/input/thumbstick/click",   self._act_left_stick_click),
                ("/user/hand/right/input/thumbstick/click",  self._act_right_stick_click),
                ("/user/hand/left/input/menu/click",         self._act_menu_btn),
                ("/user/hand/left/input/squeeze/value",      self._act_left_grip),
                ("/user/hand/right/input/squeeze/value",     self._act_right_grip),
                ("/user/hand/right/input/a/click",           self._act_a_btn),
                ("/user/hand/right/input/b/click",           self._act_b_btn),
                ("/user/hand/left/input/y/click",            self._act_y_btn),
                ("/user/hand/left/input/trigger/value",      self._act_left_trigger),
                ("/user/hand/right/input/trigger/value",     self._act_right_trigger),
                ("/user/hand/left/input/aim/pose",           self._act_aim_left),
                ("/user/hand/right/input/aim/pose",          self._act_aim_right),
            ],
            "/interaction_profiles/valve/index_controller": [
                ("/user/hand/left/input/thumbstick",         self._act_left_stick),
                ("/user/hand/right/input/thumbstick",        self._act_right_stick),
                ("/user/hand/left/input/thumbstick/click",   self._act_left_stick_click),
                ("/user/hand/right/input/thumbstick/click",  self._act_right_stick_click),
                ("/user/hand/left/input/trackpad/click",     self._act_menu_btn),
                ("/user/hand/left/input/squeeze/value",      self._act_left_grip),
                ("/user/hand/right/input/squeeze/value",     self._act_right_grip),
                ("/user/hand/right/input/a/click",           self._act_a_btn),
                ("/user/hand/right/input/b/click",           self._act_b_btn),
                ("/user/hand/left/input/trigger/value",      self._act_left_trigger),
                ("/user/hand/right/input/trigger/value",     self._act_right_trigger),
                ("/user/hand/left/input/aim/pose",           self._act_aim_left),
                ("/user/hand/right/input/aim/pose",         self._act_aim_right),
            ],
            # KHR simple only has select/click (boolean) and menu — no sticks or grip
            "/interaction_profiles/khr/simple_controller": [
                ("/user/hand/left/input/menu/click",    self._act_menu_btn),
                ("/user/hand/left/input/aim/pose",      self._act_aim_left),
                ("/user/hand/right/input/aim/pose",     self._act_aim_right),
            ],
        }

        for profile, pairs in _b.items():
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
                            for path, act in pairs
                        ],
                    ),
                )
            except Exception:
                pass

        xr.attach_session_action_sets(
            self._xr_session,
            xr.SessionActionSetsAttachInfo(action_sets=[self._action_set]),
        )

        # Create action spaces for aim poses (used to locate controller each frame)
        for act, attr in [
            (self._act_aim_left,  "_aim_space_l"),
            (self._act_aim_right, "_aim_space_r"),
        ]:
            try:
                space = xr.create_action_space(
                    self._xr_session,
                    xr.ActionSpaceCreateInfo(
                        action=act,
                        pose_in_action_space=xr.Posef(),
                    ),
                )
                setattr(self, attr, space)
            except Exception as e:
                print(f"[OpenXRViewer] Aim space creation failed: {e}")

    def _init_textures(self, w, h):
        if self.color_tex:
            self.color_tex.release()
        if self.depth_tex:
            self.depth_tex.release()
        self.color_tex = self.ctx.texture((w, h), 3, dtype='f1')
        self.depth_tex = self.ctx.texture((w, h), 1, dtype='f4')
        self._texture_size = (w, h)

    def _init_keyboard(self):
        """Build keyboard key geometry and render the keyboard texture via PIL."""
        TW, TH   = _KB_TEX_W, _KB_TEX_H
        ROW_H    = TH / len(_KB_ROWS)                  # pixel height per row
        UNIT_W   = TW / float(_KB_UNITS_WIDE)          # pixel width of one key unit
        UNIT_M   = self._keyboard_width / float(_KB_UNITS_WIDE)
        PAD      = 3                                   # pixel inset for key face

        img  = Image.new('RGBA', (TW, TH), (30, 30, 35, 230))
        draw = ImageDraw.Draw(img)
        # Try a font that contains arrow / shift / enter symbols. Segoe UI Symbol
        # ships on every modern Windows install and covers the Unicode keyboard
        # glyphs; fall back to the app font, then PIL's bitmap default.
        fnt = None
        for candidate in (r"C:\Windows\Fonts\seguisym.ttf",
                          r"C:\Windows\Fonts\segoeui.ttf",
                          self.font_type):
            if not candidate:
                continue
            try:
                fnt = ImageFont.truetype(candidate, 16)
                break
            except Exception:
                continue

        self._keyboard_keys = []
        kw_half  = self._keyboard_width  / 2.0
        kh_half  = self._keyboard_height / 2.0
        row_h_m  = self._keyboard_height / len(_KB_ROWS)  # metres per row

        for row_i, row in enumerate(_KB_ROWS):
            py0 = int(row_i * ROW_H)
            py1 = int((row_i + 1) * ROW_H)
            # Local Y of this row: top of keyboard = +kh_half, rows go downward
            ly1 = kh_half - row_i * row_h_m
            ly0 = ly1 - row_h_m
            px  = 0.0
            lx  = -kw_half
            for (label, vk_normal, _, vk_shifted, width_units) in row:
                px_end  = px + width_units * UNIT_W
                lx_end  = lx + width_units * UNIT_M

                # vk == -1 marks a layout gap: advance position but do not draw
                # the key face and do not register a hit target.
                if vk_normal == -1:
                    px = px_end
                    lx = lx_end
                    continue

                # Draw key background and outline
                draw.rectangle([px + PAD, py0 + PAD, px_end - PAD, py1 - PAD],
                               fill=(60, 62, 70, 255), outline=(130, 132, 140, 255))

                # Draw label centred in key face
                if fnt:
                    tx = (px + px_end) / 2
                    ty = (py0 + py1) / 2
                    draw.text((tx, ty), label, font=fnt, fill=(220, 220, 225, 255), anchor='mm')
                else:
                    draw.text((int(px + PAD + 2), int(py0 + PAD + 2)), label,
                               fill=(220, 220, 225, 255))

                # UV rect in [0,1]
                uv_rect = (px / TW, py0 / TH, px_end / TW, py1 / TH)
                # Local rect in metres (x grows right, y grows up)
                loc_rect = (lx, ly0, lx_end, ly1)

                self._keyboard_keys.append(_KeyEntry(
                    label=label,
                    vk=vk_normal,
                    shifted_vk=vk_shifted if vk_shifted is not None else vk_normal,
                    rect_uv=uv_rect,
                    rect_local=loc_rect,
                ))

                px  = px_end
                lx  = lx_end

        # Upload texture
        tex_data = np.flipud(np.array(img, dtype=np.uint8))
        if self._keyboard_tex is not None:
            self._keyboard_tex.release()
        self._keyboard_tex = self.ctx.texture((TW, TH), 4, dtype='f1')
        self._keyboard_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._keyboard_tex.write(tex_data.tobytes())

        # VAO — reuse _overlay_prog (already compiled, same WORLD_VERT + OVERLAY_FRAG)
        verts = np.array([-1,-1,0,0, 1,-1,1,0, -1,1,0,1, 1,1,1,1], dtype='f4')
        if self._keyboard_vao is not None:
            self._keyboard_vao.release()
        self._keyboard_vao = self.ctx.vertex_array(
            self._overlay_prog, [(self.ctx.buffer(verts.tobytes()), '2f 2f', 'in_position', 'in_uv')]
        )

    def _render_keyboard(self, mgl_fbo, view_mat, proj_mat):
        """Render the virtual keyboard quad and highlight hovered keys."""
        if self._keyboard_tex is None or self._keyboard_vao is None:
            return

        kw2 = self._keyboard_width  / 2.0
        kh2 = self._keyboard_height / 2.0
        kb_model = np.array([
            [kw2, 0,   0, 0.0                  ],
            [0,   kh2, 0, self._keyboard_pan_y  ],
            [0,   0,   1, -self._keyboard_distance],
            [0,   0,   0, 1                     ],
        ], dtype=np.float32)
        mvp = proj_mat @ view_mat @ kb_model

        mgl_fbo.use()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._keyboard_tex.use(location=2)
        self._overlay_prog['u_mvp'].write(mvp.T.astype('f4').tobytes())
        self._keyboard_vao.render(moderngl.TRIANGLE_STRIP)

        def _hl_quad(rect_local, color):
            x0, y0, x1, y1 = rect_local
            cx = (x0 + x1) / 2.0; cy_ = (y0 + y1) / 2.0
            hw = (x1 - x0) / 2.0; hh  = (y1 - y0) / 2.0
            hl_model = np.array([
                [hw,  0,   0, cx                                  ],
                [0,   hh,  0, self._keyboard_pan_y + cy_           ],
                [0,   0,   1, -self._keyboard_distance + 0.001     ],
                [0,   0,   0, 1                                   ],
            ], dtype=np.float32)
            hl_mvp = proj_mat @ view_mat @ hl_model
            self._border_prog['u_mvp'].write(hl_mvp.T.astype('f4').tobytes())
            self._border_prog['u_color'].value = color
            self._border_vao.render(moderngl.TRIANGLE_STRIP)

        # Amber highlight on every key whose VK matches an armed modifier
        VK_SHIFT = 0x10; VK_CAPS = 0x14; VK_CTRL = 0x11; VK_ALT = 0x12; VK_WIN = 0x5B
        active_vks = set()
        if self._shift_active: active_vks.add(VK_SHIFT)
        if self._caps_lock:    active_vks.add(VK_CAPS)
        if self._ctrl_active:  active_vks.add(VK_CTRL)
        if self._alt_active:   active_vks.add(VK_ALT)
        if self._win_active:   active_vks.add(VK_WIN)
        if active_vks:
            for key in self._keyboard_keys:
                if key.vk in active_vks:
                    _hl_quad(key.rect_local, (1.0, 0.7, 0.15, 0.45))

        # Cyan highlight on keys hovered by either laser
        for hover_idx in set(x for x in [self._kb_hover_l, self._kb_hover_r] if x is not None):
            _hl_quad(self._keyboard_keys[hover_idx].rect_local, (0.2, 0.7, 1.0, 0.35))

        self.ctx.disable(moderngl.BLEND)

    def _init_cuda_pbos(self, w, h):
        """Create or recreate PBOs and register them with CUDA/HIP."""
        if not self._cuda_gl or BACKEND not in ("CUDA", "HIP"):
            return
        # Unregister old resources before deleting PBOs
        if self._pbo_color is not None:
            try:
                self._cuda_gl.unregister_resource(self._cuda_res_color)
                self._cuda_gl.unregister_resource(self._cuda_res_depth)
                glDeleteBuffers(2, [self._pbo_color, self._pbo_depth])
            except Exception:
                pass

        ids = glGenBuffers(2)
        self._pbo_color = int(ids[0])
        self._pbo_depth = int(ids[1])

        for pbo_id, nbytes in [
            (self._pbo_color, w * h * 3),   # RGB uint8
            (self._pbo_depth, w * h * 4),   # float32
        ]:
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_id)
            glBufferData(GL_PIXEL_UNPACK_BUFFER, nbytes, None, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

        self._cuda_res_color = self._cuda_gl.register_buffer(self._pbo_color)
        self._cuda_res_depth = self._cuda_gl.register_buffer(self._pbo_depth)
        self._pbo_texture_size = (w, h)
        print(f"[OpenXRViewer] GPU interop PBOs created ({BACKEND}) {w}x{h}")

    # ------------------------------------------------------------------
    # Per-frame helpers
    # ------------------------------------------------------------------

    def _update_frame(self, rgb, depth):
        """Upload RGB and depth to GL textures — GPU path when available, CPU fallback."""
        import torch

        is_tensor = hasattr(rgb, 'data_ptr')

        # Resolve depth shape and GPU tensor
        if hasattr(depth, 'detach'):
            depth_gpu = depth.detach().contiguous().float()
            h, w = depth_gpu.shape[0], depth_gpu.shape[1]
            depth_np = None
        else:
            depth_gpu = None
            depth_np = np.asarray(depth, dtype=np.float32)
            h, w = depth_np.shape[0], depth_np.shape[1]

        if self._texture_size != (w, h):
            self._init_textures(w, h)
            self.frame_size = (w, h)
            self.screen_height = None

        # Lazy GPU interop init
        if self._cuda_gl is None and CUDART_GL is not None and BACKEND in ("CUDA", "HIP"):
            try:
                self._cuda_gl = CUDART_GL()
                print(f"[OpenXRViewer] GPU interop active ({BACKEND})")
            except Exception as e:
                print(f"[OpenXRViewer] GPU interop unavailable: {e}")
                self._cuda_gl = False   # sentinel: don't retry

        gpu_ok = bool(self._cuda_gl) and is_tensor and BACKEND in ("CUDA", "HIP")

        if gpu_ok:
            if self._pbo_texture_size != (w, h):
                self._init_cuda_pbos(w, h)

            # Color: CHW tensor → HWC contiguous uint8 on GPU, DMA into PBO
            rgb_gpu = rgb.permute(1, 2, 0).contiguous().clamp(0, 255).to(torch.uint8)
            ptr = self._cuda_gl.map_resource(self._cuda_res_color)
            self._cuda_gl.memcpy_d2d(ptr, rgb_gpu.data_ptr(), rgb_gpu.nbytes)
            self._cuda_gl.unmap_resource(self._cuda_res_color)
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self._pbo_color)
            glBindTexture(GL_TEXTURE_2D, self.color_tex.glo)
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
            glBindTexture(GL_TEXTURE_2D, 0)

            # Depth: float tensor on GPU → DMA into PBO
            ptr = self._cuda_gl.map_resource(self._cuda_res_depth)
            self._cuda_gl.memcpy_d2d(ptr, depth_gpu.data_ptr(), depth_gpu.nbytes)
            self._cuda_gl.unmap_resource(self._cuda_res_depth)
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self._pbo_depth)
            glBindTexture(GL_TEXTURE_2D, self.depth_tex.glo)
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RED, GL_FLOAT, ctypes.c_void_p(0))
            glBindTexture(GL_TEXTURE_2D, 0)
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        else:
            # CPU fallback
            if hasattr(rgb, 'detach'):
                rgb_np = (
                    rgb.permute(1, 2, 0).detach().contiguous()
                    .clamp(0, 255).to(torch.uint8).cpu().numpy()
                )
            else:
                rgb_np = np.asarray(rgb, dtype=np.uint8)
            if depth_np is None:
                depth_np = depth_gpu.cpu().numpy()
            self.color_tex.write(rgb_np.astype('uint8', copy=False).tobytes())
            self.depth_tex.write(depth_np.tobytes())

    def _build_model_mat4(self):
        """Construct the screen quad's world-space model matrix (math row/col convention).

        The quad's in_position spans [-1,+1] in X and Y in model space.
        We scale to physical metres, translate to (pan_x, pan_y, -distance) in world
        space (OpenXR: right-hand Y-up, forward = -Z), then apply pitch and yaw.
        Caller must transpose before writing to OpenGL.
        """
        if self.screen_height is None:
            fw, fh = self.frame_size
            self.screen_height = self.screen_width * (fh / fw if fw > 0 else 9 / 16)

        sx  = self.screen_width  / 2.0
        sy  = self.screen_height / 2.0

        cy  = math.cos(self.screen_yaw)
        sy_ = math.sin(self.screen_yaw)
        rot_y = np.array([
            [ cy,  0, sy_, 0],
            [  0,  1,   0, 0],
            [-sy_, 0,  cy, 0],
            [  0,  0,   0, 1],
        ], dtype=np.float32)

        cp  = math.cos(self.screen_pitch)
        sp  = math.sin(self.screen_pitch)
        rot_x = np.array([
            [1,  0,   0, 0],
            [0,  cp, -sp, 0],
            [0,  sp,  cp, 0],
            [0,  0,   0,  1],
        ], dtype=np.float32)

        # Scale + translate: translation in last column, row 3 = [0,0,0,1]
        model = np.array([
            [sx,  0,  0, self.screen_pan_x    ],
            [ 0, sy,  0, self.screen_pan_y    ],
            [ 0,  0,  1, -self.screen_distance],
            [ 0,  0,  0, 1                    ],
        ], dtype=np.float32)

        return rot_y @ rot_x @ model

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

    def _render_fps_overlay(self, eye_index, mgl_fbo, view_mat, proj_mat):
        """Render the FPS/latency text quad just below the virtual screen (in VR)."""
        if self.screen_height is None:
            return

        now = time.perf_counter()
        
        # Update cached values once per second
        if now - self._last_overlay_update >= 1.0:
            self._cached_actual_fps = self.actual_fps
            self._cached_sbs_fps = self.sbs_fps
            self._cached_latency = self.total_latency
            self._last_overlay_update = now
            
            # Rebuild text texture once per stereo pair (left eye = index 0)
            if eye_index == 0 and self.font is not None:
                ow, oh = self._overlay_tex_size
                img = Image.new('RGBA', (ow, oh), (0, 0, 0, 180))
                draw = ImageDraw.Draw(img)
                line1 = f"[FPS] XR:{self._cached_actual_fps:5.1f}   SBS:{self._cached_sbs_fps:5.1f}"
                # Pull live capture/resize/depth latencies from main.py's shared dict
                if self._cached_latency > 0:
                    line2 = f"[Latency] {self._cached_latency:.0f} ms"
                else:
                    line2 = "[Latency] —"
                draw.text((8,  6), line1, font=self.font, fill=(0, 255, 100, 255))
                draw.text((8, 34), line2, font=self.font, fill=(0, 220, 220, 255))
                data = np.flipud(np.array(img, dtype=np.uint8))
                self._overlay_tex.write(data.tobytes())

        GAP       = 0.05
        OVERLAY_H = 0.09
        ow, oh    = self._overlay_tex_size
        OVERLAY_W = OVERLAY_H * (ow / oh)   # preserve texture pixel aspect (512/80 ≈ 6.4)

        cy  = math.cos(self.screen_yaw);   sy_ = math.sin(self.screen_yaw)
        cp  = math.cos(self.screen_pitch); sp  = math.sin(self.screen_pitch)
        rot_y = np.array([
            [ cy,  0, sy_, 0],
            [  0,  1,   0, 0],
            [-sy_, 0,  cy, 0],
            [  0,  0,   0, 1],
        ], dtype=np.float32)
        rot_x = np.array([
            [1,  0,   0, 0],
            [0,  cp, -sp, 0],
            [0,  sp,  cp, 0],
            [0,  0,   0,  1],
        ], dtype=np.float32)

        y_off = self.screen_pan_y - self.screen_height / 2.0 - GAP - OVERLAY_H / 2.0
        overlay_model = np.array([
            [OVERLAY_W / 2, 0,             0, self.screen_pan_x   ],
            [0,             OVERLAY_H / 2, 0, y_off               ],
            [0,             0,             1, -self.screen_distance],
            [0,             0,             0, 1                    ],
        ], dtype=np.float32)

        mvp = proj_mat @ view_mat @ rot_y @ rot_x @ overlay_model

        mgl_fbo.use()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._overlay_tex.use(location=2)
        self._overlay_prog['u_mvp'].write(mvp.T.astype('f4').tobytes())
        self._overlay_vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.disable(moderngl.BLEND)

    def _update_aim_poses(self, display_time):
        """Locate both controller aim spaces and cache their world-space 4×4 matrices."""
        for space, attr in [
            (self._aim_space_l, "_aim_mat_l"),
            (self._aim_space_r, "_aim_mat_r"),
        ]:
            if space is None:
                setattr(self, attr, None)
                continue
            try:
                loc = xr.locate_space(space, self._xr_space, display_time)
                if loc.location_flags & xr.SPACE_LOCATION_POSITION_VALID_BIT:
                    # Build a 4×4 model matrix from the aim pose (position + orientation)
                    R = _xr_quat_to_mat4(loc.pose.orientation)
                    R[:3, 3] = [loc.pose.position.x, loc.pose.position.y, loc.pose.position.z]
                    setattr(self, attr, R)
                else:
                    setattr(self, attr, None)
            except Exception:
                setattr(self, attr, None)

    def _laser_screen_hit_dist(self, ctrl_pos, fwd_w):
        """Return the distance along fwd_w where the aim ray hits the visible screen rect.

        Returns BEAM_MAX (5 m) if the ray misses the screen rectangle entirely or
        is parallel to it — so the laser only clips when it actually hits the screen.
        """
        BEAM_MAX = 30.0   # long enough to look infinite in any room-scale space
        # Compute height inline during resize (when screen_height is temporarily None)
        sh = self.screen_height
        if sh is None:
            fw, fh = self.frame_size
            sh = self.screen_width * (fh / fw if fw > 0 else 9.0 / 16.0)
        cp  = math.cos(self.screen_pitch); sp  = math.sin(self.screen_pitch)
        cy  = math.cos(self.screen_yaw);   sy_ = math.sin(self.screen_yaw)
        screen_n   = np.array([cp * sy_, -sp, cp * cy], dtype='f8')
        screen_pos = np.array([self.screen_pan_x, self.screen_pan_y, -self.screen_distance], dtype='f8')
        denom = float(np.dot(screen_n, fwd_w))
        if abs(denom) < 1e-6:
            return BEAM_MAX
        t = float(np.dot(screen_n, screen_pos - ctrl_pos)) / denom
        if t < 0.01:
            return BEAM_MAX   # screen is behind controller
        # Check hit is within the actual screen rectangle (not the infinite plane)
        hit   = ctrl_pos + fwd_w * t
        diff  = hit - screen_pos
        r_ax  = np.array([cy, 0.0, -sy_], dtype='f8')
        u_ax  = np.array([sp*sy_, cp, sp*cy], dtype='f8')
        loc_x = float(np.dot(diff, r_ax))
        loc_y = float(np.dot(diff, u_ax))
        if abs(loc_x) <= self.screen_width / 2.0 and abs(loc_y) <= sh / 2.0:
            return max(0.01, t - 0.005)   # hit within screen — stop 5 mm before
        return BEAM_MAX   # outside screen rectangle — let laser go to max

    def _render_lasers(self, mgl_fbo, view_mat, proj_mat):
        """Render a thin laser beam from each tracked controller aim pose."""
        BEAM_W   = 0.003   # half-width in metres
        DOT_SIZE = 0.025   # half-size of the controller origin dot

        for aim_mat, is_grab in [
            (self._aim_mat_l, self._grabbed),
            (self._aim_mat_r, self._resizing or (not self._grabbed)),
        ]:
            if aim_mat is None:
                continue

            fwd_w = -aim_mat[:3, 2].astype('f8')   # controller forward in world space

            # Controller origin dot
            scale_dot = np.diag([DOT_SIZE, DOT_SIZE, DOT_SIZE, 1.0]).astype('f4')
            dot_model = aim_mat @ scale_dot
            dot_mvp   = proj_mat @ view_mat @ dot_model
            color = (0.3, 0.7, 1.0, 1.0) if is_grab else (1.0, 1.0, 1.0, 0.9)
            mgl_fbo.use()
            self._border_prog['u_mvp'].write(dot_mvp.T.astype('f4').tobytes())
            self._border_prog['u_color'].value = color
            self._dot_vao.render(moderngl.TRIANGLE_STRIP)

            # Laser beam — stops at the screen surface, never passes through it.
            # Build transform from scratch:
            #   col0 = right × BEAM_W   (thin)
            #   col1 = fwd  × half_len  (long, along aim direction)
            #   col3 = ctrl_origin + fwd × half_len  (centre of beam quad)
            beam_len = self._laser_screen_hit_dist(aim_mat[:3, 3].astype('f8'), fwd_w)
            beam_mat = np.zeros((4, 4), dtype='f4')
            beam_mat[:3, 0] = aim_mat[:3, 0] * BEAM_W
            beam_mat[:3, 1] = fwd_w * (beam_len / 2.0)
            beam_mat[:3, 3] = aim_mat[:3, 3] + fwd_w * (beam_len / 2.0)
            beam_mat[3, 3]  = 1.0
            beam_mvp    = proj_mat @ view_mat @ beam_mat
            beam_color  = (*color[:3], 0.55)
            self._border_prog['u_mvp'].write(beam_mvp.T.astype('f4').tobytes())
            self._border_prog['u_color'].value = beam_color
            self._laser_vao.render(moderngl.TRIANGLE_STRIP)

    def _render_glow(self, mgl_fbo, view_mat, proj_mat):
        """Render a large Gaussian glow quad behind the screen (cinema ambient effect).

        The quad is GLOW_SCALE times the screen size, pushed 2 cm behind the screen
        surface, and composited with ADDITIVE blending so it brightens the dark VR
        background without changing the screen itself.
        """
        if self._glow_prog is None or self._glow_vao is None:
            return
        if self.screen_height is None:
            return

        GLOW_SCALE  = 5.0   # glow extends 5× beyond each screen edge (cinema wall fill)
        GLOW_OFFSET = 0.02  # push 2 cm behind the screen to avoid z-fighting

        cp  = math.cos(self.screen_pitch); sp  = math.sin(self.screen_pitch)
        cy  = math.cos(self.screen_yaw);   sy_ = math.sin(self.screen_yaw)
        # Screen normal — used to push the glow quad slightly behind the screen
        screen_n = np.array([cp * sy_, -sp, cp * cy], dtype=np.float32)

        rot_y = np.array([[ cy, 0, sy_, 0], [0, 1,  0, 0], [-sy_, 0, cy, 0], [0, 0, 0, 1]], dtype=np.float32)
        rot_x = np.array([[1, 0,  0, 0], [0, cp, -sp, 0], [0, sp, cp, 0], [0, 0, 0, 1]], dtype=np.float32)

        gx = self.screen_width  * GLOW_SCALE / 2.0
        gy = self.screen_height * GLOW_SCALE / 2.0
        # Centre of glow quad = screen centre pushed back along its normal
        gc = np.array([self.screen_pan_x, self.screen_pan_y, -self.screen_distance],
                      dtype=np.float32) - screen_n * GLOW_OFFSET
        glow_model = np.array([
            [gx, 0,  0, gc[0]],
            [0,  gy, 0, gc[1]],
            [0,  0,  1, gc[2]],
            [0,  0,  0, 1    ],
        ], dtype=np.float32)
        mvp = proj_mat @ view_mat @ rot_y @ rot_x @ glow_model

        mgl_fbo.use()
        self.ctx.enable(moderngl.BLEND)
        # Additive blend: glow adds light rather than replacing background colour
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        self._glow_prog['u_mvp'].write(mvp.T.astype('f4').tobytes())
        self._glow_vao.render(moderngl.TRIANGLE_STRIP)
        # Restore standard alpha blend for subsequent draws
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.disable(moderngl.BLEND)

    def _render_border(self, mgl_fbo, view_mat, proj_mat):
        """Render a thin solid-color quad slightly larger than the screen.

        Rendered BEFORE the screen so the screen covers the centre, leaving
        only a visible frame at the perimeter. Color is white normally and
        cyan when the user is grabbing (left grip) or resizing (right grip).
        """
        if self.screen_height is None:
            return
        alpha = self._border_alpha
        if alpha <= 0.0:
            return

        BORDER = 0.008   # metres of extra half-width on each side
        sx = self.screen_width  / 2.0 + BORDER
        sy = self.screen_height / 2.0 + BORDER

        cy  = math.cos(self.screen_yaw);   sy_ = math.sin(self.screen_yaw)
        cp  = math.cos(self.screen_pitch); sp  = math.sin(self.screen_pitch)
        rot_y = np.array([[ cy, 0, sy_, 0], [0, 1, 0, 0], [-sy_, 0, cy, 0], [0, 0, 0, 1]], dtype='f4')
        rot_x = np.array([[1, 0, 0, 0], [0, cp, -sp, 0], [0, sp, cp, 0], [0, 0, 0, 1]], dtype='f4')
        border_model = np.array([
            [sx, 0, 0, self.screen_pan_x],
            [0, sy, 0, self.screen_pan_y],
            [0, 0, 1, -self.screen_distance + 0.001],  # 1 mm closer → no z-fight
            [0, 0, 0, 1],
        ], dtype='f4')
        mvp = proj_mat @ view_mat @ rot_y @ rot_x @ border_model

        if self._grabbed:
            color = (0.3, 0.7, 1.0, alpha)   # cyan — moving
        else:
            color = (0.75, 0.75, 0.75, alpha * 0.9)  # light grey — idle

        mgl_fbo.use()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._border_prog['u_mvp'].write(mvp.T.astype('f4').tobytes())
        self._border_prog['u_color'].value = color
        self._border_vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.disable(moderngl.BLEND)

    def _render_eye(self, eye_index, mgl_fbo, view_mat, proj_mat):
        """Render one eye's parallax view into the swapchain FBO using world-space MVP.

        Left eye:  u_eye_offset = -ipd/2
        Right eye: u_eye_offset = +ipd/2
        """
        sc_w, sc_h = self._swapchain_sizes[eye_index]

        mgl_fbo.use()
        self.ctx.viewport = (0, 0, sc_w, sc_h)
        mgl_fbo.clear(0.0, 0.0, 0.0, 1.0)

        if not self._screen_visible:
            self.ctx.screen.use()
            return

        if self.color_tex is None or self.depth_tex is None:
            self.ctx.screen.use()
            return

        # 0. Cinema glow (large Gaussian behind everything, additive blend)
        if self._glow_visible:
            self._render_glow(mgl_fbo, view_mat, proj_mat)
            self.ctx.viewport = (0, 0, sc_w, sc_h)

        # 1. Border (behind the screen, slightly larger)
        self._render_border(mgl_fbo, view_mat, proj_mat)
        self.ctx.viewport = (0, 0, sc_w, sc_h)

        # 2. Main screen
        mgl_fbo.use()
        self.color_tex.use(location=0)
        self.depth_tex.use(location=1)

        model = self._build_model_mat4()
        mvp   = proj_mat @ view_mat @ model
        self.prog['u_mvp'].write(mvp.T.astype('f4').tobytes())

        eye_sign = -1.0 if eye_index == 0 else 1.0
        self.prog['u_eye_offset'].value     = eye_sign * self.ipd_uv / 2.0
        self.prog['u_depth_strength'].value = self.depth_strength * self.depth_ratio

        # Pass texture resolution so the disocclusion test uses correct pixel-space
        # gradients — matches viewer.py's effective behaviour with proper edge fill.
        if self._texture_size:
            self.prog['u_resolution'].value = (float(self._texture_size[0]),
                                               float(self._texture_size[1]))
        else:
            self.prog['u_resolution'].value = (0.0, 0.0)

        self.quad_vao.render(moderngl.TRIANGLE_STRIP)

        # 3. Laser pointers
        self.ctx.viewport = (0, 0, sc_w, sc_h)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._render_lasers(mgl_fbo, view_mat, proj_mat)
        self.ctx.disable(moderngl.BLEND)

        # 4. FPS overlay
        if self._fps_overlay_visible and self._overlay_tex is not None:
            self.ctx.viewport = (0, 0, sc_w, sc_h)
            self._render_fps_overlay(eye_index, mgl_fbo, view_mat, proj_mat)

        # 5. Virtual keyboard
        if self._keyboard_visible and self._keyboard_tex is not None:
            self.ctx.viewport = (0, 0, sc_w, sc_h)
            self._render_keyboard(mgl_fbo, view_mat, proj_mat)

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

    def _read_bool_action(self, action, hand_path_str="/user/hand/left"):
        """Return True if the boolean action is currently pressed on the given hand."""
        if action is None:
            return False
        try:
            path  = xr.string_to_path(self._xr_instance, hand_path_str)
            state = xr.get_action_state_boolean(
                self._xr_session,
                xr.ActionStateGetInfo(action=action, subaction_path=path),
            )
            return bool(state.is_active and state.current_state)
        except Exception:
            return False

    def _read_float_action(self, action, hand_path_str="/user/hand/left"):
        """Return the float value [0,1] of a trigger/squeeze action."""
        if action is None:
            return 0.0
        try:
            path  = xr.string_to_path(self._xr_instance, hand_path_str)
            state = xr.get_action_state_float(
                self._xr_session,
                xr.ActionStateGetInfo(action=action, subaction_path=path),
            )
            return float(state.current_state) if state.is_active else 0.0
        except Exception:
            return 0.0

    def _laser_screen_hit_uv(self, ctrl_pos, fwd_w):
        """Return (u, v, t) where the aim ray hits the screen surface, or None.

        u, v are in [0, 1] (u=0 left, v=0 bottom). t is the along-ray distance.
        Returns None if the ray misses the screen rect or is parallel to it.
        """
        sh = self.screen_height
        if sh is None:
            fw, fh = self.frame_size
            sh = self.screen_width * (fh / fw if fw > 0 else 9.0 / 16.0)
        cp  = math.cos(self.screen_pitch); sp  = math.sin(self.screen_pitch)
        cy  = math.cos(self.screen_yaw);   sy_ = math.sin(self.screen_yaw)
        screen_n   = np.array([cp * sy_, -sp, cp * cy], dtype='f8')
        screen_pos = np.array([self.screen_pan_x, self.screen_pan_y, -self.screen_distance], dtype='f8')
        denom = float(np.dot(screen_n, fwd_w))
        if abs(denom) < 1e-6:
            return None
        t = float(np.dot(screen_n, screen_pos - ctrl_pos)) / denom
        if t < 0.01:
            return None
        hit  = ctrl_pos + fwd_w * t
        diff = hit - screen_pos
        r_ax = np.array([cy, 0.0, -sy_],         dtype='f8')
        u_ax = np.array([sp * sy_, cp, sp * cy],  dtype='f8')
        loc_x = float(np.dot(diff, r_ax))
        loc_y = float(np.dot(diff, u_ax))
        if abs(loc_x) <= self.screen_width / 2.0 and abs(loc_y) <= sh / 2.0:
            u = 0.5 + loc_x / self.screen_width
            v = 0.5 + loc_y / sh
            return u, v, t
        return None

    def _keyboard_laser_hit(self, ctrl_pos, fwd_w):
        """Return (key_index, t) if the aim ray hits a key on the virtual keyboard, else (None, None)."""
        if not self._keyboard_keys:
            return None, None
        # Keyboard plane: normal = (0, 0, 1), position = (0, _keyboard_pan_y, -_keyboard_distance)
        kb_pos = np.array([0.0, self._keyboard_pan_y, -self._keyboard_distance], dtype='f8')
        kb_n   = np.array([0.0, 0.0, 1.0], dtype='f8')
        denom  = float(np.dot(kb_n, fwd_w))
        if abs(denom) < 1e-6:
            return None, None
        t = float(np.dot(kb_n, kb_pos - ctrl_pos)) / denom
        if t < 0.05:
            return None, None
        hit  = ctrl_pos + fwd_w * t
        # Local keyboard coords (keyboard is axis-aligned, centred at kb_pos)
        lx = float(hit[0]) - 0.0
        ly = float(hit[1]) - self._keyboard_pan_y
        for i, key in enumerate(self._keyboard_keys):
            x0, y0, x1, y1 = key.rect_local
            if x0 <= lx <= x1 and y0 <= ly <= y1:
                return i, t
        return None, None

    def _handle_cursor(self):
        """Move the Windows mouse cursor when a controller laser is pointing at the screen.

        The controller whose laser hits the screen controls the cursor.  When both hit,
        the one that was already active keeps priority to avoid jitter.
        """
        dw, dh = _get_desktop_size()
        hit_l = hit_r = None
        if self._aim_mat_l is not None:
            cp = self._aim_mat_l[:3, 3].astype('f8')
            fw = -self._aim_mat_l[:3, 2].astype('f8')
            hit_l = self._laser_screen_hit_uv(cp, fw)
        if self._aim_mat_r is not None:
            cp = self._aim_mat_r[:3, 3].astype('f8')
            fw = -self._aim_mat_r[:3, 2].astype('f8')
            hit_r = self._laser_screen_hit_uv(cp, fw)

        self._cursor_uv_l = (hit_l[0], hit_l[1]) if hit_l else None
        self._cursor_uv_r = (hit_r[0], hit_r[1]) if hit_r else None

        # Pick active controller — prefer current one if still on screen
        if hit_l and self._cursor_ctrl == 'left':
            u, v = hit_l[0], hit_l[1]
        elif hit_r and self._cursor_ctrl == 'right':
            u, v = hit_r[0], hit_r[1]
        elif hit_l:
            self._cursor_ctrl = 'left'
            u, v = hit_l[0], hit_l[1]
        elif hit_r:
            self._cursor_ctrl = 'right'
            u, v = hit_r[0], hit_r[1]
        else:
            self._cursor_ctrl = None
            return

        _set_cursor_pos(int(u * dw), int((1.0 - v) * dh))

    def _handle_triggers(self):
        """Map controller triggers to the Windows left mouse button.

        Trigger pulled past CLICK_THRESH → LEFTDOWN (held for as long as the trigger is held).
        Trigger released below CLICK_THRESH → LEFTUP.
        This makes single-clicks AND click-and-drag both work naturally — release the
        trigger to drop. Right-click is on the B button (see _poll_controller_input).
        If the controller's laser is over the virtual keyboard, the trigger drives the
        keyboard instead and the left button is released cleanly.
        """
        CLICK_THRESH = 0.7

        lt = self._read_float_action(self._act_left_trigger,  "/user/hand/left")
        rt = self._read_float_action(self._act_right_trigger, "/user/hand/right")

        left_on_kb  = self._kb_hover_l is not None
        right_on_kb = self._kb_hover_r is not None

        for trig_now, trig_prev_attr, on_kb, btn_down_attr in [
            (lt, '_left_trig_prev',  left_on_kb,  '_left_btn_down'),
            (rt, '_right_trig_prev', right_on_kb, '_right_btn_down'),
        ]:
            trig_prev = getattr(self, trig_prev_attr)
            btn_down  = getattr(self, btn_down_attr)

            if on_kb:
                # Laser on keyboard — release any held left button cleanly
                if btn_down:
                    _send_mouse_flags(_MOUSEEVENTF_LEFTUP)
                    setattr(self, btn_down_attr, False)
            else:
                if trig_now >= CLICK_THRESH and trig_prev < CLICK_THRESH and not btn_down:
                    # Rising edge — press the left mouse button (and hold while trigger held)
                    _send_mouse_flags(_MOUSEEVENTF_LEFTDOWN)
                    setattr(self, btn_down_attr, True)
                elif trig_now < CLICK_THRESH and trig_prev >= CLICK_THRESH and btn_down:
                    # Falling edge — release the left mouse button (completes click or drag)
                    _send_mouse_flags(_MOUSEEVENTF_LEFTUP)
                    setattr(self, btn_down_attr, False)

            setattr(self, trig_prev_attr, trig_now)

    def _handle_keyboard_input(self):
        """Send Windows keystrokes when a controller trigger fires on a keyboard key."""
        CLICK_THRESH = 0.7
        VK_SHIFT     = 0x10
        VK_CAPS      = 0x14

        lt = self._read_float_action(self._act_left_trigger,  "/user/hand/left")
        rt = self._read_float_action(self._act_right_trigger, "/user/hand/right")

        for trig_now, trig_prev_attr, hover_attr, aim_mat in [
            (lt, '_kb_trig_prev_l', '_kb_hover_l', self._aim_mat_l),
            (rt, '_kb_trig_prev_r', '_kb_hover_r', self._aim_mat_r),
        ]:
            trig_prev = getattr(self, trig_prev_attr)
            if aim_mat is not None:
                cp  = aim_mat[:3, 3].astype('f8')
                fw  = -aim_mat[:3, 2].astype('f8')
                idx, _ = self._keyboard_laser_hit(cp, fw)
            else:
                idx = None
            setattr(self, hover_attr, idx)

            # Rising edge on a key
            if trig_now >= CLICK_THRESH and trig_prev < CLICK_THRESH and idx is not None:
                key = self._keyboard_keys[idx]
                VK_CTRL = 0x11
                VK_ALT  = 0x12
                VK_WIN  = 0x5B
                if key.vk == VK_SHIFT:
                    self._shift_active = not self._shift_active
                elif key.vk == VK_CAPS:
                    self._caps_lock = not self._caps_lock
                elif key.vk == VK_CTRL:
                    self._ctrl_active = not self._ctrl_active
                elif key.vk == VK_ALT:
                    self._alt_active  = not self._alt_active
                elif key.vk == VK_WIN:
                    self._win_active  = not self._win_active
                else:
                    use_shift = self._shift_active ^ self._caps_lock
                    vk_to_use = key.shifted_vk if use_shift else key.vk
                    # Send the key with all currently-armed modifiers wrapping it
                    _send_key(vk_to_use,
                              shift=use_shift and vk_to_use == key.vk,
                              ctrl=self._ctrl_active,
                              alt=self._alt_active,
                              win=self._win_active)
                    # Auto-release one-shot modifiers after a non-modifier key
                    self._shift_active = False
                    self._ctrl_active  = False
                    self._alt_active   = False
                    self._win_active   = False

            setattr(self, trig_prev_attr, trig_now)

    def _poll_controller_input(self, dt):
        """Quest-like controls:
          Grip held       → grab screen: 1:1 controller delta tracking, perp to laser
          Right stick Y   → resize screen while grabbing
          Left stick Y    → push/pull screen distance while grabbing
          No grip         → right stick = yaw/pitch, left stick = pan
          A button        → left mouse click    B button → right mouse click
          Left trigger    → left click (on laser-screen hit)
          Right trigger   → hold = right button down; release = right button up
          Left stick click → toggle virtual keyboard
          Right stick click → toggle cinema glow
          Menu (left)     → toggle FPS overlay
        """
        if self._action_set is None:
            return

        def vec2(action, hand):
            try:
                path  = xr.string_to_path(self._xr_instance, hand)
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

        grip_l = self._read_bool_action(self._act_left_grip,  "/user/hand/left")
        grip_r = self._read_bool_action(self._act_right_grip, "/user/hand/right")

        prev_grabbed  = self._any_grip_last   # was ANY grip held last frame?
        any_grip      = grip_l or grip_r
        self._grabbed  = any_grip   # both grips show cyan — unified move mode
        self._resizing = False

        # Pick active controller: prefer left, fall back to right
        if grip_l and self._aim_mat_l is not None:
            aim_mat = self._aim_mat_l
        elif grip_r and self._aim_mat_r is not None:
            aim_mat = self._aim_mat_r
        elif grip_l and self._aim_mat_r is not None:
            aim_mat = self._aim_mat_r   # left gripped but no left aim yet
        else:
            aim_mat = None

        if any_grip and aim_mat is not None:
            ctrl_pos = aim_mat[:3, 3].astype('f8')
            fwd_w    = -aim_mat[:3, 2].astype('f8')   # controller forward in world space

            # Screen basis from current orientation
            cp = math.cos(self.screen_pitch); sp = math.sin(self.screen_pitch)
            cy = math.cos(self.screen_yaw);   sy_ = math.sin(self.screen_yaw)
            screen_n   = np.array([cp * sy_, -sp, cp * cy], dtype='f8')
            screen_r   = np.array([cy, 0.0, -sy_],          dtype='f8')
            screen_u   = np.array([sp * sy_, cp, sp * cy],  dtype='f8')
            screen_ctr = np.array([self.screen_pan_x, self.screen_pan_y,
                                   -self.screen_distance], dtype='f8')

            if not prev_grabbed:
                # Grab start: record grab_ray_dist and screen-local offsets only.
                # No world-space anchor stored — we recompute every frame from ctrl_pos+fwd_w*dist.
                denom = float(np.dot(screen_n, fwd_w))
                if abs(denom) > 1e-6:
                    t = float(np.dot(screen_n, screen_ctr - ctrl_pos)) / denom
                    if t > 0.05:
                        hit  = ctrl_pos + fwd_w * t
                        diff = hit - screen_ctr
                        self._grab_offset_x = float(np.dot(diff, screen_r))
                        self._grab_offset_y = float(np.dot(diff, screen_u))
                        self._grab_ray_dist = t
                    else:
                        self._grab_offset_x = 0.0
                        self._grab_offset_y = 0.0
                        self._grab_ray_dist = max(0.5, float(np.linalg.norm(screen_ctr - ctrl_pos)))
                else:
                    self._grab_offset_x = 0.0
                    self._grab_offset_y = 0.0
                    self._grab_ray_dist = max(0.5, float(np.linalg.norm(screen_ctr - ctrl_pos)))

            # Right thumbstick Y: resize — scale screen and offsets together
            if abs(ry) > DEAD:
                scale = 1.0 + ry * 1.5 * dt
                self._grab_offset_x *= scale
                self._grab_offset_y *= scale
                self.screen_width = max(0.3, self.screen_width * scale)
                self.screen_height = None

            # Left thumbstick Y: push/pull along aim ray
            if abs(ly) > DEAD:
                self._grab_ray_dist = max(0.05, self._grab_ray_dist + ly * self._dist_speed * dt)

            # Anchor = exact laser point at grab_ray_dist — zero drift, works for both
            # translation and rotation of the controller.
            anchor = ctrl_pos + fwd_w * self._grab_ray_dist

            # Rotate screen perpendicular to laser
            _fy = float(np.clip(fwd_w[1], -1.0, 1.0))
            self.screen_pitch = math.asin(_fy)
            self.screen_yaw   = math.atan2(-fwd_w[0], -fwd_w[2])
            _cp = math.cos(self.screen_pitch); _sp = math.sin(self.screen_pitch)
            _cy = math.cos(self.screen_yaw);   _sy = math.sin(self.screen_yaw)
            _sr = np.array([_cy, 0.0, -_sy],            dtype='f8')
            _su = np.array([_sp * _sy, _cp, _sp * _cy], dtype='f8')

            # Position screen so anchor lands at screen-local (grab_offset_x, grab_offset_y)
            _new_ctr = anchor - _sr * self._grab_offset_x - _su * self._grab_offset_y
            self.screen_pan_x    = float(_new_ctr[0])
            self.screen_pan_y    = float(_new_ctr[1])
            self.screen_distance = max(0.3, float(-_new_ctr[2]))

        elif not any_grip:
            # Free mode: right stick = yaw/pitch; left stick = pan
            if abs(rx) > DEAD: self.screen_yaw   += (-rx) * self._rot_speed * dt  # Inverted X
            if abs(ry) > DEAD: self.screen_pitch += ry * self._pitch_speed * dt
            if abs(lx) > DEAD: self.screen_pan_x += lx * self._dist_speed * 0.6 * dt
            if abs(ly) > DEAD: self.screen_pan_y += ly * self._dist_speed * 0.6 * dt

        # Menu (left) / Y (left) — same as A single-press for left-handed reach
        menu_now = self._read_bool_action(self._act_menu_btn, "/user/hand/left")
        if menu_now and not self._menu_pressed_last:
            self._fps_overlay_visible = not self._fps_overlay_visible
        self._menu_pressed_last = menu_now

        # A (right): left mouse click at current cursor position
        a_now = self._read_bool_action(self._act_a_btn, "/user/hand/right")
        if a_now and not self._a_last:
            _send_mouse_flags(_MOUSEEVENTF_LEFTDOWN)
            _send_mouse_flags(_MOUSEEVENTF_LEFTUP)
        self._a_last = a_now

        # B (right): right mouse click / context menu at current cursor position
        b_now = self._read_bool_action(self._act_b_btn, "/user/hand/right")
        if b_now and not self._b_last:
            _send_mouse_flags(_MOUSEEVENTF_RIGHTDOWN)
            _send_mouse_flags(_MOUSEEVENTF_RIGHTUP)
        self._b_last = b_now

        # Y (left): reset screen to default position / size / orientation
        y_now = self._read_bool_action(self._act_y_btn, "/user/hand/left")
        if y_now and not self._y_last:
            self.screen_distance = 2.0
            self.screen_width    = 2.0
            self.screen_height   = None       # rederived from frame aspect next render
            self.screen_pan_x    = 0.0
            self.screen_pan_y    = float(self._initial_head_y)  # eye height at session start
            self.screen_yaw      = 0.0
            self.screen_pitch    = 0.0
            self._border_alpha   = 1.0        # flash the border to confirm the reset
            self._border_idle_t  = time.perf_counter()
        self._y_last = y_now

        # Left thumbstick click: toggle virtual keyboard
        lsc_now = self._read_bool_action(self._act_left_stick_click, "/user/hand/left")
        if lsc_now and not self._left_stick_click_prev:
            self._keyboard_visible = not self._keyboard_visible
            if self._keyboard_visible and self._keyboard_tex is None:
                self._init_keyboard()
        self._left_stick_click_prev = lsc_now

        # Right thumbstick click: toggle cinema glow effect
        rsc_now = self._read_bool_action(self._act_right_stick_click, "/user/hand/right")
        if rsc_now and not self._right_stick_click_prev:
            self._glow_visible = not self._glow_visible
        self._right_stick_click_prev = rsc_now

        self._any_grip_last = any_grip  # persist for next frame's prev_grabbed check

        # Border fade: snap to 1 during grip; decay to 0 when idle
        interacting = grip_l or grip_r
        if interacting:
            self._border_alpha  = 1.0
            self._border_idle_t = time.perf_counter()
        else:
            idle = time.perf_counter() - self._border_idle_t
            FADE_DELAY = 1.5   # seconds before starting to fade
            FADE_DUR   = 0.8   # fade-out duration in seconds
            if idle > FADE_DELAY:
                self._border_alpha = max(0.0, 1.0 - (idle - FADE_DELAY) / FADE_DUR)

        # Cursor + trigger input (runs every frame regardless of grip state)
        self._handle_keyboard_input()   # updates _kb_hover_l/r, consumes keyboard triggers
        self._handle_cursor()           # moves Windows cursor when laser hits screen
        self._handle_triggers()         # fires mouse clicks (skips keys claimed by keyboard)

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

        last_input_t = time.perf_counter()

        while (
            not glfw.window_should_close(self.window)
            and not shutdown_event.is_set()
        ):
            now = time.perf_counter()
            dt = now - last_input_t
            last_input_t = now

            glfw.poll_events()
            self._poll_xr_events()

            if not self._session_running:
                time.sleep(0.01)
                continue

            # — Wait for the runtime to signal frame timing —
            frame_state = xr.wait_frame(self._xr_session, xr.FrameWaitInfo())
            xr.begin_frame(self._xr_session, xr.FrameBeginInfo())

            # sync_actions must happen before xr.locate_space for action spaces.
            # Do it here so _update_aim_poses gets fresh locations this frame.
            if self._action_set is not None:
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
                    pass

            # Locate controller spaces (now valid after sync_actions)
            self._update_aim_poses(frame_state.predicted_display_time)
            # Poll button/stick states (sync already done above)
            self._poll_controller_input(dt)

            composition_layers = []

            if frame_state.should_render:
                # Drain depth_q non-blocking — keep only the newest frame
                latest = None
                try:
                    while True:
                        latest = self.depth_q.get_nowait()
                except _queue.Empty:
                    pass

                if latest is not None:
                    rgb, depth, frame_ts = latest
                    self._update_frame(rgb, depth)
                    if frame_ts is not None:
                        self.total_latency = (time.perf_counter() - frame_ts) * 1000.0
                    # SBS source rate: time of arrival of each unique frame from depth_q
                    sbs_now = time.perf_counter()
                    self._sbs_ts_ring.append(sbs_now)
                    m = len(self._sbs_ts_ring)
                    if m >= 2:
                        sbs_span = sbs_now - self._sbs_ts_ring[0]
                        if sbs_span > 0:
                            self.sbs_fps = (m - 1) / sbs_span

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

                # Align screen_pan_y to headset eye height on the first valid frame.
                if not self._screen_eye_init and views and views[0] is not None:
                    try:
                        ey = (views[0].pose.position.y + views[1].pose.position.y) / 2.0
                        self.screen_pan_y    = float(ey)
                        self._initial_head_y = float(ey)
                    except Exception:
                        pass
                    self._screen_eye_init = True

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

            # Timestamp-ring FPS: (N-1) frames / (last_ts - first_ts) — exact, O(1)
            t_now = time.perf_counter()
            self._frame_ts_ring.append(t_now)
            n = len(self._frame_ts_ring)
            if n >= 2:
                span = t_now - self._frame_ts_ring[0]
                if span > 0:
                    self.actual_fps = (n - 1) / span

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

        # Release GPU interop PBOs
        if self._pbo_color is not None and self._cuda_gl:
            try:
                self._cuda_gl.unregister_resource(self._cuda_res_color)
                self._cuda_gl.unregister_resource(self._cuda_res_depth)
                glDeleteBuffers(2, [self._pbo_color, self._pbo_depth])
            except Exception:
                pass
        self._pbo_color = self._pbo_depth = None

        for tex in (self._overlay_tex, self.color_tex, self.depth_tex):
            if tex:
                try:
                    tex.release()
                except Exception:
                    pass
        self._overlay_tex = self.color_tex = self.depth_tex = None

        for swapchain in self._xr_swapchains.values():
            try:
                xr.destroy_swapchain(swapchain)
            except Exception:
                pass
        self._xr_swapchains.clear()

        for space_attr in ("_aim_space_l", "_aim_space_r", "_xr_space"):
            sp = getattr(self, space_attr, None)
            if sp:
                try:
                    xr.destroy_space(sp)
                except Exception:
                    pass
                setattr(self, space_attr, None)

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

if __name__ == "__main__":
    # Add this test function at the end of openxr_viewer.py

    def test_openxr_basic():
        """Minimal test to verify OpenXR initializes without errors."""
        if not OPENXR_AVAILABLE:
            print("[TEST] OpenXR not available - skipping test")
            return False
        
        print("[TEST] Starting basic OpenXR test...")
        
        try:
            # Create viewer with minimal config
            viewer = OpenXRViewer(
                frame_size=(640, 480),
                fps=30,
                show_fps=False
            )
            
            # Test GLFW/ModernGL init
            print("[TEST] Initializing GLFW/ModernGL...")
            viewer._init_glfw()
            viewer._init_moderngl()
            print("[TEST] ✓ GLFW/ModernGL OK")
            
            # Test OpenXR init (without starting session)
            print("[TEST] Initializing OpenXR...")
            viewer._init_openxr()
            print("[TEST] ✓ OpenXR initialized")
            
            # Test texture creation
            print("[TEST] Creating test textures...")
            viewer._init_textures(640, 480)
            print("[TEST] ✓ Textures created")
            
            # Cleanup
            print("[TEST] Cleaning up...")
            if viewer.window:
                glfw.destroy_window(viewer.window)
                glfw.terminate()
            
            print("[TEST] ✅ All basic tests passed!")
            return True
            
        except Exception as e:
            print(f"[TEST] ❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    # Run test when script is executed directly
    success = test_openxr_basic()
    sys.exit(0 if success else 1)