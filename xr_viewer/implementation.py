# openxr_viewer.py
# Renders Desktop2Stereo's depth-parallax left/right eye views into a VR headset
# via the pyopenxr binding (pip install pyopenxr).
# Uses a world-space virtual screen quad with proper per-eye view/projection matrices
# derived from xr.locate_views() for full 6DoF/3DoF head tracking.
# The depth-parallax FRAGMENT_SHADER from viewer.py is reused unchanged.

import sys
import os
import math
import time
import ctypes
import json
import queue as _queue
import collections

import glfw
import moderngl
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from utils import get_font_type, ROWS
from OpenGL.GL import (
    glGenFramebuffers, glBindFramebuffer, glFramebufferTexture2D,
    glDeleteFramebuffers, glCheckFramebufferStatus,
    GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
    GL_FRAMEBUFFER_COMPLETE, GL_RGBA8,
    glGenBuffers, glDeleteBuffers, glBindBuffer, glBufferData, glBufferSubData,
    glBindTexture, glTexSubImage2D, glGenerateMipmap,
    GL_PIXEL_UNPACK_BUFFER, GL_PIXEL_PACK_BUFFER, GL_DYNAMIC_DRAW, GL_STREAM_DRAW, GL_STREAM_READ,
    GL_RGB, GL_RED, GL_RGBA, GL_BGRA, GL_UNSIGNED_BYTE, GL_FLOAT,
    glDisable, glEnable, GL_FRAMEBUFFER_SRGB, GL_CULL_FACE,
    glFrontFace, GL_CW, GL_CCW,
    glReadBuffer, glBlitFramebuffer, glGenRenderbuffers, glBindRenderbuffer,
    glRenderbufferStorage, glFramebufferRenderbuffer, glDeleteRenderbuffers,
    GL_READ_FRAMEBUFFER, GL_DRAW_FRAMEBUFFER, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_LINEAR,
    GL_RENDERBUFFER, GL_DEPTH_ATTACHMENT, GL_DEPTH_COMPONENT24,
    glTexParameterf, GL_TEXTURE_LOD_BIAS,
    glMapBuffer, glUnmapBuffer, GL_READ_ONLY, GL_MAP_UNSYNCHRONIZED_BIT,
    glClear, glReadPixels, glFlush, glGenTextures, glDeleteTextures,
    glFenceSync, glClientWaitSync, glDeleteSync,
    GL_SYNC_GPU_COMMANDS_COMPLETE, GL_SYNC_FLUSH_COMMANDS_BIT,
)

try:
    import xr
    OPENXR_AVAILABLE = True
except ImportError:
    OPENXR_AVAILABLE = False
    print("[OpenXRViewer] pyopenxr not installed. Run: pip install pyopenxr")

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_MODULE_DIR)


from . import render as _render
from .constants import (
    EDGE_STRENGTH, KB_CURSOR_PRIORITY_BIAS, KB_CURSOR_RELEASE_GRACE,
    _DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, _DXGI_FORMAT_R8G8B8A8_UNORM,
    _DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, _DXGI_FORMAT_B8G8R8A8_UNORM,
    _D3D11_PREFERRED_FORMATS, _GL_SRGB8_ALPHA8,
    _BG_COLORS, _DEFAULT_BC, _DEFAULT_EF, _DEFAULT_TO, _DEFAULT_TS,
    _SCREEN_ENV_DEPTH_BIAS_M, _CURVED_CURVATURE_SCALE, _CURVED_HALF_ANGLE_RAD,
    DEAD, _VIVE_TB_Y, _KB_UNITS_WIDE, _KB_ROWS, _KeyEntry, _KB_TEX_W, _KB_TEX_H,
)
from .glsl import (
    _WORLD_VERT, _OVERLAY_FRAG, _BLIT_FRAG, _SOLID_VERT, _SOLID_FRAG,
    _BEAM_VERT, _BEAM_FRAG, _CURVED_VERT, _CTRL_VERT, _CTRL_FRAG,
    _ENV_VERT, _ENV_FRAG, _GLOW_FRAG, _FROST_GLOW_VERT,
    _FROST_CURVED_VERT, _FROST_GLOW_FRAG, _FROST_VEIL_FRAG,
    _PANORAMA_VERT, _PANORAMA_FRAG,
)
from .input import (
    _MOUSEEVENTF_LEFTDOWN, _MOUSEEVENTF_LEFTUP,
    _MOUSEEVENTF_RIGHTDOWN, _MOUSEEVENTF_RIGHTUP,
    _TOUCH_AVAILABLE, _TOUCH_CONTACT_ID_LEFT, _TOUCH_CONTACT_ID_RIGHT,
    _TOUCH_PINCH_SPREAD_GAIN, _get_desktop_size, _send_hscroll, _send_key,
    _send_mouse_flags, _send_vscroll, _set_cursor_pos, _touch_injector,
    _U32, _KEYEVENTF_KEYUP,
    EMAPositionFilter, OneEuroFilter, OneEuroFilter3D,
)
from .render import (
    _apply_transform, _build_node_matrices, _create_d3d11_device,
    _create_d3d11_shared_texture, _d3d11_update_subresource, _euler_to_mat4,
    _fov_to_proj_mat4, _fov_to_proj_mat4_cached, _get_accessor, _mat3_to_quat_xyzw,
    _mat4_to_xr_posef, _pose_to_view_mat4, _quat_to_mat4, _read_glb_chunks,
    _view_mat_inv, _xr_pose_to_model_mat4, _xr_quat_to_mat4, load_glb_model,
)





from viewer import FRAGMENT_SHADER as _BASE_FRAGMENT_SHADER, BACKEND
from .d3d11_backend import D3D11BackendMixin
from .environment import EnvironmentMixin
from .overlay import OverlayMixin

try:
    from viewer import CUDART_GL
except ImportError:
    CUDART_GL = None


def _make_xr_fragment_shader(src):
    """Add XR-only source cropping without changing the desktop viewer shader."""
    if "uniform vec4 u_source_crop;" not in src:
        src = src.replace(
            "uniform float u_roll;          // screen roll (radians), rotates parallax direction",
            "uniform float u_roll;          // screen roll (radians), rotates parallax direction\n"
            "    uniform vec4 u_source_crop;    // xy = source top-left, zw = source size",
            1,
        )
    src = src.replace(
        "vec2 flipped_uv = vec2(uv.x, 1.0 - uv.y);",
        "vec2 screen_flipped_uv = vec2(uv.x, 1.0 - uv.y);\n"
        "        vec2 flipped_uv = u_source_crop.xy + screen_flipped_uv * u_source_crop.zw;",
        1,
    )
    return src


FRAGMENT_SHADER = _make_xr_fragment_shader(_BASE_FRAGMENT_SHADER)


class OpenXRViewer(D3D11BackendMixin, EnvironmentMixin, OverlayMixin):
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
        controller_model='pico',
        capture_mode='Monitor',
        monitor_index=1,
        environment_name=None,
        glow_mode=None,
        screen_curve=None,
        **kwargs,
    ):
        self._controller_model = controller_model
        self._capture_mode = capture_mode
        self._input_monitor_index = monitor_index
        self.ipd_uv = ipd
        self.depth_strength = 0.1 # multiplied by depth_ratio; effective = depth_strength * depth_ratio
        self.depth_ratio = depth_ratio
        self.convergence = convergence
        self.frame_size = frame_size
        self.fps = fps
        self.depth_q = depth_q
        self.show_fps = show_fps

        self._screen_grab_local_l = None
        self._screen_grab_local_r = None
        # Keyboard grip-to-move anchor local (lx, ly) on keyboard plane
        # at the moment the grip was pressed.  Mirrors _screen_grab_local_*.
        self._kb_grab_local_l     = None
        self._kb_grab_local_r     = None
        # Standalone offsets for keyboard orientation, applied on top of the
        # auto-orientation that aims the panel at the head.  Defaults are 0
        # because the auto-orientation already produces the natural tilt that
        # makes the keyboard face the head (e.g. a keyboard below the head
        # gets a negative base_pitch that tilts the surface up toward the user).
        self._kb_yaw_offset       = 0.0
        self._kb_pitch_offset     = 0.0
        # Per-hand grip lock once a grip press latches onto 'screen' or
        # 'keyboard', it stays there until release.  Enforces "only grip one
        # item at a time" so a single press cannot jump between objects.
        self._grip_target_l       = None   # None | 'screen' | 'keyboard'
        self._grip_target_r       = None
        # Suppress keyboard typing while any grip is held (avoids accidental
        # key presses while repositioning the keyboard).
        self._grip_l_now          = False
        self._grip_r_now          = False

        self._x_press_t = 0.0          # timestamp when X was pressed
        self._x_long_fired = False     # whether long-ress action already fired
        self._x_glow_fired = False     # glow action fired during this hold (blocks light on release)
        self._prev_bg_color_idx = None # stores bg color idx before switching to green
        self._prev_active_env = None   # stores Environment Model before switching to green

        # FPS display timestamp ring: (len-1)/(last-first) is exact over the window
        self.actual_fps      = 0.0   # XR composition rate (this loop)
        self.sbs_fps         = 0.0   # SBS source rate (depth_q producer in main.py)
        self.total_latency   = 0.0
        self._frame_ts_ring  = collections.deque(maxlen=60)  # ~1 s at 60 Hz
        self._sbs_ts_ring    = collections.deque(maxlen=60)  # SBS frame arrivals

        # Overlay redraw throttle texture is rebuilt at most once per second
        self._last_overlay_update   = 0.0
        self._cached_actual_fps     = 0.0
        self._cached_sbs_fps        = 0.0
        self._cached_latency        = 0.0
        self._cached_screen_width   = 0.0
        self._cached_screen_height  = 0.0
        self._cached_screen_dist    = 0.0
        self._cached_screen_curved  = False
        self._cached_depth_ratio    = 1.0
        self._cached_vr_res         = (0, 0)
        self._cached_sbs_res        = (0, 0)
        self._fps_display_ema       = 0.0

        # Virtual screen transform (world space, metres / radians)
        self.screen_distance = 2.0
        self.screen_width    = 2.4
        self._screen_ref_size = 2.4   # long-dimension reference for resize
        self.screen_height   = None   # derived from frame aspect ratio on first frame

        # Auto letterbox crop for wide movies playing on a normal desktop
        # monitor.  Detection reads only the captured desktop frame and runs at
        # a low cadence; rendering reuses the last accepted crop.
        self._auto_movie_crop = bool(kwargs.get('auto_movie_crop', True))
        self._movie_crop_detect_interval = float(kwargs.get('movie_crop_detect_interval', 1.0))
        self._movie_crop_next_detect_t = 0.0
        self._movie_crop_target_uv = (0.0, 0.0, 1.0, 1.0)
        self._movie_crop_render_uv = (0.0, 0.0, 1.0, 1.0)
        self._movie_crop_full_hits = 0
        self._movie_crop_reveal_until = 0.0
        self._movie_crop_reveal_margin = 0.10
        self._movie_crop_pending_gpu = None
        self._movie_crop_sample_cache = None
        self._movie_crop_target_active = False
        self._movie_crop_render_active = False
        self._movie_crop_profile_cache_key = None
        self._movie_crop_profile_cache_value = None
        self._movie_crop_frame_uv = (0.0, 0.0, 1.0, 1.0)
        self._source_crop_uniform_cache = {}
        self._frost_uniform_cache = {}
        self._frost_layout_cache_key = None
        self._frost_layout_cache_val = None
        self._frost_model_cache_key = None
        self._frost_model_cache_bytes = None
        self._frost_model_uniform_cache = {}
        self._color_tex_mipmap_filter_active = None
        self._rgb_hwc_upload_tensor = None
        self._rgb_hwc_upload_key = None
        self._torch_mod = None

        # screen presets: (name, width_m, distance_m) height is derived from width and frame aspect ratio
        self._screen_presets = [
            ("10\" Tablet",        0.30, 0.4),
            ("27\" Monitor",       0.60, 0.6),
            ("65\" TV",            1.44, 2.0),
            ("100\" Projector 1",  2.40, 2.0),
            ("100\" Projector 2",  2.21, 2.5),
            ("1000\" IMAX",       22.0,  20),
        ]
        self._preset_index = 3  # default 4: 100-inch projector with 2.0 m distance
        self._preset_name_overlay = None   # preset name overlay (permanently visible)
        self.screen_pan_x    = 0.0
        self.screen_pan_y    = 0.0
        self.screen_yaw      = 0.0    # rotation around Y axis
        self.screen_pitch    = 0.0    # rotation around X axis
        self.screen_roll     = 0.0    # rotation around Z axis (screen normal)
        self._corner_radius = 0.03    # normalized [0, 0.5]

        # Environment model (glTF 3D scene) loaded on demand from
        # environment/<Name>/environment.glb.  One environment at a time;
        # old resources are released before loading a new one.
        self._env_model_prims = []        # active list of {'vao', 'vbo', 'ibo', 'tex_key', 'tri_count'}
        self._scene_lights = []           # KHR_lights_punctual lights from active env
        self._env_model_tex_cache = {}    # cache_key -> moderngl Texture (active env)
        self._env_model_visible = False   # hidden by default; left-stick short press cycles environments
        self._env_model_pos = [0.0, 0.0, 0.0]     # world-space position (x, y, z)
        self._env_model_rot = [0.0, 0.0, 0.0]     # Euler rotation (yaw, pitch, roll) in radians
        self._env_model_scale = [1.0, 1.0, 1.0]   # scale factor (x, y, z)
        self._env_model_init_done = False  # lazy init: scanned environment/ once
        self._env_model_path = None        # resolved GLB path for active environment

        # Environment profile (read-only at runtime from profile.json)
        self._env_profile = {}             # full loaded profile dict
        self._screen_profile = {}          # cached "screen" section from profile
        self._view_pose_profile = {}       # cached "view_pose" section from profile
        self._view_pose_profiles = []      # optional multi-seat view_poses list
        self._view_pose_index = 0           # active entry in view_poses
        self._environment_enabled = True
        self._env_allow_curve = False       # profile override: allow curved screen in locked env
        self._lighting_presets = []         # list of dicts from profile lighting_presets
        self._lighting_preset_index = 0    # current active preset

        # Environment registry
        self._environment_root = os.path.join(_MODULE_DIR, 'environments')
        self._environment_model = environment_name  # selected env folder name
        self._available_environment_models = []     # discovered room folders
        self._active_environment  = None    # current env folder name, or None when colour-only
        self._env_switch_osd_t    = 0.0     # perf_counter stamp for env-switch OSD (future use)

        # Built-in procedural "Dark Room" always available, never in the
        # cycle.  Auto-rendered behind the screen when no .glb env is loaded
        # AND the backdrop is "Black" (idx 0).
        self._dark_room_prims = []         # populated by _init_dark_room() at GL init time

        # Screen glow effect (cinema light)
        self._glow_color = (0.3, 0.6, 1.0)        # light blue, dynamically updated
        self._glow_width_m = 0.50                  # glow decay distance (larger volume)
        self._glow_intensity = 0.175               # softer glow (0.5x)
        self._glow_intensity_multiplier = 0.0     # 0 = no glow (Default), 1.5 = "Default with Glow"
        self._glow_ref_screen = 2.4               # reference screen long edge (meters)
        self._glow_target_color = (0.3, 0.6, 1.0)  # latest sampled frame average
        self._glow_color_counter = 0              # reserved for future tuning
        self._glow_mode = 'off'                   # glow | veil | frosted | off
        requested_glow_mode = glow_mode if glow_mode is not None else kwargs.get('glow_mode', None)
        if requested_glow_mode is not None:
            mode = str(requested_glow_mode or 'off').strip().lower()
            self._glow_mode = {
                'none': 'off',
                'false': 'off',
                '0': 'off',
                'frost': 'frosted',
                'frost_glow': 'frosted',
                'frosted_glow': 'frosted',
            }.get(mode, mode)
        self._active_glow_mode_cached = self._glow_mode
        self._frost_glow_intensity = 1.0
        self._frost_glow_alpha = 0.42
        self._frost_glow_threshold = 0.46
        self._frost_glow_lod = 5.4
        self._frost_glow_blend = 1.35
        self._frost_glow_thickness = 1.6
        self._frost_glow_diffuse = 0.85
        self._frost_glow_margin_m = 3.6
        self._frost_glow_inset = 0.045
        self._frost_veil_intensity = 1.5
        self._frost_veil_alpha = 1.00
        self._frost_veil_threshold = 0.0
        self._frost_veil_lod = 0.0
        self._screen_light_intensity = 3.5

        # Environment lighting config (overridable via profile.json)
        self._env_head_light_color = (0.45, 0.45, 0.48)
        self._env_ambient_color = (0.08, 0.08, 0.09)
        self._env_fill_lights = []           # [{position, color, range}, ...]
        self._env_exposure = 1.0
        self._env_gamma = 2.2
        self._env_emissive_strength = 1.0
        self._env_khr_light_scale = 1.0

        # Snapshot factory defaults so _reset_environment_profile_defaults
        # can restore them before loading each new profile.
        self._env_base_settings = {
            'model_pos': list(self._env_model_pos),
            'model_rot': list(self._env_model_rot),
            'model_scale': list(self._env_model_scale),
            'head_light_color': list(self._env_head_light_color),
            'ambient_color': list(self._env_ambient_color),
            'fill_lights': list(self._env_fill_lights),
            'exposure': self._env_exposure,
            'gamma': self._env_gamma,
            'emissive_strength': self._env_emissive_strength,
            'khr_light_scale': self._env_khr_light_scale,
            'screen_light_intensity': self._screen_light_intensity,
            'glow_intensity': self._glow_intensity,
            'glow_width': self._glow_width_m,
            'glow_intensity_multiplier': self._glow_intensity_multiplier,
            'glow_mode': self._glow_mode,
        }

        self._yaw_offset     = 0.0    # manual yaw offset (relative to face-head baseline)
        self._pitch_offset   = 0.0    # manual pitch offset

        # Interaction speeds (per second at full stick deflection)
        self._pan_speed   = 0.5    # screen translate X/Y (m/s)
        self._rot_speed   = 0.35   # screen yaw rotation (rad/s)
        self._size_speed  = 0.5    # screen width resize (m/s)
        # Distance: acceleration curve params (speed = base + (max-base) * t^exp)
        self._dist_speed_base = 0.5   # min speed at deadzone edge (m/s)
        self._dist_speed_max  = 3.0   # max speed at full deflection (m/s)
        self._dist_speed_exp  = 2.5   # exponent (>1: fine near centre, fast at edge)

        # OpenXR handles
        self._xr_instance = None
        self._xr_system_id = None
        self._xr_session = None
        self._xr_space = None
        self._xr_ref_space_type = None
        self._xr_space_pose_in_ref = np.eye(4, dtype=np.float32)
        self._xr_profile_space_applied = False
        self._last_located_views = None
        self._pending_recenter = False
        self._xr_swapchains = {}        # {eye_index: xr.Swapchain}
        self._swapchain_images = {}     # {eye_index: [XrSwapchainImageOpenGLKHR, ...]}
        self._swapchain_sizes = {}      # {eye_index: (w, h)}
        self._fbo_cache = {}            # {(eye_index, image_index): (raw_id, mgl_fbo)}
        self._depth_rb_cache = {}       # {(eye_index, image_index): depth_rb}
        self._show_preview_window = bool(kwargs.get('show_preview_window', True))
        self._preview_active = False
        self._session_running = False

        # Pre-built XR call argument structs stateless inputs reused every
        # frame to avoid 6+ ctypes allocations per frame (per eye for swapchain
        # acquire/wait/release). pyopenxr structs are read-only inputs to the
        # runtime so reusing them across calls is safe.
        self._xr_frame_wait_info     = xr.FrameWaitInfo()
        self._xr_frame_begin_info    = xr.FrameBeginInfo()
        self._xr_sc_acquire_info     = xr.SwapchainImageAcquireInfo()
        self._xr_sc_wait_info        = xr.SwapchainImageWaitInfo(timeout=xr.INFINITE_DURATION)
        self._xr_sc_release_info     = xr.SwapchainImageReleaseInfo()
        self._xr_actions_sync_info   = None   # built once _action_set is created

        # Controller action handles (set by _init_controller_actions)
        self._action_set          = None
        self._act_left_stick      = None
        self._act_right_stick     = None
        self._act_menu_btn        = None
        self._act_left_grip       = None   # grab/move mode
        self._act_right_grip      = None   # resize mode
        self._act_a_btn           = None   # right A double press = hide all
        self._act_b_btn           = None   # right B right mouse click
        self._act_x_btn           = None   # left  X toggle virtual keyboard
        self._act_y_btn           = None   # left  Y reset screen position/size/rotation
        self._act_left_trigger       = None   # left trigger (float) left mouse click
        self._act_right_trigger      = None   # right trigger (float) right mouse click / hold
        self._act_left_stick_click   = None   # left thumbstick click cycle environment
        self._act_right_stick_click  = None   # right thumbstick click cycles screen curve mode

        # Vive trackpad button emulation (computed per-frame, OR'd into reads)
        self._emu_a   = False   # right hand bottom click A
        self._emu_b   = False   # right hand top click    B
        self._emu_x   = False   # left hand bottom click  X
        self._emu_y   = False   # left hand top click     Y
        self._emu_lsc = False   # left hand center click  left stick click
        self._emu_rsc = False   # right hand center click right stick click

        # Menu button debounce + FPS overlay toggle + long-press reset
        self._menu_pressed_last   = False
        self._fps_overlay_visible = show_fps
        self._help_panel_visible  = False   # shortcuts panel, toggled independently
        self._menu_press_t        = 0.0    # perf_counter when menu was pressed
        self._menu_long_fired     = False  # True once long-press action has fired

        # Quest-like window state
        self._screen_visible = True   # hide-all toggle (A double-press)
        self._grabbed        = False  # left grip held  move mode
        self._screen_grab_grip_l = None  # grab offset (screen_centre - grip_pos) for left
        self._screen_grab_grip_r = None  # grab offset (screen_centre - grip_pos) for right
        self._resizing       = False  # right grip held resize mode
        self._seat_adjust_active = False
        self._seat_adjust_t     = 0.0
        self._seat_adjust_osd_tex      = None
        self._seat_adjust_osd_vao      = None
        self._seat_adjust_osd_tex_size = (768, 78)
        self._seat_adjust_osd_show_t   = 0.0
        self._seat_adjust_osd_alpha    = 0.0
        self._seat_adjust_osd_dirty    = True
        self._seat_adjust_grip_move    = False
        self._both_grips_last = False
        self._both_grips_hold_t    = 0.0
        self._both_grips_long_fired = False
        self._screen_state_dirty   = False
        self._screen_state_save_t  = 0.0
        self._settings_sync_dirty  = False
        self._settings_sync_save_t = 0.0
        self._last_persisted_depth_ratio = float(self.depth_ratio)
        self._a_last         = False  # A-button previous frame state
        self._a_last_t       = 0.0   # timestamp of last A press (double-press detection)
        self._b_last              = False  # B-button previous frame state
        self._ltrig_state   = 'idle'  # 'idle' | 'pressed' | 'dragging'
        self._rtrig_state   = 'idle'
        self._ltrig_press_t = 0.0    # perf_counter of last rising edge left
        self._rtrig_press_t = 0.0    # perf_counter of last rising edge right
        self._ov_ltrig_held = False  # overlay panel trigger state (separate from normal)
        self._ov_rtrig_held = False
        self._status_panel_alpha = 1.0  # animated alpha: fades when trigger held on panel
        self._y_last         = False  # Y-button previous frame state (reset screen)
        self._y_press_t      = 0.0   # perf_counter when Y was pressed
        self._y_long_fired   = False # True once long-press action fired this hold
        self._x_last         = False  # X-button previous frame state (toggle keyboard)
        # Head pose (world) cached each frame from xr.locate_views used as the orbit
        # pivot for the left thumbstick and as the anchor when the keyboard is summoned.
        self._head_pos_w      = None   # (x, y, z) head/eye centre in world space, or None
        self._head_fwd_w      = None   # (fx, fy, fz) head forward unit vector, or None
        self._screen_eye_init = False  # screen_pan_y aligned to headset height on first frame
        self._initial_head_y  = 0.0   # headset eye height at session start, used for Y-reset
        # True once a profile screen layout has been applied for the active
        # environment (set during _init_env_model on startup).
        # Used by the first-frame eye-init block to skip the "snap to default
        # in-front-of-gaze" reset so the profile layout is preserved.
        self._profile_loaded  = False
        # Border fade: shown during interaction, fades out when idle
        self._border_alpha   = 0.0    # 0.0 = invisible, 1.0 = fully opaque
        self._border_idle_t  = 0.0    # wall time when interaction last ended
        # Keyboard border fade
        self._kb_border_alpha  = 0.0
        self._kb_border_idle_t = 0.0
        self._saved_dclick_time = None  # system double-click time saved before session

        # Mouse cursor control
        self._cursor_uv_l         = None  # (u,v,t) where left laser hits screen, or None
        self._cursor_uv_r         = None  # (u,v,t) where right laser hits screen, or None
        self._overlay_hit_l       = False # left laser hits FPS overlay panel
        self._overlay_hit_r       = False # right laser hits FPS overlay panel
        self._cursor_ctrl         = None  # 'left' | 'right' | None active cursor controller
        # Smoothed UV exponential moving average tames hand tremor so the cursor
        # doesn't jitter or skip pixels at long laser distances. Reset when the active
        # controller changes so we don't drag the cursor across the screen on swap.
        self._cursor_smooth_uv    = None
        # Post-release grace: after the keyboard stops owning the cursor (typing
        # ends / ray leaves the keys), briefly keep the screen cursor suppressed so
        # ownership doesn't snap to the screen while the user lifts off toward it.
        self._KB_RELEASE_GRACE    = KB_CURSOR_RELEASE_GRACE  # seconds of keyboard-ownership hold after release
        self._kb_cursor_owned_t_l = 0.0    # perf_counter stamp: last frame KB owned cursor (left)
        self._kb_cursor_owned_t_r = 0.0    # perf_counter stamp: last frame KB owned cursor (right)
        self._left_btn_down       = False # left mouse button held via left trigger

        # Per-hand desktop pixel positions for the multi-touch injector refreshed
        # each frame by _handle_cursor and consumed by _handle_triggers. ``valid``
        # is True only when the hand currently owns a usable screen target (laser
        # on screen, not on keyboard, not on overlay).
        self._touch_px_l       = (0, 0)
        self._touch_px_r       = (0, 0)
        self._touch_valid_l    = False
        self._touch_valid_r    = False
        # EMA-smoothed positions used while a touch contact is held smooths out
        # controller jitter without adding the latency of full position smoothing.
        self._touch_smooth_l   = None
        self._touch_smooth_r   = None
        # Per-hand contact state ('idle' | 'down') for edge detection. The lower
        # level _TouchInjector tracks DOWN/UPDATE/UP transitions; we only
        # need to know when to flip the request.
        self._touch_state_l    = 'idle'
        self._touch_state_r    = 'idle'
        # Prior-frame trigger values per hand used to enforce a *true* rising
        # edge (release-then-press) for the touch DOWN transition. Without this,
        # holding the trigger while the laser slides off the virtual keyboard
        # onto the screen or while the keyboard is toggled off mid-press 
        # would synthesise a phantom click at the laser landing point.
        self._touch_trig_prev_l = 0.0
        self._touch_trig_prev_r = 0.0
        # Cursor ownership "latest click / tap wins".  When BOTH lasers are on
        # the virtual screen, _handle_cursor hands the cursor to whichever hand
        # most recently pulled its trigger.  We stamp the perf_counter time of
        # each on-screen trigger press per hand and compare the two stamps; the
        # newer one owns the cursor.  `_cursor_trig_prev_*` holds the prior
        # frame's trigger value so we can detect a true rising edge (a fresh
        # click) rather than a held trigger.
        self._cursor_click_ts_l  = 0.0
        self._cursor_click_ts_r  = 0.0
        self._cursor_trig_prev_l = 0.0
        self._cursor_trig_prev_r = 0.0

        # Target monitor for cursor mapping (multi-monitor support)
        self._target_mon_rect = None  # cached (left, top, width, height) in virtual-desktop pixels

        # Virtual keyboard
        self._keyboard_visible     = False
        self._keyboard_tex         = None  # moderngl Texture (RGBA, _KB_TEX_W × _KB_TEX_H)
        self._keyboard_vao         = None  # quad VAO using _overlay_prog
        self._keyboard_keys        = []    # list of _KeyEntry
        self._keyboard_width       = 1.6   # metres
        self._keyboard_height      = 0.33  # metres (6 rows)
        self._kb_show_shifted      = False # True render shifted labels on keys
        self._kb_last_build_width  = 0.0   # track width changes for texture rebuild
        self._kb_cached_position   = None  # None = use default; dict = user-placed position
        # World-space anchor of the keyboard centre. Re-snapped to the user's current
        # head pose every time the keyboard is toggled on so it materialises within
        # easy reach below the user's gaze direction (not at world origin).
        self._keyboard_pan_x       = 0.0
        self._keyboard_pan_y       = -0.2
        self._keyboard_distance    = 0.7   # metres in front of head
        self._keyboard_pitch       = math.radians(-35.0)   # tilt face up toward user
        self._keyboard_yaw         = 0.0                   # face the user's forward
        # Modifier state: each entry is [active, lock, last_tap_time]. Single tap arms
        # one-shot; double tap (<0.4s) engages persistent lock; tap while locked releases.
        self._mod_state = {
            'shift': [False, False, 0.0],
            'ctrl':  [False, False, 0.0],
            'alt':   [False, False, 0.0],
            'win':   [False, False, 0.0],
        }
        self._caps_lock            = False
        self._left_stick_click_prev= False
        self._scroll_accum_x       = 0.0   # fractional scroll accumulator horizontal
        self._scroll_accum_y       = 0.0   # fractional scroll accumulator vertical
        self._kb_trig_prev_l       = 0.0   # keyboard trigger debounce left controller
        self._kb_trig_prev_r       = 0.0   # keyboard trigger debounce right controller
        self._kb_hover_l           = None  # index of key under left laser, or None
        self._kb_hover_r           = None  # index of key under right laser, or None
        self._kb_held_key_l        = None  # index of key held by left trigger, or None
        self._kb_held_key_r        = None  # index of key held by right trigger, or None
        self._kb_held_mods_l       = None  # (shift, ctrl, alt, win, vk) snapshot for left held key
        self._kb_held_mods_r       = None  # (shift, ctrl, alt, win, vk) snapshot for right held key

        # GPU interop (CUDA / HIP) initialised lazily on first frame
        self._cuda_gl         = None   # CUDART_GL instance, False = permanently failed
        self._pbo_color       = None   # GL PBO id for RGB upload (GPU interop)
        self._pbo_depth       = None   # GL PBO id for depth upload (GPU interop)
        self._cpu_pbo_color   = None   # GL PBO id for CPU-path RGB upload
        self._cpu_pbo_depth   = None   # GL PBO id for CPU-path depth upload
        self._cpu_pbo_size    = (0, 0)
        self._cuda_res_color  = None   # registered resource handle
        self._cuda_res_depth  = None
        self._pbo_texture_size = None  # (w, h) at which PBOs were created

        # Font for in-VR overlay
        self.font = None
        self.label_font = None   # smaller font for section header labels
        self.font_type = get_font_type()
        self.base_font_size = 26
        # Match the bold font family so label and value share the same metrics
        _regular_fonts = [r"C:\Windows\Fonts\segoeui.ttf",
                          r"C:\Windows\Fonts\arial.ttf",
                          r"C:\Windows\Fonts\calibri.ttf"]
        for _rf in _regular_fonts:
            try:
                self.font = ImageFont.truetype(_rf, self.base_font_size)
                break
            except Exception:
                continue
        if self.font is None:
            try:
                self.font = ImageFont.truetype(self.font_type, self.base_font_size)
            except Exception:
                pass
        if self.font is None:
            try:
                self.font = ImageFont.load_default()
            except Exception:
                self.font = None
        try:
            self.label_font = ImageFont.truetype(self.font_type, 17)
        except Exception:
            self.label_font = self.font   # fall back to main font
        self.bold_font = None
        for _bf in (r"C:\Windows\Fonts\segoeuib.ttf",
                    r"C:\Windows\Fonts\arialbd.ttf",
                    r"C:\Windows\Fonts\calibrib.ttf"):
            try:
                self.bold_font = ImageFont.truetype(_bf, self.base_font_size)
                break
            except Exception:
                continue
        if self.bold_font is None:
            self.bold_font = self.font   # fall back to regular

        # In-VR FPS overlay GL resources
        self._overlay_prog     = None
        self._overlay_vao      = None
        self._overlay_tex      = None
        self._overlay_tex_size = (768, 238)  # 5-row info panel (1.25x taller)

        # Help/shortcut panel: anchored to right side of screen, shows key functions
        self._help_tex      = None
        self._help_vao      = None
        self._help_tex_size = (1, 1)  # Dynamic, _build_help_texture sets the actual size based on text layout

        # Depth-ratio OSD: floating panel that appears when depth_ratio changes
        self._depth_osd_tex       = None   # moderngl Texture (RGBA, 256×64)
        self._depth_osd_vao       = None   # quad VAO (reuses _overlay_prog)
        self._depth_osd_tex_size  = (256, 78)
        self._depth_osd_alpha     = 0.0    # current alpha (1 = fully visible, 0 = hidden)
        self._depth_osd_show_t    = -999.0 # perf_counter when OSD was last triggered
        self._depth_osd_last_val  = None   # last depth_ratio value rendered into the texture

        # Screen-preset OSD: shows preset name when user cycles through presets
        self._preset_osd_tex      = None   # moderngl Texture (RGBA, w×64)
        self._preset_osd_vao      = None
        self._preset_osd_tex_size = (768, 78)  # wide enough for long preset names
        self._preset_osd_alpha    = 0.0
        self._preset_osd_show_t   = -999.0
        self._preset_osd_last_key = None   # last preset index rendered into the texture

        # Brand switch OSD: right-controller-attached indicator
        self._brand_osd_tex       = None
        self._brand_osd_vao       = None
        self._brand_osd_tex_size  = (384, 112)
        self._brand_osd_alpha     = 0.0
        self._brand_osd_show_t    = -999.0
        self._brand_osd_last_name = None

        # Multi-brand controller models
        import os as _os
        _controllers_root = _os.path.join(_MODULE_DIR, 'controllers')
        self._controllers_root = _controllers_root
        self._all_models = {}
        self._available_brands = []
        self._current_brand = None
        self._brand_switch_osd_t = 0.0
        self._brand_sw_start = 0.0
        self._brand_sw_fired = False

        # Screen-info OSD: shows size + distance while right grip + right stick adjusts
        self._screen_osd_tex      = None   # moderngl Texture (RGBA, 512×64)
        self._screen_osd_vao      = None
        self._screen_osd_tex_size = (512, 78)
        self._screen_osd_show_t   = -999.0
        self._screen_osd_last_key = None   # (round(w,2), round(dist,2)) change detection

        # Screen border (slightly larger quad, solid color)
        self._border_prog = None
        self._border_vao  = None
        self._glow_vbo = None
        self._glow_band_params = None
        self._glow_model_params = None
        self._glow_model_mat = None
        self._curved_glow_prog = None
        self._curved_glow_vbo = None
        self._curved_glow_vao = None
        self._curved_glow_verts_params = None
        self._frost_glow_prog = None
        self._frost_glow_vbo = None
        self._frost_glow_vao = None
        self._frost_veil_prog = None
        self._frost_veil_vao = None
        self._frost_glow_verts_params = None
        self._curved_frost_prog = None
        self._curved_frost_vbo = None
        self._curved_frost_vao = None
        self._curved_veil_prog = None
        self._curved_veil_vao = None
        self._curved_frost_verts_params = None
        self._blit_prog = None
        self._blit_vao  = None
        self._panorama_prog = None
        self._panorama_vao = None
        self._panorama_vbo = None
        self._panorama_tex = None
        self._panorama_tex_path = None
        self._panorama_background_path = None
        self._panorama_background_settings = {}

        # Right thumbstick click state
        self._right_stick_click_prev = False

        # Stick click long-press + both-sticks overlay toggle
        self._lsc_press_t        = 0.0    # left stick click press start
        self._lsc_long_fired     = False
        self._rsc_press_t        = 0.0    # right stick click press start
        self._rsc_long_fired     = False
        self._both_stick_start   = 0.0    # both sticks pressed together timer
        self._both_stick_fired   = False

        # VR controller model offsets default zero, absolute offset loaded from profile.json
        self._ctrl_model_offset = [0.0, 0.0, 0.0]
        self._ctrl_model_rot_deg = 0.0

        # Controller real-time calibration
        self._calibration_mode = False
        self._calibration_temp_offset = [0.0, 0.0, 0.0]
        self._calibration_temp_rot = 0.0
        self._calib_combo_start = 0.0
        self._calib_combo_fired = False
        self._calib_tex = None
        self._calib_tex_size = (420, 200)  # Calibration combo instructions texture size

        # Right grip + A/B depth_ratio control (hold A = increase, hold B = decrease)
        # Reset to default: right grip + right thumbstick click

        # Physical mouse priority: suppress VR cursor when physical mouse is active
        self._phys_mouse_pos        = None   # (x, y) last seen physical cursor position
        self._phys_mouse_last_move  = 0.0    # perf_counter when physical mouse last moved
        self._vr_cursor_screen_pos  = None   # (px, py) last position written by VR laser
        self._last_get_cursor_pos_time = 0.0 # throttle GetCursorPos polling

        # Controller idle detection: skip input polling when no controllers tracked
        self._controller_miss_frames = 0     # consecutive frames with no controller pose

        # Curved screen mode
        self._screen_curved   = False   # True = cylindrical arc; False = flat quad
        self._screen_curve_axis = 'horizontal'  # horizontal | vertical; ignored when flat
        self._curved_prog     = None    # shader program (uses _CURVED_VERT)
        self._curved_vbo      = None    # dynamic VBO for arc strip vertices
        self._curved_vao      = None
        self._curved_border_prog = None  # solid-color arc border program
        self._curved_border_vbo  = None  # dynamic VBO for border arc vertices
        self._curved_border_vao  = None
        self._curved_verts_params = None        # curved screen VBO cache dirty flag
        self._curved_border_verts_params = None # border VBO cache dirty flag
        requested_curve = screen_curve if screen_curve is not None else kwargs.get('screen_curve', None)
        if requested_curve is not None:
            self._set_screen_curve_mode(requested_curve)

        # Background color index: 0 = black (default), 1 = green (passthrough,
        # toggled via long-press X). Left thumbstick click cycles environments
        # (see _cycle_environment).
        self._bg_color_idx    = 0       # index into _BG_COLORS

        # Cached XrPath handles populated by _init_controller_actions to avoid
        # calling xr.string_to_path on every frame (it's a round-trip into the runtime).
        self._path_left  = None
        self._path_right = None

        # Controller aim poses + laser pointer rendering
        self._act_aim_left  = None   # XrAction POSE_INPUT for left aim
        self._act_aim_right = None   # XrAction POSE_INPUT for right aim
        self._aim_space_l   = None   # XrSpace for left aim
        self._aim_space_r   = None   # XrSpace for right aim
        self._laser_vao     = None   # thin quad for laser beam
        self._dot_vao       = None   # small square for controller origin dot
        self._circle_vao    = None   # tessellated circle for hit-point indicator
        # Cached aim poses updated each frame (numpy 4x4 view-space matrices)
        self._aim_mat_l     = None
        self._aim_mat_r     = None

        # Controller grip poses + 3D model rendering
        self._act_grip_left  = None   # XrAction POSE_INPUT for left grip
        self._act_grip_right = None   # XrAction POSE_INPUT for right grip
        self._grip_space_l   = None   # XrSpace for left grip
        self._grip_space_r   = None   # XrSpace for right grip
        self._grip_mat_l     = None   # 4x4 world-space matrix
        self._grip_mat_r     = None   # 4x4 world-space matrix
        self._controller_prog   = None   # textured shader for controller
        self._ctrl_prims_l      = []     # list of {vao, vbo, ibo, tex_id} for left
        self._ctrl_prims_r      = []     # list of {vao, vbo, ibo, tex_id} for right
        self._ctrl_tex_cache    = {}     # tex_id -> moderngl Texture

        # Laser auto-hide: track last movement time and previous pose per controller
        _now = time.perf_counter()
        self._frame_now = _now   # per-frame timestamp; refreshed at top of main loop
        self._laser_last_move_l  = _now
        self._laser_last_move_r  = _now
        self._laser_prev_mat_l   = None
        self._laser_prev_mat_r   = None
        self._LASER_HIDE_AFTER   = 10.0   # seconds of idle before hiding
        self._LASER_MOVE_THRESH  = 0.015 # metres or radians minimum motion to count

        # Ray smoothing (quaternion SLERP + position EMA, simulates VD damping effect)
        self._smooth_ray_origin_l = None   # left hand smoothed position
        self._smooth_ray_origin_r = None   # right hand smoothed position
        self._smooth_ray_quat_l = None     # left hand smoothed direction (quaternion xyzw)
        self._smooth_ray_quat_r = None     # right hand smoothed direction (quaternion xyzw)
        self._smooth_ray_fwd_l  = None     # cached forward vector (pre-computed per frame)
        self._smooth_ray_fwd_r  = None
        self._pos_smooth = 0.02   # position smoothing (base damping)
        self._rot_smooth = 0.10   # rotation smoothing (base damping)
        self._ray_deadzone_rad = 0.0052  # angle dead zone (~0.3 degrees)
        self._ray_deadzone_pos = 0.002   # position dead zone (2mm)
        self._ray_edge_deadzone_rad = 0.1745  # edge snap release angle (~10 degrees)
        self._ray_edge_margin = 0.04   # edge deceleration region (~10cm at 2.4m screen)
        self._ray_edge_slow = 0.012  # smoothing factor after edge deceleration
        self._ray_prev_uv_l = None
        self._ray_prev_uv_r = None
        self._smooth_ray_prev_fwd_l = None  # previous frame smoothed direction (used for edge lock)
        self._smooth_ray_prev_fwd_r = None

        # One Euro Filter for raw controller position replaces EMA in
        # _apply_ray_smoothing.  Smooths the source data so laser beam,
        # cursor, and grip-to-move all benefit from the same stabilized input.
        # Tuned for VR controller tracking (meter-scale, ~90 fps):
        #   min_cutoff=8 Hz less lag, more responsive tracking
        #   beta=8         more stable at speed
        self._ray_filter_min_cutoff       = 8.0   # Hz (higher = less lag)
        self._ray_filter_beta             = 8.0   # speed sensitivity
        self._ray_filter_derivative_cutoff = 8.0  # Hz
        self._ray_filter_l = OneEuroFilter3D(
            self._ray_filter_min_cutoff,
            self._ray_filter_beta,
            self._ray_filter_derivative_cutoff)
        self._ray_filter_r = OneEuroFilter3D(
            self._ray_filter_min_cutoff,
            self._ray_filter_beta,
            self._ray_filter_derivative_cutoff)
        self._last_frame_dt = 0.011  # ~90fps default, updated each frame
        self._frame_count   = 0      # incremented each frame (cache invalidation)
        self._cached_beams  = None
        self._beams_frame   = -1

        # Screen position animation used by home-button gaze-reset to glide
        # smoothly instead of snapping.  None = no animation in progress.
        self._anim_target_pan_x    = None   # target screen_pan_x
        self._anim_target_pan_y    = None   # target screen_pan_y
        self._anim_target_distance = None   # target screen_distance
        self._anim_target_yaw      = None   # target screen_yaw
        self._anim_target_pitch    = None   # target screen_pitch
        self._anim_target_roll     = None   # target screen_roll

        # D3D11 backend state (populated by _init_d3d11_device when D3D11 path is active)
        self._use_d3d11             = False   # True = D3D11 OpenXR session + readback path
        self._d3d11_device          = None    # c_void_p ID3D11Device*
        self._d3d11_context         = None    # c_void_p ID3D11DeviceContext*
        self._d3d11_swapchain_fmt   = _DXGI_FORMAT_R8G8B8A8_UNORM_SRGB
        self._swapchain_is_bgra     = False  # True when WMR runtime only offers BGRA
        # Offscreen GL FBOs used when rendering for D3D11 swapchain images.
        # Key: (eye_index, img_index) (mgl_fbo, mgl_tex, raw_fbo_id, w, h)
        self._offscreen_fbo_cache   = {}
        # PBOs for async pixel readback in the D3D11 path.
        # Key: (eye_index, img_index) (pbo_id, w, h)
        self._d3d11_pbo_cache       = {}

        # GPU interop state (NV_DX_interop2 or EXT_memory_object) for zero-copy
        self._interop_mode      = None   # 'nv_dx' | 'ext_mem' | None (PBO fallback)
        self._nv_dx_device      = None   # HANDLE from wglDXOpenDeviceNV
        self._nv_dx_objects     = {}     # {img_index: GL_tex_id} for registered swapchain textures
        self._ext_shared_tex    = {}     # {(eye): (d3d11_tex, gl_mem_obj, gl_tex, mgl_fbo)}

        # ModernGL / GL handles
        self.window = None
        self.ctx = None
        self.prog = None
        self.quad_vao = None
        self.color_tex = None
        self.depth_tex = None
        self._texture_size = None

    # Initialisation helpers
    def _init_glfw(self):
        if not glfw.init():
            raise RuntimeError("[OpenXRViewer] GLFW init failed")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 5)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)   # hidden GL context only
        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
        self.window = glfw.create_window(1, 1, "XR Preview", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("[OpenXRViewer] GLFW window creation failed")
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)

        # Keyboard controls keep a reference so it isn't GC'd
        self._key_callback_ref = self._make_key_callback()
        glfw.set_key_callback(self.window, self._key_callback_ref)

    def _make_key_callback(self):
        viewer = self
        def _cb(window, key, scancode, action, mods):
            if action not in (glfw.PRESS, glfw.REPEAT):
                return
            d = 0.1; s = 0.15; p = 0.1; r = 0.05
            screen_locked = viewer._environment_screen_locked()
            if key == glfw.KEY_F:
                viewer._fps_overlay_visible = not viewer._fps_overlay_visible
            elif key == glfw.KEY_Z:
                viewer.depth_strength = max(0.0, viewer.depth_strength - 0.01)
            elif key == glfw.KEY_C:
                viewer.depth_strength = min(0.5, viewer.depth_strength + 0.01)
            elif key == glfw.KEY_X:
                viewer.depth_strength = 0.0   # flat mode no parallax distortion
            elif key == glfw.KEY_R:
                viewer._reset_screen_to_default(show_border=True)
            elif key == glfw.KEY_N:
                viewer._env_model_visible = not viewer._env_model_visible
            elif screen_locked:
                return
            elif key == glfw.KEY_W:     viewer.screen_distance = max(0.3, viewer.screen_distance - d)
            elif key == glfw.KEY_S:     viewer.screen_distance += d
            elif key == glfw.KEY_UP:    viewer.screen_pan_y += p
            elif key == glfw.KEY_DOWN:  viewer.screen_pan_y -= p
            elif key == glfw.KEY_LEFT:  viewer.screen_pan_x -= p
            elif key == glfw.KEY_RIGHT: viewer.screen_pan_x += p
            elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
                viewer._screen_ref_size += s; viewer.screen_height = None
            elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
                viewer._screen_ref_size = max(0.8, viewer._screen_ref_size - s)
                viewer.screen_height = None
            elif key == glfw.KEY_Q: viewer.screen_yaw += r
            elif key == glfw.KEY_E: viewer.screen_yaw -= r
            elif key == glfw.KEY_T: viewer.screen_pitch += r
            elif key == glfw.KEY_G: viewer.screen_pitch -= r
        return _cb

    def _init_moderngl(self):
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST)

        # World-space stereo rendering program (HMD eyes)
        self.prog = self.ctx.program(
            vertex_shader=_WORLD_VERT,
            fragment_shader=FRAGMENT_SHADER,
        )
        self.prog['u_convergence'].value = self.convergence
        self.prog['tex_color'].value = 0
        self.prog['tex_depth'].value = 1
        self.prog['u_source_crop'].value = (0.0, 0.0, 1.0, 1.0)

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

        # Glow quad (soft glow rendered behind the screen beyond the border)
        self._glow_prog = self.ctx.program(
            vertex_shader=_WORLD_VERT,
            fragment_shader=_GLOW_FRAG,
        )
        self._glow_vbo = self.ctx.buffer(reserve=24 * 4 * 4, dynamic=True)
        self._glow_vao = self.ctx.vertex_array(
            self._glow_prog, [(self._glow_vbo, '2f 2f', 'in_position', 'in_uv')]
        )

        self._frost_glow_prog = self.ctx.program(
            vertex_shader=_FROST_GLOW_VERT,
            fragment_shader=_FROST_GLOW_FRAG,
        )
        self._frost_glow_prog['u_source'].value = 0
        self._frost_glow_prog['u_source_crop'].value = (0.0, 0.0, 1.0, 1.0)
        self._frost_glow_vbo = self.ctx.buffer(reserve=24 * 5 * 4, dynamic=True)
        self._frost_glow_vao = self.ctx.vertex_array(
            self._frost_glow_prog,
            [(self._frost_glow_vbo, '3f 2f', 'in_position', 'in_uv')],
        )
        self._frost_veil_prog = self.ctx.program(
            vertex_shader=_FROST_GLOW_VERT,
            fragment_shader=_FROST_VEIL_FRAG,
        )
        self._frost_veil_prog['u_source'].value = 0
        self._frost_veil_prog['u_source_crop'].value = (0.0, 0.0, 1.0, 1.0)
        self._frost_veil_vao = self.ctx.vertex_array(
            self._frost_veil_prog,
            [(self._frost_glow_vbo, '3f 2f', 'in_position', 'in_uv')],
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
        # Flat beam (quadrilateral) + rainbow flow animation
        self._beam_prog = self.ctx.program(
            vertex_shader=_BEAM_VERT,
            fragment_shader=_BEAM_FRAG,
        )
        # Single quadrilateral: Y=0(base, thick) Y=1(tip, thin)
        beam_verts = np.array([
            -1.0, 0.0, 0.0, 0.0,   # bottom-left, v=0
            1.0, 0.0, 0.0, 0.0,   # bottom-right, v=0
            -0.15, 1.0, 0.0, 1.0,   # top-left, v=1
            0.15, 1.0, 0.0, 1.0,   # top-right, v=1
        ], dtype='f4')
        beam_vbo = self.ctx.buffer(beam_verts.tobytes())
        self._beam_vao = self.ctx.vertex_array(
            self._beam_prog,
            [(beam_vbo, '3f 4x 1f', 'in_position', 'in_v')],
        )
        # Hit-point indicator: tessellated circle (TRIANGLE_FAN), blue stroke + white fill
        N_SEG = 32
        circle_data = [0.0, 0.0, 0.0, 0.0]  # centre vertex
        for i in range(N_SEG + 1):
            a = 2.0 * math.pi * i / N_SEG
            circle_data.extend([math.cos(a), math.sin(a), 0.0, 0.0])
        self._circle_vao = self.ctx.vertex_array(
            self._border_prog,
            [(self.ctx.buffer(np.array(circle_data, dtype='f4').tobytes()), '2f 8x', 'in_position')],
        )
        # Controller origin dot: tiny square at the controller position
        self._dot_vao = self.ctx.vertex_array(
            self._border_prog,
            [(self.ctx.buffer(laser_verts.tobytes()), '2f 8x', 'in_position')],
        )

        # In-VR FPS overlay program (world-space quad, plain RGBA blit)
        self._overlay_prog = self.ctx.program(
            vertex_shader=_WORLD_VERT,
            fragment_shader=_OVERLAY_FRAG,
        )
        self._overlay_prog['tex'].value     = 2   # texture unit 2
        self._overlay_prog['u_alpha'].value = 1.0
        ow, oh = self._overlay_tex_size
        self._overlay_tex = self.ctx.texture((ow, oh), 4, dtype='f1')
        self._overlay_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        vbo2 = self.ctx.buffer(vertices.tobytes())
        self._overlay_vao = self.ctx.vertex_array(
            self._overlay_prog, [(vbo2, '2f 2f', 'in_position', 'in_uv')]
        )

        # Depth OSD: small floating panel (reuses _overlay_prog)
        dw, dh = self._depth_osd_tex_size
        self._depth_osd_tex = self.ctx.texture((dw, dh), 4, dtype='f1')
        self._depth_osd_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        vbo_dosd = self.ctx.buffer(vertices.tobytes())
        self._depth_osd_vao = self.ctx.vertex_array(
            self._overlay_prog, [(vbo_dosd, '2f 2f', 'in_position', 'in_uv')]
        )

        # Preset OSD: preset name panel (reuses _overlay_prog)
        pw, ph = self._preset_osd_tex_size
        self._preset_osd_tex = self.ctx.texture((pw, ph), 4, dtype='f1')
        self._preset_osd_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        vbo_posd = self.ctx.buffer(vertices.tobytes())
        self._preset_osd_vao = self.ctx.vertex_array(
            self._overlay_prog, [(vbo_posd, '2f 2f', 'in_position', 'in_uv')]
        )

        # Brand switch OSD: controller model indicator (reuses _overlay_prog)
        bw, bh = self._brand_osd_tex_size
        self._brand_osd_tex = self.ctx.texture((bw, bh), 4, dtype='f1')
        self._brand_osd_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        vbo_bosd = self.ctx.buffer(vertices.tobytes())
        self._brand_osd_vao = self.ctx.vertex_array(
            self._overlay_prog, [(vbo_bosd, '2f 2f', 'in_position', 'in_uv')]
        )

        # Seat adjust OSD: position indicator (reuses _overlay_prog)
        saw, sah = self._seat_adjust_osd_tex_size
        self._seat_adjust_osd_tex = self.ctx.texture((saw, sah), 4, dtype='f1')
        self._seat_adjust_osd_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        vbo_saosd = self.ctx.buffer(vertices.tobytes())
        self._seat_adjust_osd_vao = self.ctx.vertex_array(
            self._overlay_prog, [(vbo_saosd, '2f 2f', 'in_position', 'in_uv')]
        )

        # Screen-info OSD: size + distance panel (reuses _overlay_prog)
        sw, sh = self._screen_osd_tex_size
        self._screen_osd_tex = self.ctx.texture((sw, sh), 4, dtype='f1')
        self._screen_osd_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        vbo_sosd = self.ctx.buffer(vertices.tobytes())
        self._screen_osd_vao = self.ctx.vertex_array(
            self._overlay_prog, [(vbo_sosd, '2f 2f', 'in_position', 'in_uv')]
        )

        # Help/shortcut panel: anchored to right side of screen (reuses _overlay_prog)
        hw, hh = self._help_tex_size
        self._help_tex = self.ctx.texture((hw, hh), 4, dtype='f1')
        self._help_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        vbo_help = self.ctx.buffer(vertices.tobytes())
        self._help_vao = self.ctx.vertex_array(
            self._overlay_prog, [(vbo_help, '2f 2f', 'in_position', 'in_uv')]
        )
        self._build_help_texture()

        # Swizzle blit program for the D3D11/EXT_memory_object interop path.
        # Copies the offscreen RGBA eye-render into the shared GL texture, swapping
        # R/B when the OpenXR runtime exposes a BGRA D3D11 swapchain.
        self._blit_prog = self.ctx.program(
            vertex_shader=_WORLD_VERT,
            fragment_shader=_BLIT_FRAG,
        )
        self._blit_prog['u_src'].value = 4
        self._blit_prog['u_swap_rb'].value = 0
        self._blit_prog['u_mvp'].write(np.eye(4, dtype='f4').T.tobytes())
        _blit_verts = np.array([
            -1, -1, 0, 0,
             1, -1, 1, 0,
            -1,  1, 0, 1,
             1,  1, 1, 1,
        ], dtype='f4')
        self._blit_vao = self.ctx.vertex_array(
            self._blit_prog,
            [(self.ctx.buffer(_blit_verts.tobytes()), '2f 2f', 'in_position', 'in_uv')],
        )

        self._panorama_prog = self.ctx.program(
            vertex_shader=_PANORAMA_VERT,
            fragment_shader=_PANORAMA_FRAG,
        )
        self._panorama_prog['u_tex'].value = 8
        self._panorama_prog['u_yaw_offset'].value = 0.0
        self._panorama_prog['u_exposure'].value = 1.0
        self._panorama_prog['u_flip_y'].value = 0
        _pano_verts = np.array([
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
             1.0,  1.0,
        ], dtype='f4')
        self._panorama_vbo = self.ctx.buffer(_pano_verts.tobytes())
        self._panorama_vao = self.ctx.vertex_array(
            self._panorama_prog,
            [(self._panorama_vbo, '2f', 'in_position')],
        )

        # Curved screen program: same fragment shader, but world-space arc geometry
        # (no model matrix verts are built directly in world space each frame).
        self._curved_prog = self.ctx.program(
            vertex_shader=_CURVED_VERT,
            fragment_shader=FRAGMENT_SHADER,
        )
        self._curved_prog['u_convergence'].value = self.convergence
        self._curved_prog['tex_color'].value = 0
        self._curved_prog['tex_depth'].value  = 1
        self._curved_prog['u_source_crop'].value = (0.0, 0.0, 1.0, 1.0)
        # Allocate dynamic VBO large enough for N=48 segments × 2 verts × (3+2) floats.
        _CURVED_N = 48
        _curved_buf_bytes = (_CURVED_N + 1) * 2 * (3 + 2) * 4   # f4
        self._curved_vbo = self.ctx.buffer(reserve=_curved_buf_bytes, dynamic=True)
        self._curved_vao = self.ctx.vertex_array(
            self._curved_prog,
            [(self._curved_vbo, '3f 2f', 'in_position', 'in_uv')],
        )

        # Curved border: reuse _CURVED_VERT (vec3 pos + vec2 uv) with solid-color frag.
        # _SOLID_FRAG doesn't use UV, so the GLSL optimizer strips in_uv from the
        # linked program's active attribute table.  Only bind in_position (3f) and
        # skip the trailing 8 bytes of UV data same pattern as the flat border VAO.
        self._curved_border_prog = self.ctx.program(
            vertex_shader=_CURVED_VERT,
            fragment_shader=_SOLID_FRAG,
        )
        self._curved_border_vbo = self.ctx.buffer(reserve=_curved_buf_bytes, dynamic=True)
        self._curved_border_vao = self.ctx.vertex_array(
            self._curved_border_prog,
            [(self._curved_border_vbo, '3f 8x', 'in_position')],
        )

        self._curved_glow_prog = self.ctx.program(
            vertex_shader=_CURVED_VERT,
            fragment_shader=_GLOW_FRAG,
        )
        self._curved_glow_vbo = self.ctx.buffer(reserve=_curved_buf_bytes, dynamic=True)
        self._curved_glow_vao = self.ctx.vertex_array(
            self._curved_glow_prog,
            [(self._curved_glow_vbo, '3f 2f', 'in_position', 'in_uv')],
        )
        self._curved_frost_prog = self.ctx.program(
            vertex_shader=_FROST_CURVED_VERT,
            fragment_shader=_FROST_GLOW_FRAG,
        )
        self._curved_frost_prog['u_source'].value = 0
        self._curved_frost_prog['u_source_crop'].value = (0.0, 0.0, 1.0, 1.0)
        _curved_frost_buf_bytes = ((12 * _CURVED_N) + 12) * (3 + 2 + 3) * 4
        self._curved_frost_vbo = self.ctx.buffer(reserve=_curved_frost_buf_bytes, dynamic=True)
        self._curved_frost_vao = self.ctx.vertex_array(
            self._curved_frost_prog,
            [(self._curved_frost_vbo, '3f 2f 3f', 'in_position', 'in_uv', 'in_local')],
        )
        self._curved_veil_prog = self.ctx.program(
            vertex_shader=_FROST_CURVED_VERT,
            fragment_shader=_FROST_VEIL_FRAG,
        )
        self._curved_veil_prog['u_source'].value = 0
        self._curved_veil_prog['u_source_crop'].value = (0.0, 0.0, 1.0, 1.0)
        self._curved_veil_vao = self.ctx.vertex_array(
            self._curved_veil_prog,
            [(self._curved_frost_vbo, '3f 2f 3f', 'in_position', 'in_uv', 'in_local')],
        )

        # VR controller 3D model shader
        self._controller_prog = self.ctx.program(
            vertex_shader=_CTRL_VERT,
            fragment_shader=_CTRL_FRAG,
        )
        self._controller_prog['u_tex'].value = 3
        self._controller_prog['u_use_texture'].value = 1
        self._controller_prog['u_base_color_factor'].value = (1.0, 1.0, 1.0)
        self._controller_prog['u_light_color'].value = (0.60, 0.60, 0.65)
        self._controller_prog['u_ambient_color'].value = (0.22, 0.22, 0.24)

        # Environment model shader (no gl_FrontFacing discard, double-sided lighting)
        self._env_prog = self.ctx.program(
            vertex_shader=_ENV_VERT,
            fragment_shader=_ENV_FRAG,
        )
        self._env_prog['u_tex'].value = 3
        self._env_prog['u_use_texture'].value = 1
        self._env_prog['u_base_color_factor'].value = (1.0, 1.0, 1.0)
        self._env_prog['u_light_color'].value = (0.45, 0.45, 0.48)
        self._env_prog['u_ambient_color'].value = (0.08, 0.08, 0.09)
        self._env_prog['u_roughness'].value = 1.0
        self._env_prog['u_metallic'].value = 0.0
        self._env_prog['u_emissive_factor'].value = (0.0, 0.0, 0.0)
        self._env_prog['u_unlit'].value = 0
        self._env_prog['u_alpha_cutoff'].value = 0.5
        self._env_prog['u_alpha_mode'].value = 0
        self._env_prog['u_mr_tex'].value = 6
        self._env_prog['u_use_mr_tex'].value = 0
        self._env_prog['u_emissive_tex'].value = 7
        self._env_prog['u_use_emissive_tex'].value = 0
        self._env_prog['u_tex_offset'].value = (0.0, 0.0)
        self._env_prog['u_tex_scale'].value = (1.0, 1.0)
        self._env_prog['u_light_dir'].value = (0.0, -1.0, 0.0)
        self._env_prog['u_light_intensity'].value = (0.0, 0.0, 0.0)
        self._env_prog['u_normal_tex'].value = 4
        self._env_prog['u_use_normal_tex'].value = 0
        self._env_prog['u_normal_scale'].value = 1.0
        self._env_prog['u_occlusion_tex'].value = 5
        self._env_prog['u_use_occlusion_tex'].value = 0
        self._env_prog['u_occlusion_strength'].value = 1.0
        # Cinema bias-light defaults: disabled until _render_env_model writes
        # per-frame values.  Safe defaults keep the shader silent if the
        # _render_env_model path forgets to update them on a given frame.
        self._env_prog['u_screen_light_enabled'].value     = 0
        self._env_prog['u_screen_light_pos'].value         = (0.0, 0.0, -2.0)
        self._env_prog['u_screen_light_normal'].value      = (0.0, 0.0, 1.0)
        self._env_prog['u_screen_light_half_size'].value   = (1.2, 0.675)
        self._env_prog['u_screen_light_color'].value       = (0.3, 0.6, 1.0)
        self._env_prog['u_screen_light_intensity'].value   = 2.0
        # Fill lights: disabled by default (zero perf impact)
        self._env_prog['u_fill_light_enabled0'].value = 0
        self._env_prog['u_fill_light_pos0'].value = (0.0, 0.0, 0.0)
        self._env_prog['u_fill_light_color0'].value = (0.0, 0.0, 0.0)
        self._env_prog['u_fill_light_range0'].value = 1.0
        self._env_prog['u_fill_light_enabled1'].value = 0
        self._env_prog['u_fill_light_pos1'].value = (0.0, 0.0, 0.0)
        self._env_prog['u_fill_light_color1'].value = (0.0, 0.0, 0.0)
        self._env_prog['u_fill_light_range1'].value = 1.0
        # Post-processing defaults
        self._env_prog['u_env_exposure'].value = 1.0
        self._env_prog['u_env_gamma'].value = 2.2
        self._env_prog['u_emissive_strength'].value = 1.0
        self._env_prog['u_base_alpha'].value = 1.0

        self._ctrl_tex_cache = {}
        self._ctrl_prims_l = []
        self._ctrl_prims_r = []
        self._init_all_controller_models()
        # Environment model loaded lazily after OpenXR session starts

    def _init_openxr(self):
        """Try OpenGL first; fall back to D3D11 on Windows if OpenGL fails."""
        try:
            self._init_openxr_opengl()
            return
        except Exception as e:
            if self._is_openxr_device_unavailable(e):
                self._cleanup_partial_openxr()
                raise
            if sys.platform != "win32":
                raise
            print(f"[OpenXRViewer] OpenGL init failed ({e}), falling back to D3D11")
            self._cleanup_partial_openxr()

        self._init_openxr_d3d11()
        self._use_d3d11 = True

    def _is_openxr_device_unavailable(self, exc):
        """Return True when the runtime exists but no headset is currently available."""
        unavailable_cls = getattr(getattr(xr, 'exception', None), 'FormFactorUnavailableError', None)
        if unavailable_cls is not None and isinstance(exc, unavailable_cls):
            return True
        return exc.__class__.__name__ == 'FormFactorUnavailableError'

    def _wait_for_openxr_device(self, shutdown_event, retry_delay=2.0):
        """Retry OpenXR init until the headset is connected or startup is cancelled."""
        prompted = False
        while not glfw.window_should_close(self.window) and not shutdown_event.is_set():
            try:
                self._init_openxr()
                if prompted:
                    print("[OpenXRViewer] XR device connected; continuing startup")
                return True
            except Exception as exc:
                self._cleanup_partial_openxr()
                if not self._is_openxr_device_unavailable(exc):
                    raise
                if not prompted:
                    print("[OpenXRViewer] Waiting for XR device. Please connect or power on the headset...")
                    prompted = True
                end_t = time.perf_counter() + float(retry_delay)
                while time.perf_counter() < end_t:
                    glfw.poll_events()
                    if glfw.window_should_close(self.window) or shutdown_event.is_set():
                        return False
                    time.sleep(0.05)
        return False

    def _cleanup_partial_openxr(self):
        """Tear down any partially-initialised OpenXR + D3D11 state so a retry is clean."""
        for swapchain in self._xr_swapchains.values():
            try:
                xr.destroy_swapchain(swapchain)
            except Exception:
                pass
        self._xr_swapchains.clear()
        self._swapchain_images.clear()
        self._swapchain_sizes.clear()

        for attr in ("_xr_space", "_aim_space_l", "_aim_space_r", "_grip_space_l", "_grip_space_r"):
            sp = getattr(self, attr, None)
            if sp:
                try:
                    xr.destroy_space(sp)
                except Exception:
                    pass
                setattr(self, attr, None)

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

        self._xr_system_id = None

        # Release D3D11 COM objects if they were created
        for d3d_obj in (self._d3d11_context, self._d3d11_device):
            if d3d_obj is not None:
                try:
                    vtbl = ctypes.cast(d3d_obj, ctypes.POINTER(ctypes.c_void_p)).contents.value
                    release_fn = ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(
                        ctypes.cast(vtbl + 2 * ctypes.sizeof(ctypes.c_void_p),
                                    ctypes.POINTER(ctypes.c_void_p)).contents.value
                    )
                    release_fn(d3d_obj.value)
                except Exception:
                    pass
        self._d3d11_device  = None
        self._d3d11_context = None

    def _init_openxr_opengl(self):
        """Original OpenGL-backed OpenXR session."""
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

        # 2. System
        self._xr_system_id = xr.get_system(
            self._xr_instance,
            xr.SystemGetInfo(form_factor=xr.FormFactor.HEAD_MOUNTED_DISPLAY),
        )
        print("[OpenXRViewer] XrInstance created (OpenGL)")

        # 3. Verify GL requirements (mandatory before session creation)
        _pfn = ctypes.cast(
            xr.get_instance_proc_addr(self._xr_instance, "xrGetOpenGLGraphicsRequirementsKHR"),
            xr.PFN_xrGetOpenGLGraphicsRequirementsKHR,
        )
        _reqs = xr.GraphicsRequirementsOpenGLKHR()
        xr.check_result(xr.Result(_pfn(self._xr_instance, self._xr_system_id, ctypes.byref(_reqs))))

        # 4. Graphics binding platform-specific
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
        print("[OpenXRViewer] XrSession created (OpenGL)")

        # 6. Reference space prefer STAGE (floor origin), fall back to LOCAL
        available_spaces = xr.enumerate_reference_spaces(self._xr_session)
        ref_type = (
            xr.ReferenceSpaceType.STAGE
            if xr.ReferenceSpaceType.STAGE in available_spaces
            else xr.ReferenceSpaceType.LOCAL
        )
        self._xr_ref_space_type = ref_type
        self._xr_space = xr.create_reference_space(
            self._xr_session,
            xr.ReferenceSpaceCreateInfo(
                reference_space_type=ref_type,
                pose_in_reference_space=xr.Posef(),
            ),
        )

        # 7. Swapchains one per eye
        view_configs = xr.enumerate_view_configuration_views(
            self._xr_instance,
            self._xr_system_id,
            xr.ViewConfigurationType.PRIMARY_STEREO,
        )
        for eye_index, vcv in enumerate(view_configs):
            rec_w = vcv.recommended_image_rect_width
            rec_h = vcv.recommended_image_rect_height
            # Use exactly the recommended resolution this matches the HMD panel
            # pixel density and is what the runtime expects for correct reprojection.
            sc_w = rec_w & ~1
            sc_h = rec_h & ~1
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

        # 8. Controller actions (optional silently disabled if action set creation fails)
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
        # Cache hand XrPath values so per-frame action reads don't call string_to_path
        self._path_left  = subpaths[0]
        self._path_right = subpaths[1]

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
        self._act_x_btn     = make_bool("x_btn",      "X Button")
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

        # Grip pose actions used for placing controller 3D models
        self._act_grip_left = xr.create_action(
            self._action_set,
            xr.ActionCreateInfo(
                action_type=xr.ActionType.POSE_INPUT,
                action_name="grip_left",
                localized_action_name="Left Grip Pose",
                count_subaction_paths=1,
                subaction_paths=[subpaths[0]],
            ),
        )
        self._act_grip_right = xr.create_action(
            self._action_set,
            xr.ActionCreateInfo(
                action_type=xr.ActionType.POSE_INPUT,
                action_name="grip_right",
                localized_action_name="Right Grip Pose",
                count_subaction_paths=1,
                subaction_paths=[subpaths[1]],
            ),
        )

        # Per-profile binding table.
        # Use squeeze/value (float path) for grip the runtime auto-thresholds it
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
                ("/user/hand/left/input/x/click",            self._act_x_btn),
                ("/user/hand/left/input/y/click",            self._act_y_btn),
                ("/user/hand/left/input/trigger/value",      self._act_left_trigger),
                ("/user/hand/right/input/trigger/value",     self._act_right_trigger),
                ("/user/hand/left/input/aim/pose",           self._act_aim_left),
                ("/user/hand/right/input/aim/pose",          self._act_aim_right),
                ("/user/hand/left/input/grip/pose",          self._act_grip_left),
                ("/user/hand/right/input/grip/pose",         self._act_grip_right),
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
                ("/user/hand/left/input/grip/pose",         self._act_grip_left),
                ("/user/hand/right/input/grip/pose",        self._act_grip_right),
            ],
            # HTC Vive wand: trackpad (no thumbstick), squeeze/click (boolean,
            # no analog value), trigger value/click, menu no A/B/X/Y buttons.
            # The trackpad's 2D parent binds to the Vector2f stick actions, and
            # trackpad/click stands in for the thumbstick click.  Grip uses
            # squeeze/click directly since the wand has no analog squeeze.
            # Ref: https://registry.khronos.org/OpenXR/specs/1.0/man/html/openxr.html
            "/interaction_profiles/htc/vive_controller": [
                ("/user/hand/left/input/trackpad",           self._act_left_stick),
                ("/user/hand/right/input/trackpad",          self._act_right_stick),
                ("/user/hand/left/input/trackpad/click",     self._act_left_stick_click),
                ("/user/hand/right/input/trackpad/click",    self._act_right_stick_click),
                ("/user/hand/left/input/menu/click",         self._act_menu_btn),
                ("/user/hand/left/input/squeeze/click",      self._act_left_grip),
                ("/user/hand/right/input/squeeze/click",     self._act_right_grip),
                ("/user/hand/left/input/trigger/value",      self._act_left_trigger),
                ("/user/hand/right/input/trigger/value",     self._act_right_trigger),
                ("/user/hand/left/input/aim/pose",           self._act_aim_left),
                ("/user/hand/right/input/aim/pose",          self._act_aim_right),
                ("/user/hand/left/input/grip/pose",          self._act_grip_left),
                ("/user/hand/right/input/grip/pose",         self._act_grip_right),
            ],
            # KHR simple only has select/click (boolean) and menu no sticks or grip
            "/interaction_profiles/khr/simple_controller": [
                ("/user/hand/left/input/menu/click",    self._act_menu_btn),
                ("/user/hand/left/input/aim/pose",      self._act_aim_left),
                ("/user/hand/right/input/aim/pose",     self._act_aim_right),
                ("/user/hand/left/input/grip/pose",     self._act_grip_left),
                ("/user/hand/right/input/grip/pose",    self._act_grip_right),
            ],
            # PICO 4 Ultra controller interaction profile
            "/interaction_profiles/bytedance/pico_4u_controller": [
                ("/user/hand/left/input/thumbstick",         self._act_left_stick),
                ("/user/hand/right/input/thumbstick",        self._act_right_stick),
                ("/user/hand/left/input/thumbstick/click",   self._act_left_stick_click),
                ("/user/hand/right/input/thumbstick/click",  self._act_right_stick_click),
                ("/user/hand/left/input/menu/click",         self._act_menu_btn),
                ("/user/hand/left/input/squeeze/value",      self._act_left_grip),
                ("/user/hand/right/input/squeeze/value",     self._act_right_grip),
                ("/user/hand/right/input/a/click",           self._act_a_btn),
                ("/user/hand/right/input/b/click",           self._act_b_btn),
                ("/user/hand/left/input/x/click",            self._act_x_btn),
                ("/user/hand/left/input/y/click",            self._act_y_btn),
                ("/user/hand/left/input/trigger/value",      self._act_left_trigger),
                ("/user/hand/right/input/trigger/value",     self._act_right_trigger),
                ("/user/hand/left/input/aim/pose",           self._act_aim_left),
                ("/user/hand/right/input/aim/pose",          self._act_aim_right),
                ("/user/hand/left/input/grip/pose",          self._act_grip_left),
                ("/user/hand/right/input/grip/pose",         self._act_grip_right),
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

        # Pre-build the sync_actions arg now that the action set exists.
        # Same struct is reused every frame saves a list+struct allocation
        # per frame inside the hot loop.
        self._xr_actions_sync_info = xr.ActionsSyncInfo(active_action_sets=[
            xr.ActiveActionSet(
                action_set=self._action_set,
                subaction_path=xr.NULL_PATH,
            )
        ])

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

        # Create action spaces for grip poses (used to place controller 3D models)
        for act, attr in [
            (self._act_grip_left,  "_grip_space_l"),
            (self._act_grip_right, "_grip_space_r"),
        ]:
            if act is None:
                continue
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
                print(f"[OpenXRViewer] Grip space creation failed: {e}")

    def _init_textures(self, w, h):
        if self.color_tex:
            self.color_tex.release()
        if self.depth_tex:
            self.depth_tex.release()
        self.color_tex = self.ctx.texture((w, h), 3, dtype='f1')
        self.color_tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self._color_tex_mipmap_filter_active = True
        self.color_tex.repeat_x = False
        self.color_tex.repeat_y = False
        self.color_tex.build_mipmaps()
        try:
            self.color_tex.anisotropy = 16.0
        except Exception:
            pass
        # Negative LOD bias: bias the sampler toward sharper (higher-res) mip levels.
        # -0.5 = use a mip level 0.5 finer than the GPU would naturally pick,
        # preserving anti-aliasing while recovering perceived sharpness.
        glBindTexture(GL_TEXTURE_2D, self.color_tex.glo)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_LOD_BIAS, -0.5)
        glBindTexture(GL_TEXTURE_2D, 0)
        # Depth texture: plain LINEAR NO mipmaps.
        # Mipmapping depth averages foreground+background values at edges,
        # producing wrong depth that breaks the DIBR shift formula and
        # disocclusion detection. viewer.py (FullSBS reference) also uses
        # default LINEAR; this keeps openxr_viewer DIBR output numerically
        # consistent with viewer.py for the same RGB+depth input.
        self.depth_tex = self.ctx.texture((w, h), 1, dtype='f4')
        self.depth_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._texture_size = (w, h)

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

    def _init_cpu_pbos(self, w, h):
        """Create unpack PBOs for CPU-path texture upload (async DMA)."""
        try:
            ids = glGenBuffers(2)
            self._cpu_pbo_color = int(ids[0])
            self._cpu_pbo_depth = int(ids[1])
            color_bytes = w * h * 3
            depth_bytes = w * h * 4
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self._cpu_pbo_color)
            glBufferData(GL_PIXEL_UNPACK_BUFFER, color_bytes, None, GL_STREAM_DRAW)
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self._cpu_pbo_depth)
            glBufferData(GL_PIXEL_UNPACK_BUFFER, depth_bytes, None, GL_STREAM_DRAW)
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
            self._cpu_pbo_size = (w, h)
            print(f"[OpenXRViewer] CPU-path PBOs created {w}x{h}")
        except Exception as exc:
            print(f"[OpenXRViewer] CPU PBO init failed, using direct upload: {exc}")
            self._cpu_pbo_color = None
            self._cpu_pbo_depth = None
            self._cpu_pbo_size = (0, 0)

    def _sample_glow_target_color(self, rgb, is_tensor):
        """Update glow target from a thin frame border with minimal CPU work."""
        try:
            if is_tensor:
                # Tensor frames are CHW. Keep the border reduction on GPU and
                # transfer only the final 3 floats back to CPU.
                h, w = int(rgb.shape[1]), int(rgb.shape[2])
                x0, y0, x1, y1 = self._movie_crop_pixel_bounds(w, h)
                rgb = rgb[:, y0:y1, x0:x1]
                h, w = int(rgb.shape[1]), int(rgb.shape[2])
                bt = max(1, int(min(h, w) * 0.08))

                top_h = min(bt, h)
                bot_h = min(bt, h)
                total = rgb[:, :top_h, :].float().sum(dim=(1, 2))
                total = total + rgb[:, max(0, h - bot_h):, :].float().sum(dim=(1, 2))
                count = (top_h * w) + (bot_h * w)

                mid_h = max(0, h - top_h - bot_h)
                side_w = min(bt, w)
                if mid_h > 0 and side_w > 0:
                    y0 = top_h
                    y1 = h - bot_h
                    total = total + rgb[:, y0:y1, :side_w].float().sum(dim=(1, 2))
                    total = total + rgb[:, y0:y1, max(0, w - side_w):].float().sum(dim=(1, 2))
                    count += mid_h * side_w * 2

                avg = (total / max(1, count)).clamp(0, 255).detach().cpu().numpy()
                self._glow_target_color = (
                    float(avg[0]) / 255.0,
                    float(avg[1]) / 255.0,
                    float(avg[2]) / 255.0,
                )
                return

            rgb_np = np.asarray(rgb, dtype=np.uint8)
            h, w = rgb_np.shape[:2]
            x0, y0, x1, y1 = self._movie_crop_pixel_bounds(w, h)
            rgb_np = rgb_np[y0:y1, x0:x1, :]
            h, w = rgb_np.shape[:2]
            bt = max(1, int(min(h, w) * 0.08))
            top_h = min(bt, h)
            bot_h = min(bt, h)
            step = 4

            total = rgb_np[:top_h:step, ::step, :].sum(axis=(0, 1), dtype=np.float64)
            total += rgb_np[max(0, h - bot_h)::step, ::step, :].sum(axis=(0, 1), dtype=np.float64)
            count = (len(range(0, top_h, step)) + len(range(0, bot_h, step))) * len(range(0, w, step))

            mid_h = max(0, h - top_h - bot_h)
            side_w = min(bt, w)
            if mid_h > 0 and side_w > 0:
                y0 = top_h
                y1 = h - bot_h
                total += rgb_np[y0:y1:step, :side_w:step, :].sum(axis=(0, 1), dtype=np.float64)
                total += rgb_np[y0:y1:step, max(0, w - side_w)::step, :].sum(axis=(0, 1), dtype=np.float64)
                count += len(range(y0, y1, step)) * len(range(0, side_w, step)) * 2

            avg = total / max(1, count)
            self._glow_target_color = (
                float(avg[0]) / 255.0,
                float(avg[1]) / 255.0,
                float(avg[2]) / 255.0,
            )
        except Exception:
            pass

    def _movie_crop_profile_enabled(self):
        if not bool(getattr(self, '_auto_movie_crop', True)):
            return False

        profile = getattr(self, '_env_profile', None)
        screen = getattr(self, '_screen_profile', None)
        env_name = getattr(self, '_environment_model', None)
        key = (id(profile), id(screen), env_name)
        if key == getattr(self, '_movie_crop_profile_cache_key', None):
            return bool(getattr(self, '_movie_crop_profile_cache_value', False))

        if not isinstance(profile, dict):
            profile = {}
        if not isinstance(screen, dict):
            nested = profile.get('screen', {})
            screen = nested if isinstance(nested, dict) else {}

        enabled = None
        for source in (screen, profile):
            for name in ('auto_movie_crop', 'auto_crop', 'letterbox_crop'):
                if name in source:
                    value = source[name]
                    if isinstance(value, bool):
                        enabled = value
                    elif isinstance(value, str):
                        enabled = value.strip().lower() in ('1', 'true', 'yes', 'on')
                    else:
                        enabled = bool(value)
                    break
            if enabled is not None:
                break

        if enabled is None:
            enabled = str(env_name or '').strip().lower() != 'bedroom'

        self._movie_crop_profile_cache_key = key
        self._movie_crop_profile_cache_value = bool(enabled)
        return bool(enabled)

    def _movie_crop_is_active(self, crop=None):
        if crop is None:
            crop = getattr(self, '_movie_crop_target_uv', (0.0, 0.0, 1.0, 1.0))
        return (
            abs(float(crop[0])) > 1e-5
            or abs(float(crop[1])) > 1e-5
            or abs(float(crop[2]) - 1.0) > 1e-5
            or abs(float(crop[3]) - 1.0) > 1e-5
        )

    def _set_movie_crop_render_uv(self, crop):
        crop = tuple(float(v) for v in crop)
        old = getattr(self, '_movie_crop_render_uv', (0.0, 0.0, 1.0, 1.0))
        if max(abs(float(old[i]) - crop[i]) for i in range(4)) < 1e-5:
            self._movie_crop_frame_uv = crop
            return
        self._movie_crop_render_uv = crop
        self._movie_crop_frame_uv = crop
        self._movie_crop_render_active = self._movie_crop_is_active(crop)
        self._source_crop_uniform_cache = {}
        self._frost_uniform_cache = {}
        self._frost_layout_cache_key = None
        self._frost_layout_cache_val = None
        self._frost_model_cache_key = None
        self._frost_model_cache_bytes = None
        self._frost_model_uniform_cache = {}
        self.screen_height = None
        self._model_mat4_cache_key = None
        self._curved_verts_params = None
        self._curved_border_verts_params = None
        self._curved_glow_verts_params = None
        self._curved_frost_verts_params = None
        self._frost_glow_verts_params = None
        self._glow_band_params = None
        self._glow_model_params = None
        self._cl_pose_key = None

    def _reset_movie_crop(self):
        self._movie_crop_target_uv = (0.0, 0.0, 1.0, 1.0)
        self._movie_crop_target_active = False
        self._movie_crop_full_hits = 0
        self._movie_crop_reveal_until = 0.0
        self._movie_crop_pending_gpu = None
        self._set_movie_crop_render_uv((0.0, 0.0, 1.0, 1.0))

    def _refresh_movie_crop_profile_enabled(self):
        return self._movie_crop_profile_enabled()

    def _movie_crop_render_uv_fast(self):
        if getattr(self, '_movie_crop_target_active', False):
            reveal_until = float(getattr(self, '_movie_crop_reveal_until', 0.0))
            if reveal_until > 0.0:
                if time.perf_counter() < reveal_until:
                    if getattr(self, '_movie_crop_render_active', False):
                        self._set_movie_crop_render_uv((0.0, 0.0, 1.0, 1.0))
                    return (0.0, 0.0, 1.0, 1.0)
                self._movie_crop_reveal_until = 0.0
                if not getattr(self, '_movie_crop_render_active', False):
                    self._set_movie_crop_render_uv(getattr(self, '_movie_crop_target_uv', (0.0, 0.0, 1.0, 1.0)))
        return getattr(self, '_movie_crop_frame_uv', (0.0, 0.0, 1.0, 1.0))

    def _set_source_crop_uniform(self, prog, crop):
        crop = tuple(float(v) for v in crop)
        key = id(prog)
        cache = getattr(self, '_source_crop_uniform_cache', None)
        if not isinstance(cache, dict):
            cache = {}
            self._source_crop_uniform_cache = cache
        if cache.get(key) == crop:
            return
        prog['u_source_crop'].value = crop
        cache[key] = crop

    def _current_movie_crop_uv(self):
        if not self._refresh_movie_crop_profile_enabled():
            if getattr(self, '_movie_crop_render_active', False):
                self._reset_movie_crop()
            return (0.0, 0.0, 1.0, 1.0)

        if getattr(self, '_movie_crop_target_active', False) and time.perf_counter() < self._movie_crop_reveal_until:
            crop = (0.0, 0.0, 1.0, 1.0)
        else:
            crop = getattr(self, '_movie_crop_target_uv', (0.0, 0.0, 1.0, 1.0))
        self._set_movie_crop_render_uv(crop)
        return getattr(self, '_movie_crop_frame_uv', crop)

    def _movie_crop_pixel_bounds(self, w, h, crop=None):
        if crop is None:
            crop = self._current_movie_crop_uv()
        x, y, cw, ch = (float(crop[0]), float(crop[1]), float(crop[2]), float(crop[3]))
        x0 = max(0, min(int(round(x * w)), max(0, w - 1)))
        y0 = max(0, min(int(round(y * h)), max(0, h - 1)))
        x1 = max(x0 + 1, min(int(round((x + cw) * w)), w))
        y1 = max(y0 + 1, min(int(round((y + ch) * h)), h))
        return x0, y0, x1, y1

    def _effective_frame_aspect(self):
        fw, fh = self.frame_size
        crop = self._current_movie_crop_uv()
        eff_w = max(float(fw) * float(crop[2]), 1.0)
        eff_h = max(float(fh) * float(crop[3]), 1.0)
        return eff_h / eff_w

    def _apply_movie_crop_detection(self, detected, h):
        active = self._movie_crop_is_active(detected)
        if active:
            self._movie_crop_full_hits = 0
            old = self._movie_crop_target_uv
            if max(abs(float(old[i]) - float(detected[i])) for i in range(4)) >= (2.0 / max(h, 1)):
                self._movie_crop_target_uv = tuple(float(v) for v in detected)
                self._movie_crop_target_active = True
                self._current_movie_crop_uv()
            else:
                self._movie_crop_target_active = True
        else:
            self._movie_crop_full_hits += 1
            if self._movie_crop_full_hits >= 3 and getattr(self, '_movie_crop_target_active', False):
                self._reset_movie_crop()

    def _poll_movie_crop_gpu_result(self):
        pending = getattr(self, '_movie_crop_pending_gpu', None)
        if not pending:
            return False
        event = pending.get('event')
        try:
            if event is not None and not bool(event.query()):
                return True
            stats = pending['host'].tolist()
            detected = self._movie_crop_from_stats(stats, pending['y_rows'], pending['w'], pending['h'])
            self._apply_movie_crop_detection(detected, pending['h'])
        except Exception:
            self._apply_movie_crop_detection((0.0, 0.0, 1.0, 1.0), int(pending.get('h', 1)))
        finally:
            self._movie_crop_pending_gpu = None
        return False

    def _movie_crop_from_stats(self, stats, y_rows, w, h):
        top_i_f, bottom_count_f, center_mean, center_bright = stats
        n_rows = int(y_rows.shape[0])
        top_i = int(round(float(top_i_f)))
        bottom_count = int(round(float(bottom_count_f)))
        if top_i <= 0 or bottom_count <= 0 or top_i + bottom_count >= n_rows:
            return (0.0, 0.0, 1.0, 1.0)

        bottom_anchor_i = n_rows - bottom_count - 1
        if bottom_anchor_i < top_i:
            return (0.0, 0.0, 1.0, 1.0)
        top = int(y_rows[min(top_i, n_rows - 1)])
        bottom = int(h) - min(int(h), int(y_rows[bottom_anchor_i]) + 1)

        min_bar = max(8, int(h * 0.035))
        if top < min_bar or bottom < min_bar:
            return (0.0, 0.0, 1.0, 1.0)

        bigger = max(top, bottom)
        smaller = min(top, bottom)
        if bigger - smaller > max(18, int(bigger * 0.25)):
            return (0.0, 0.0, 1.0, 1.0)

        edge_trim = max(2, min(8, int(round(h * 0.004))))
        crop_top = max(0, min(top + edge_trim, h - 2))
        crop_bottom = max(crop_top + 1, h - bottom - edge_trim)
        crop_h = crop_bottom - crop_top
        removed = h - crop_h
        if removed < max(16, int(h * 0.07)):
            return (0.0, 0.0, 1.0, 1.0)

        full_aspect = w / max(float(h), 1.0)
        movie_aspect = w / max(float(crop_h), 1.0)
        if movie_aspect < full_aspect * 1.12:
            return (0.0, 0.0, 1.0, 1.0)

        if float(center_mean) < 14.0 or float(center_bright) < 0.035:
            return (0.0, 0.0, 1.0, 1.0)

        return (0.0, crop_top / float(h), 1.0, crop_h / float(h))

    def _movie_crop_sample_plan(self, w, h, device=None, torch_mod=None):
        device_type = getattr(device, 'type', None)
        device_index = getattr(device, 'index', None)
        key = (int(w), int(h), device_type, device_index)
        cache = getattr(self, '_movie_crop_sample_cache', None)
        if isinstance(cache, dict) and cache.get('key') == key:
            return cache

        x0 = int(w * 0.10)
        x1 = max(x0 + 1, int(w * 0.90))
        row_stride = max(1, (int(h) + 359) // 360)
        y_rows = np.arange(0, int(h), row_stride, dtype=np.int64)
        if y_rows.size == 0 or int(y_rows[-1]) != int(h) - 1:
            y_rows = np.append(y_rows, int(h) - 1)
        step_x = max(1, (x1 - x0) // 128)
        center_mask_np = (y_rows >= int(h * 0.35)) & (y_rows < int(h * 0.65))

        plan = {
            'key': key,
            'x0': x0,
            'x1': x1,
            'step_x': step_x,
            'y_rows': y_rows,
            'center_mask_np': center_mask_np,
            'center_has_rows': bool(np.any(center_mask_np)),
        }
        if torch_mod is not None and device is not None:
            plan['y_idx'] = torch_mod.as_tensor(y_rows, device=device, dtype=torch_mod.long)
            center_mask_t = torch_mod.as_tensor(center_mask_np, device=device, dtype=torch_mod.float32)
            plan['center_mask'] = center_mask_t
            plan['center_count'] = center_mask_t.sum().clamp_min(1.0)
            if device_type == 'cuda':
                try:
                    plan['host'] = torch_mod.empty((4,), dtype=torch_mod.float32, pin_memory=True)
                except Exception:
                    pass
                try:
                    plan['event'] = torch_mod.cuda.Event(blocking=False)
                except Exception:
                    pass
        self._movie_crop_sample_cache = plan
        return plan

    def _detect_movie_letterbox_crop(self, rgb, is_tensor, w, h, async_gpu=False):
        if w < 64 or h < 64:
            return (0.0, 0.0, 1.0, 1.0)

        dark_luma = 13.0
        dark_bright_frac = 0.040

        if is_tensor:
            torch = getattr(self, '_torch_mod', None)
            if torch is None:
                torch = __import__('torch')
                self._torch_mod = torch
            with torch.no_grad():
                device = getattr(rgb, 'device', None)
                plan = self._movie_crop_sample_plan(w, h, device=device, torch_mod=torch)
                y_rows = plan['y_rows']
                y_idx = plan['y_idx']
                x0 = plan['x0']
                x1 = plan['x1']
                step_x = plan['step_x']
                sample = rgb.index_select(1, y_idx)[:, :, x0:x1:step_x].detach().float()
                luma = sample[0] * 0.2126 + sample[1] * 0.7152 + sample[2] * 0.0722
                row_mean = luma.mean(dim=1)
                bright_frac = (luma > 20.0).float().mean(dim=1)
                dark_i = ((row_mean < dark_luma) & (bright_frac < dark_bright_frac)).to(torch.int32)
                top_i_t = torch.cumprod(dark_i, dim=0).sum().float()
                bottom_count_t = torch.cumprod(torch.flip(dark_i, dims=(0,)), dim=0).sum().float()
                center_mask = plan['center_mask']
                center_count = plan['center_count']
                center_mean_t = (row_mean * center_mask).sum() / center_count
                center_bright_t = (bright_frac * center_mask).sum() / center_count
                stats_t = torch.stack((top_i_t, bottom_count_t, center_mean_t, center_bright_t))
                if async_gpu and getattr(device, 'type', '') == 'cuda':
                    host = plan.get('host')
                    if host is None:
                        host = torch.empty((4,), dtype=torch.float32, pin_memory=True)
                        plan['host'] = host
                    host.copy_(stats_t.detach(), non_blocking=True)
                    event = plan.get('event')
                    if event is None:
                        event = torch.cuda.Event(blocking=False)
                        plan['event'] = event
                    event.record(torch.cuda.current_stream(device))
                    self._movie_crop_pending_gpu = {
                        'host': host,
                        'event': event,
                        'y_rows': y_rows,
                        'w': int(w),
                        'h': int(h),
                    }
                    return None
                stats = stats_t.detach().cpu().tolist()
            return self._movie_crop_from_stats(stats, y_rows, w, h)

        plan = self._movie_crop_sample_plan(w, h)
        y_rows = plan['y_rows']
        x0 = plan['x0']
        x1 = plan['x1']
        step_x = plan['step_x']
        arr = np.asarray(rgb, dtype=np.uint8)
        sample = arr[y_rows, x0:x1:step_x, :].astype(np.float32, copy=False)
        luma = sample[:, :, 0] * 0.2126 + sample[:, :, 1] * 0.7152 + sample[:, :, 2] * 0.0722
        row_mean_np = luma.mean(axis=1)
        bright_frac_np = (luma > 20.0).mean(axis=1)
        dark_rows = (row_mean_np < dark_luma) & (bright_frac_np < dark_bright_frac)
        top_i = 0
        n_rows = int(dark_rows.shape[0])
        while top_i < n_rows and bool(dark_rows[top_i]):
            top_i += 1
        bottom_count = 0
        while bottom_count < n_rows - top_i and bool(dark_rows[n_rows - 1 - bottom_count]):
            bottom_count += 1
        center_mask = plan['center_mask_np']
        if plan.get('center_has_rows', False):
            center_mean = float(np.mean(row_mean_np[center_mask]))
            center_bright = float(np.mean(bright_frac_np[center_mask]))
        else:
            center_mean = 0.0
            center_bright = 0.0
        return self._movie_crop_from_stats(
            (float(top_i), float(bottom_count), center_mean, center_bright),
            y_rows,
            w,
            h,
        )

    def _maybe_update_movie_crop(self, rgb, is_tensor, w, h):
        if not self._movie_crop_profile_enabled():
            self._movie_crop_pending_gpu = None
            if self._movie_crop_is_active(getattr(self, '_movie_crop_target_uv', None)) or self._movie_crop_is_active(getattr(self, '_movie_crop_render_uv', None)):
                self._reset_movie_crop()
            return

        if self._poll_movie_crop_gpu_result():
            return

        now = time.perf_counter()
        if now < self._movie_crop_next_detect_t:
            return
        self._movie_crop_next_detect_t = now + max(0.2, self._movie_crop_detect_interval)

        try:
            detected = self._detect_movie_letterbox_crop(rgb, is_tensor, w, h, async_gpu=bool(is_tensor))
        except Exception:
            detected = (0.0, 0.0, 1.0, 1.0)

        if detected is not None:
            self._apply_movie_crop_detection(detected, h)

    def _movie_crop_note_cursor_uv(self, u, v):
        if not getattr(self, '_movie_crop_target_active', False):
            return
        if not self._refresh_movie_crop_profile_enabled():
            return
        target = getattr(self, '_movie_crop_target_uv', (0.0, 0.0, 1.0, 1.0))
        margin = float(getattr(self, '_movie_crop_reveal_margin', 0.10))
        top_down_v = 1.0 - float(v)
        y0 = float(target[1])
        y1 = float(target[1] + target[3])
        should_reveal = False
        if not getattr(self, '_movie_crop_render_active', False):
            should_reveal = top_down_v <= y0 or top_down_v >= y1
        else:
            fv = float(v)
            should_reveal = fv <= margin or fv >= (1.0 - margin)
        if should_reveal:
            self._movie_crop_reveal_until = time.perf_counter() + 1.4
            self._current_movie_crop_uv()

    def _screen_uv_to_source_top_uv(self, u, v):
        crop = self._current_movie_crop_uv()
        return (
            float(crop[0]) + float(u) * float(crop[2]),
            float(crop[1]) + (1.0 - float(v)) * float(crop[3]),
        )

    # Per-frame helpers
    def _update_frame(self, rgb, depth):
        """Upload RGB and depth to GL textures GPU path when available, CPU fallback."""
        is_tensor = hasattr(rgb, 'data_ptr')
        torch = None
        if is_tensor:
            torch = getattr(self, '_torch_mod', None)
            if torch is None:
                torch = __import__('torch')
                self._torch_mod = torch
        glow_sample_rgb = rgb
        glow_sample_is_tensor = is_tensor

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
            self._movie_crop_sample_cache = None
            self._rgb_hwc_upload_tensor = None
            self._rgb_hwc_upload_key = None
            self._reset_movie_crop()
            self._target_mon_rect = None  # force re-query: resolution changed, cursor mapping must update

        active_glow_mode = self._active_glow_mode()
        self._active_glow_mode_cached = active_glow_mode
        color_mips_needed = self._color_mipmaps_needed(active_glow_mode)
        update_color_mips = (
            color_mips_needed
            or getattr(self, '_color_tex_mipmap_filter_active', None) != color_mips_needed
        )

        # Lazy GPU interop init (includes PBO registration to verify interop)
        if self._cuda_gl is None and CUDART_GL is not None and BACKEND in ("CUDA", "HIP"):
            try:
                self._cuda_gl = CUDART_GL()
                self._init_cuda_pbos(w, h)   # create PBOs + register with HIP
                print(f"[OpenXRViewer] GPU interop active ({BACKEND})")
            except Exception as e:
                print(f"[OpenXRViewer] GPU interop unavailable: {e}")
                self._cuda_gl = False   # sentinel: don't retry

        gpu_ok = bool(self._cuda_gl) and is_tensor and BACKEND in ("CUDA", "HIP")

        if gpu_ok:
            self._maybe_update_movie_crop(rgb, True, w, h)
            if self._pbo_texture_size != (w, h):
                self._init_cuda_pbos(w, h)

            # Color: CHW tensor HWC contiguous uint8 on GPU, DMA into PBO
            rgb_hwc = rgb.permute(1, 2, 0)
            if rgb_hwc.dtype == torch.uint8:
                upload_key = (int(w), int(h), getattr(rgb, 'device', None))
                rgb_gpu = self._rgb_hwc_upload_tensor
                if rgb_gpu is None or self._rgb_hwc_upload_key != upload_key:
                    rgb_gpu = torch.empty((int(h), int(w), 3), dtype=torch.uint8, device=rgb.device)
                    self._rgb_hwc_upload_tensor = rgb_gpu
                    self._rgb_hwc_upload_key = upload_key
                rgb_gpu.copy_(rgb_hwc, non_blocking=True)
            else:
                rgb_gpu = rgb_hwc.contiguous().clamp(0, 255).to(torch.uint8)
            ptr = self._cuda_gl.map_resource(self._cuda_res_color)
            self._cuda_gl.memcpy_d2d(ptr, rgb_gpu.data_ptr(), rgb_gpu.nbytes)
            self._cuda_gl.unmap_resource(self._cuda_res_color)
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self._pbo_color)
            glBindTexture(GL_TEXTURE_2D, self.color_tex.glo)
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
            if update_color_mips:
                self._maybe_generate_color_mipmaps(active_glow_mode)
            glBindTexture(GL_TEXTURE_2D, 0)

            ptr = self._cuda_gl.map_resource(self._cuda_res_depth)
            self._cuda_gl.memcpy_d2d(ptr, depth_gpu.contiguous().data_ptr(), depth_gpu.nbytes)
            self._cuda_gl.unmap_resource(self._cuda_res_depth)
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self._pbo_depth)
            glBindTexture(GL_TEXTURE_2D, self.depth_tex.glo)
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RED, GL_FLOAT, ctypes.c_void_p(0))
            # No glGenerateMipmap for depth: keep DIBR sampling at full-res
            # to match viewer.py FullSBS numerics.
            glBindTexture(GL_TEXTURE_2D, 0)
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        else:
            # CPU fallback - use PBO for async DMA when available
            if hasattr(rgb, 'detach'):
                rgb_hwc = rgb.permute(1, 2, 0).detach().contiguous()
                if rgb_hwc.dtype != torch.uint8:
                    rgb_hwc = rgb_hwc.clamp(0, 255).to(torch.uint8)
                rgb_np = rgb_hwc.cpu().numpy()
            else:
                rgb_np = np.asarray(rgb, dtype=np.uint8)
            glow_sample_rgb = rgb_np
            glow_sample_is_tensor = False
            self._maybe_update_movie_crop(rgb_np, False, w, h)
            if depth_np is None:
                depth_np = depth_gpu.cpu().numpy()
            rgb_bytes = rgb_np.astype('uint8', copy=False).tobytes()
            depth_bytes = depth_np.tobytes()
            cpu_pbo = getattr(self, '_cpu_pbo_color', None)
            if cpu_pbo is not None and self._cpu_pbo_size == (w, h):
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self._cpu_pbo_color)
                glBufferSubData(GL_PIXEL_UNPACK_BUFFER, 0, len(rgb_bytes), rgb_bytes)
                glBindTexture(GL_TEXTURE_2D, self.color_tex.glo)
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
                if update_color_mips:
                    self._maybe_generate_color_mipmaps(active_glow_mode)
                glBindTexture(GL_TEXTURE_2D, 0)
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self._cpu_pbo_depth)
                glBufferSubData(GL_PIXEL_UNPACK_BUFFER, 0, len(depth_bytes), depth_bytes)
                glBindTexture(GL_TEXTURE_2D, self.depth_tex.glo)
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RED, GL_FLOAT, ctypes.c_void_p(0))
                glBindTexture(GL_TEXTURE_2D, 0)
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
            else:
                if cpu_pbo is None:
                    self._init_cpu_pbos(w, h)
                self.color_tex.write(rgb_bytes)
                glBindTexture(GL_TEXTURE_2D, self.color_tex.glo)
                if update_color_mips:
                    self._maybe_generate_color_mipmaps(active_glow_mode)
                glBindTexture(GL_TEXTURE_2D, 0)
                self.depth_tex.write(depth_bytes)
            # No glGenerateMipmap for depth: keep DIBR sampling at full-res
            # to match viewer.py FullSBS numerics.

        # Sample screen-colour lighting only when an active effect consumes it.
        # Veil/frost sample the source texture directly in the shader, so they
        # should not trigger this periodic CPU-side colour reduction.
        glow_active = (
            active_glow_mode == 'glow'
            and float(getattr(self, '_glow_intensity', 0.0)) > 0.0
            and float(getattr(self, '_glow_intensity_multiplier', 0.0)) > 0.0
        )
        env_spill_active = (
            self._bg_color_idx != 1
            and
            bool(getattr(self, '_env_model_visible', False))
            and bool(getattr(self, '_env_model_prims', []))
            and float(getattr(self, '_screen_light_intensity', 0.0)) > 0.0
        )
        if glow_active or env_spill_active:
            self._glow_color_counter += 1
            if self._glow_color_counter >= 15:
                self._glow_color_counter = 0
                self._sample_glow_target_color(glow_sample_rgb, glow_sample_is_tensor)

    def _build_model_mat4(self, normal_offset=0.0):
        """
        Build the model matrix for the screen in world space.
        Caller must transpose before writing to OpenGL.
        """
        if self.screen_height is None:
            fw, fh = self.frame_size
            ar = self._effective_frame_aspect() if fw > 0 else 9.0 / 16.0
            if fh > fw:
                self.screen_height = self._screen_ref_size
                self.screen_width = self.screen_height / ar
            else:
                self.screen_width = self._screen_ref_size
                self.screen_height = self.screen_width * ar

        _key = (self.screen_yaw, self.screen_pitch, self.screen_roll,
                self.screen_pan_x, self.screen_pan_y, self.screen_distance,
                self.screen_width, self.screen_height, normal_offset)
        if _key == getattr(self, '_model_mat4_cache_key', None):
            return self._model_mat4_cache_val

        sx  = self.screen_width  / 2.0
        sy  = self.screen_height / 2.0
        cy  = math.cos(self.screen_yaw)
        sy_ = math.sin(self.screen_yaw)
        cp  = math.cos(self.screen_pitch)
        sp  = math.sin(self.screen_pitch)
        cr = math.cos(self.screen_roll)
        sr = math.sin(self.screen_roll)
        R = np.array([
            [ cy*cr + sy_*sp*sr, -cy*sr + sy_*sp*cr,  sy_*cp, 0],
            [ cp*sr,              cp*cr,              -sp,     0],
            [-sy_*cr + cy*sp*sr,  sy_*sr + cy*sp*cr,   cy*cp, 0],
            [ 0,                  0,                   0,      1],
        ], dtype=np.float32)
        S = np.diag([sx, sy, 1.0, 1.0]).astype(np.float32)
        T = np.eye(4, dtype=np.float32)
        T[0, 3] = self.screen_pan_x
        T[1, 3] = self.screen_pan_y
        T[2, 3] = -self.screen_distance
        if normal_offset:
            n = R[:3, 2]
            T[0, 3] += float(n[0]) * normal_offset
            T[1, 3] += float(n[1]) * normal_offset
            T[2, 3] += float(n[2]) * normal_offset
        result = T @ R @ S
        self._model_mat4_cache_key = _key
        self._model_mat4_cache_val = result
        return result

    def _screen_curve_mode(self):
        if not getattr(self, '_screen_curved', False):
            return 'none'
        axis = str(getattr(self, '_screen_curve_axis', 'horizontal') or 'horizontal').strip().lower()
        return axis if axis in ('horizontal', 'vertical') else 'horizontal'

    def _set_screen_curve_mode(self, mode):
        mode = str(mode or 'none').strip().lower()
        if mode in ('vertical', 'v'):
            self._screen_curved = True
            self._screen_curve_axis = 'vertical'
        elif mode in ('horizontal', 'h', 'curved', 'true'):
            self._screen_curved = True
            self._screen_curve_axis = 'horizontal'
        else:
            self._screen_curved = False
        self._curved_verts_params = None
        self._curved_border_verts_params = None
        self._curved_glow_verts_params = None
        self._curved_frost_verts_params = None

    def _cycle_screen_curve_mode(self):
        mode = self._screen_curve_mode()
        next_mode = {
            'none': 'horizontal',
            'horizontal': 'vertical',
            'vertical': 'none',
        }.get(mode, 'horizontal')
        self._set_screen_curve_mode(next_mode)
        label = {
            'horizontal': 'curved horizontally',
            'vertical': 'curved vertically',
            'none': 'no curve',
        }[next_mode]
        print(f"[OpenXRViewer] Screen curve: {label}")
        self._save_curve_to_active_profile()
        return next_mode

    def _build_curved_screen_verts(self, N=48, width_override=None, height_override=None,
                                   dist_offset=0.0, normal_offset=0.0):
        """Return a float32 numpy array for a TRIANGLE_STRIP curved screen arc.

        The arc is a cylinder section centred on (pan_x, pan_y, -distance) with a
        fixed angular span for all screen sizes. Wider screens get a larger
        radius, smaller screens get a smaller radius, and the curve angle stays
        consistent. Vertices are in world space so the shader only needs the
        view-projection matrix (no model matrix).

        width_override / height_override: use instead of screen_width/height (for border).
        dist_offset: added to screen_distance to push the surface slightly back.
        normal_offset: shifts the finished surface along the screen forward axis.

        Layout per vertex: x y z  u v  (5 floats).
        The strip has (N+1)*2 vertices one column pair per segment.
        """
        if self.screen_height is None:
            fw, fh = self.frame_size
            ar = self._effective_frame_aspect() if fw > 0 else 9.0 / 16.0
            if fh > fw:
                self.screen_height = self._screen_ref_size
                self.screen_width = self.screen_height / ar
            else:
                self.screen_width = self._screen_ref_size
                self.screen_height = self.screen_width * ar

        half_w = (width_override if width_override is not None else self.screen_width) / 2.0
        half_h = (height_override if height_override is not None else self.screen_height) / 2.0
        half_ang = min(_CURVED_HALF_ANGLE_RAD, math.pi / 2)
        axis = self._screen_curve_mode()
        if axis == 'none':
            axis = 'horizontal'

        Rm, center = self._screen_effect_basis()
        rot = Rm[:3, :3]
        normal = rot[:, 2]
        n_cols = N + 1
        angles = np.linspace(-half_ang, half_ang, n_cols)
        out = np.empty((n_cols * 2, 5), dtype=np.float32)

        if axis == 'vertical':
            radius = half_h / max(half_ang, 1e-6)
            vs = np.linspace(0.0, 1.0, n_cols)
            for i, (ang, v) in enumerate(zip(angles, vs)):
                ly = radius * math.sin(float(ang))
                lz = radius * (1.0 - math.cos(float(ang))) - dist_offset
                for j, (lx, u) in enumerate(((-half_w, 0.0), (half_w, 1.0))):
                    local = np.array([lx, ly, lz], dtype=np.float32)
                    world = center + rot @ local
                    if normal_offset:
                        world = world + normal * float(normal_offset)
                    out[i * 2 + j, 0:3] = world
                    out[i * 2 + j, 3] = u
                    out[i * 2 + j, 4] = v
        else:
            radius = half_w / max(half_ang, 1e-6)
            us = np.linspace(0.0, 1.0, n_cols)
            for i, (ang, u) in enumerate(zip(angles, us)):
                lx = radius * math.sin(float(ang))
                lz = radius * (1.0 - math.cos(float(ang))) - dist_offset
                for j, (ly, v) in enumerate(((-half_h, 0.0), (half_h, 1.0))):
                    local = np.array([lx, ly, lz], dtype=np.float32)
                    world = center + rot @ local
                    if normal_offset:
                        world = world + normal * float(normal_offset)
                    out[i * 2 + j, 0:3] = world
                    out[i * 2 + j, 3] = u
                    out[i * 2 + j, 4] = v

        return out.ravel()

    def _get_or_create_fbo(self, eye_index, image_index, texture_id, w, h):
        """Lazily create and cache a ModernGL Framebuffer wrapping the swapchain texture.

        ctx.detect_framebuffer() is used so ModernGL's internal state tracking stays
        consistent raw glBindFramebuffer() is invisible to ModernGL and would cause
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
        depth_rb = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, depth_rb)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_rb)
        glBindRenderbuffer(GL_RENDERBUFFER, 0)
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        if status != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(
                f"[OpenXRViewer] FBO incomplete for eye {eye_index}, "
                f"image {image_index}: {status:#x}"
            )
        mgl_fbo = self.ctx.detect_framebuffer(raw_id)
        self._fbo_cache[key] = (raw_id, mgl_fbo)
        self._depth_rb_cache[key] = depth_rb
        return raw_id, mgl_fbo

    def _update_aim_poses(self, display_time):
        """Locate both controller aim spaces and cache their world-space 4×4 matrices."""
        now = self._frame_now
        for space, mat_attr, prev_attr, move_attr in [
            (self._aim_space_l, "_aim_mat_l", "_laser_prev_mat_l", "_laser_last_move_l"),
            (self._aim_space_r, "_aim_mat_r", "_laser_prev_mat_r", "_laser_last_move_r"),
        ]:
            if space is None:
                setattr(self, mat_attr, None)
                continue
            try:
                loc = xr.locate_space(space, self._xr_space, display_time)
                if loc.location_flags & xr.SPACE_LOCATION_POSITION_VALID_BIT:
                    R = _xr_quat_to_mat4(loc.pose.orientation)
                    R[:3, 3] = [loc.pose.position.x, loc.pose.position.y, loc.pose.position.z]
                    setattr(self, mat_attr, R)
                    # Compare position+orientation to previous pose to detect motion
                    prev = getattr(self, prev_attr)
                    if prev is not None:
                        pos_delta = float(np.linalg.norm(R[:3, 3] - prev[:3, 3]))
                        # Rotation difference via Frobenius norm of delta rotation matrix
                        rot_delta = float(np.linalg.norm(R[:3, :3] - prev[:3, :3]))
                        if pos_delta > self._LASER_MOVE_THRESH or rot_delta > self._LASER_MOVE_THRESH:
                            setattr(self, move_attr, now)
                    else:
                        setattr(self, move_attr, now)
                    setattr(self, prev_attr, R.copy())
                else:
                    setattr(self, mat_attr, None)
            except Exception:
                setattr(self, mat_attr, None)

    def _update_grip_poses(self, display_time):
        """Locate controller grip spaces and cache 4x4 world-space matrices.
        Controller 3D models are placed at the grip center (aim pose is at the tracking ring).
        Also update movement timestamps for 5-second idle auto-hide."""
        now = self._frame_now
        for space, mat_attr, move_attr in [
            (self._grip_space_l, "_grip_mat_l", "_laser_last_move_l"),
            (self._grip_space_r, "_grip_mat_r", "_laser_last_move_r"),
        ]:
            if space is None:
                setattr(self, mat_attr, None)
                continue
            try:
                loc = xr.locate_space(space, self._xr_space, display_time)
                if loc.location_flags & xr.SPACE_LOCATION_POSITION_VALID_BIT:
                    R = _xr_quat_to_mat4(loc.pose.orientation)
                    R[:3, 3] = [loc.pose.position.x, loc.pose.position.y, loc.pose.position.z]
                    # Detect movement
                    prev = getattr(self, mat_attr)
                    if prev is not None:
                        delta = float(np.linalg.norm(R[:3, 3] - prev[:3, 3]))
                        if delta > self._LASER_MOVE_THRESH:
                            setattr(self, move_attr, now)
                    else:
                        setattr(self, move_attr, now)
                    setattr(self, mat_attr, R)
                else:
                    setattr(self, mat_attr, None)
            except Exception:
                setattr(self, mat_attr, None)

    def _reset_orientation_offsets(self):
        """Clear manual yaw/pitch offsets so screen faces head baseline."""
        self._yaw_offset   = 0.0
        self._pitch_offset = 0.0

    def _clear_screen_grab_anchors(self):
        """Drop every stored grip-to-move anchor.

        Called whenever the screen is teleported by a reset (Y button,
        long-press Home, preset switch, etc.) so a grip that is held
        across the reset re-anchors at the screen's NEW pose on the next
        frame instead of dragging it back to the pre-reset position.

        The grip handler in ``_handle_triggers`` records ``saved_local``
        (the 2-D screen-local hit point at first grip press) and keeps
        moving the screen so the laser continues to hit that exact local
        point.  Without this clear, a reset's new position is overridden
        within one frame by that stale anchor the screen "snaps back".
        """
        self._screen_grab_local_l = None
        self._screen_grab_local_r = None
        self._screen_grab_grip_l  = None
        self._screen_grab_grip_r  = None
        self._kb_grab_local_l     = None
        self._kb_grab_local_r     = None

    def _screen_pose_mat4(self):
        """Screen rigid transform without size scale."""
        cy = math.cos(self.screen_yaw);   sy = math.sin(self.screen_yaw)
        cp = math.cos(self.screen_pitch); sp = math.sin(self.screen_pitch)
        cr = math.cos(self.screen_roll);  sr = math.sin(self.screen_roll)
        ry = np.array([[cy, 0, sy, 0],
                       [0, 1, 0, 0],
                       [-sy, 0, cy, 0],
                       [0, 0, 0, 1]], dtype='f4')
        rx = np.array([[1, 0, 0, 0],
                       [0, cp, -sp, 0],
                       [0, sp, cp, 0],
                       [0, 0, 0, 1]], dtype='f4')
        rz = np.array([[cr, -sr, 0, 0],
                       [sr, cr, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]], dtype='f4')
        trans = np.eye(4, dtype='f4')
        trans[0, 3] = self.screen_pan_x
        trans[1, 3] = self.screen_pan_y
        trans[2, 3] = -self.screen_distance
        return trans @ ry @ rx @ rz

    def _decompose_env_model_mat4(self, mat):
        """Store a model matrix back into env pos / yaw-pitch-roll / scale."""
        mat = np.asarray(mat, dtype=np.float64)
        self._env_model_pos = [float(mat[0, 3]), float(mat[1, 3]), float(mat[2, 3])]
        rot_scale = mat[:3, :3].copy()
        scale = [float(np.linalg.norm(rot_scale[:, i])) for i in range(3)]
        for i, s in enumerate(scale):
            if s > 1e-9:
                rot_scale[:, i] /= s
        pitch = math.asin(max(-1.0, min(1.0, -float(rot_scale[1, 2]))))
        yaw = math.atan2(float(rot_scale[0, 2]), float(rot_scale[2, 2]))
        roll = math.atan2(float(rot_scale[1, 0]), float(rot_scale[1, 1]))
        self._env_model_rot = [yaw, pitch, roll]
        self._env_model_scale = scale

    def _move_env_with_screen_delta(self, old_screen_mat):
        """When locked, apply the same rigid screen reset delta to the env."""
        if not self._environment_screen_locked() or old_screen_mat is None:
            return
        try:
            new_screen_mat = self._screen_pose_mat4().astype('f8')
            delta = new_screen_mat @ np.linalg.inv(old_screen_mat.astype('f8'))
            env_new = delta @ self._env_model_mat4().astype('f8')
            self._decompose_env_model_mat4(env_new)
        except Exception as exc:
            print(f"[OpenXRViewer] screen/env lock transform failed: {exc}")

    def _reset_screen_to_gaze(self, show_border=False):
        """Instantly snap the screen to 2 m in front of the current head gaze.

        Size and shape are preserved only position/orientation are updated.
        """
        RESET_DIST = 2.0
        old_screen_mat = self._screen_pose_mat4() if self._environment_screen_locked() else None
        self._anim_target_pan_x = None  # cancel any stale animation
        self._reset_orientation_offsets()
        self._clear_screen_grab_anchors()
        if self._head_pos_w is not None and self._head_fwd_w is not None:
            hx, hy, hz = self._head_pos_w
            fx, fy, fz = self._head_fwd_w
            flen = math.sqrt(fx*fx + fy*fy + fz*fz)
            if flen > 1e-4:
                fx /= flen; fy /= flen; fz /= flen
            else:
                fx, fy, fz = 0.0, 0.0, -1.0
            # World-space target point
            tx = hx + fx * RESET_DIST
            ty = hy + fy * RESET_DIST
            tz = hz + fz * RESET_DIST
            horiz = math.sqrt(fx*fx + fz*fz)
            yaw   = math.atan2(-fx, -fz) if horiz > 1e-4 else self.screen_yaw
            pitch = math.asin(max(-0.999, min(0.999, fy)))
            # Invert model matrix (rot_y @ rot_x then translate) to get local coords
            cy, sy_ = math.cos(yaw),   math.sin(yaw)
            cp, sp  = math.cos(pitch), math.sin(pitch)
            # Inverse rot_y
            x1 =  cy * tx - sy_ * tz
            y1 =  ty
            z1 =  sy_ * tx + cy * tz
            # Inverse rot_x
            x2 =  x1
            y2 =  cp * y1 + sp * z1
            z2 = -sp * y1 + cp * z1
            self.screen_pan_x    = x2
            self.screen_pan_y    = y2
            self.screen_distance = -z2
            self.screen_yaw      = yaw
            self.screen_pitch    = pitch
            self.screen_roll     = 0.0
        else:
            self.screen_distance = RESET_DIST
            self.screen_pan_x    = 0.0
            self.screen_pan_y    = float(self._initial_head_y)
            self.screen_pitch    = 0.0
            self.screen_yaw      = 0.0
            self.screen_roll     = 0.0
        self._move_env_with_screen_delta(old_screen_mat)
        if show_border:
            self._border_alpha  = 1.0
            self._border_idle_t = time.perf_counter()
        if self._keyboard_visible:
            self._anchor_keyboard_below_screen()

    def _kb_restore_cached_position(self, cached):
        """Restore keyboard to a previously cached position."""
        self._keyboard_pan_x    = float(cached['pan_x'])
        self._keyboard_pan_y    = float(cached['pan_y'])
        self._keyboard_distance = float(cached['distance'])
        self._keyboard_width    = float(cached.get('width', self.screen_width * 0.75))
        self._keyboard_yaw      = float(cached.get('yaw', 0.0))
        self._keyboard_pitch    = float(cached.get('pitch', 0.0))

    def _tick_screen_anim(self, dt):
        """Exponential-decay glide toward the animation target set by _reset_screen_to_gaze.

        Uses a critically-damped-style lerp: alpha = 1 - exp(-k * dt), which gives
        frame-rate-independent smoothing.  k controls speed: higher = snappier.
        Clears targets once the screen is close enough to avoid infinite ticking.
        """
        if self._anim_target_pan_x is None and self._anim_target_roll is None:
            return

        K     = 6.0   # decay constant: ~63% of the gap closed per 1/K seconds
        alpha = 1.0 - math.exp(-K * max(dt, 1e-4))

        def _lerp(a, b): return a + alpha * (b - a)
        def _lerp_angle(a, b):
            # Shortest-path lerp for angles
            d = (b - a + math.pi) % (2 * math.pi) - math.pi
            return a + alpha * d

        self.screen_pan_x    = _lerp(self.screen_pan_x,    self._anim_target_pan_x)
        self.screen_pan_y    = _lerp(self.screen_pan_y,    self._anim_target_pan_y)
        self.screen_distance = _lerp(self.screen_distance, self._anim_target_distance)
        self.screen_yaw      = _lerp_angle(self.screen_yaw,   self._anim_target_yaw)
        self.screen_pitch    = _lerp_angle(self.screen_pitch, self._anim_target_pitch)
        self.screen_roll     = _lerp_angle(self.screen_roll,  self._anim_target_roll)

        # Stop animating once close enough (< 1 mm / 0.01°)
        close = (
            abs(self.screen_pan_x    - self._anim_target_pan_x)    < 0.001 and
            abs(self.screen_pan_y    - self._anim_target_pan_y)    < 0.001 and
            abs(self.screen_distance - self._anim_target_distance) < 0.001 and
            abs((self.screen_yaw   - self._anim_target_yaw   + math.pi) % (2*math.pi) - math.pi) < 0.0002 and
            abs((self.screen_pitch - self._anim_target_pitch + math.pi) % (2*math.pi) - math.pi) < 0.0002 and
            abs((self.screen_roll  - self._anim_target_roll  + math.pi) % (2*math.pi) - math.pi) < 0.0002
        )
        if close:
            self.screen_pan_x    = self._anim_target_pan_x
            self.screen_pan_y    = self._anim_target_pan_y
            self.screen_distance = self._anim_target_distance
            self.screen_yaw      = self._anim_target_yaw
            self.screen_pitch    = self._anim_target_pitch
            self.screen_roll     = self._anim_target_roll
            self._anim_target_pan_x = None   # clear animation complete

    def _reset_screen_direction(self):
        """Reset screen to face the user's current gaze, preserving distance and size."""
        if self._head_pos_w is None or self._head_fwd_w is None:
            return
        old_screen_mat = self._screen_pose_mat4() if self._environment_screen_locked() else None
        hx, hy, hz = self._head_pos_w
        fx, fy, fz = self._head_fwd_w
        flen = math.sqrt(fx*fx + fy*fy + fz*fz)
        if flen > 1e-4:
            fx /= flen; fy /= flen; fz /= flen
        else:
            fx, fy, fz = 0.0, 0.0, -1.0

        # Preserve Euclidean distance from head to screen centre
        dx = self.screen_pan_x - hx
        dy = self.screen_pan_y - hy
        dz = -self.screen_distance - hz
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)

        # Target screen centre in world space, following head gaze
        tx = hx + fx * dist
        ty = hy + fy * dist
        tz = hz + fz * dist
        pitch = math.asin(max(-0.999, min(0.999, fy)))

        # Wipe ALL cached orientation state BEFORE writing the new pose so
        # the next grip-drag frame (which recomputes screen_yaw/pitch as
        # `base_yaw + self._yaw_offset` and `base_pitch + self._pitch_offset`)
        # re-latches at the fresh face-head baseline instead of snapping
        # back to the user's pre-reset yaw/pitch tilt.  This is the same
        # invariant the Y-button and preset-switch reset paths maintain
        # via `_reset_orientation_offsets()`.
        self._reset_orientation_offsets()
        # Drop any stale grip anchors so a grip held across this reset
        # re-latches at the screen's new pose on the next frame instead
        # of dragging the screen back to its pre-reset position.
        self._clear_screen_grab_anchors()
        # Also cancel any in-flight glide animation that might be heading
        # toward the old pre-reset target.
        self._anim_target_pan_x    = None
        self._anim_target_pan_y    = None
        self._anim_target_distance = None
        self._anim_target_yaw      = None
        self._anim_target_pitch    = None
        self._anim_target_roll     = None

        self.screen_pan_x = tx
        self.screen_pan_y = ty
        self.screen_distance = -tz
        self.screen_yaw = math.atan2(-fx, -fz)
        self.screen_pitch = pitch
        # Home long-press / playspace recenter also flattens any user roll
        # back to upright matches the Meta-style "recenter view" contract
        # where the screen returns to the canonical level orientation.
        self.screen_roll = 0.0
        self._move_env_with_screen_delta(old_screen_mat)

        self._border_alpha = 1.0
        self._border_idle_t = time.perf_counter()

    def _reset_locked_environment_to_profile(self, show_border=False):
        """Restore a locked room's calibrated environment/screen pose."""
        if not self._active_environment:
            return False
        return self._apply_profile_screen_layout(show_border=show_border)

    def _apply_preset(self, index):
        """Apply screen preset: size, distance, and reposition to face the user."""
        if self._environment_screen_locked():
            self._preset_index = index
            if self._reset_locked_environment_to_profile(show_border=True):
                return
            self._reset_screen_to_default(show_border=True)
            return

        name, width, dist = self._screen_presets[index]
        self._reset_orientation_offsets()
        self._clear_screen_grab_anchors()
        self.screen_width    = width
        self._screen_ref_size = width
        self.screen_height   = None
        self.screen_pitch    = 0.0
        self.screen_roll     = 0.0
        if not self._environment_screen_locked():
            self._set_screen_curve_mode('none')
        self._preset_index            = index
        self.screen_pan_y    = float(self._initial_head_y)

        if self._head_pos_w is not None and self._head_fwd_w is not None:
            hx, hy, hz = self._head_pos_w
            fx, fy, fz = self._head_fwd_w
            flen = math.sqrt(fx*fx + fy*fy + fz*fz)
            if flen > 1e-4:
                fx /= flen; fy /= flen; fz /= flen
            else:
                fx, fy, fz = 0.0, 0.0, -1.0
            self.screen_pan_x    = hx + fx * dist
            self.screen_distance = -(hz + fz * dist)
            self.screen_yaw      = math.atan2(-fx, -fz)
        else:
            self.screen_pan_x    = 0.0
            self.screen_distance = dist
            self.screen_yaw      = 0.0

        self._preset_name_overlay = f"{name}  {width:.2f}m / {dist:.2f}m"
        self._last_overlay_update = 0.0
        self._border_alpha  = 1.0
        self._border_idle_t = time.perf_counter()

    def _reset_seating_vertical(self):
        """Adjust the XR space offset vertically so the user's eye height
        matches the profile's intended viewing height.

        Only the Y component of the space offset is updated; horizontal
        position and facing direction are preserved so the room doesn't
        rotate or slide sideways.
        """
        view = getattr(self, '_view_pose_profile', {}) or {}
        if not isinstance(view, dict) or not view:
            return
        if 'y' in view:
            x, y, z, angle = self._seat_adjust_current_pos()
            self._apply_seat_adjust_xr_space(x, y, z, angle)
            return
        views = getattr(self, '_last_located_views', None)
        if not views or views[0] is None or views[1] is None:
            return
        raw_head = self._head_model_mat4_from_views(views)
        if raw_head is None:
            return
        auto_center = bool(view.get('auto_center_on_screen', False))
        pos_keys = ('position', 'camera_position', 'viewer_position')
        rot_deg_keys = ('rotation_deg', 'camera_rotation_deg', 'viewer_rotation_deg')
        rot_rad_keys = ('rotation', 'camera_rotation', 'viewer_rotation')
        has_rot = any(key in view for key in rot_deg_keys + rot_rad_keys)
        has_pos = any(key in view for key in pos_keys)
        if auto_center:
            auto_pos = self._auto_view_position_from_screen(view, has_rot, rot_deg_keys, rot_rad_keys)
            if auto_pos is None and has_pos:
                auto_pos = self._profile_vec3(view, pos_keys, [0.0, 0.0, 0.0])
            if auto_pos is None:
                return
            desired_y = float(auto_pos[1])
        elif has_pos:
            desired_y = float(self._profile_vec3(view, pos_keys, [0.0, 0.0, 0.0])[1])
        else:
            return
        current_space = getattr(self, '_xr_space_pose_in_ref', np.eye(4, dtype=np.float32))
        native_head_y = float((current_space @ raw_head)[1, 3])
        dy = native_head_y - desired_y
        new_space_in_ref = current_space.copy()
        new_space_in_ref[1, 3] = dy
        try:
            new_space = xr.create_reference_space(
                self._xr_session,
                xr.ReferenceSpaceCreateInfo(
                    reference_space_type=self._xr_ref_space_type,
                    pose_in_reference_space=_mat4_to_xr_posef(new_space_in_ref),
                ),
            )
        except Exception as exc:
            print(f"[OpenXRViewer] Vertical reseat failed: {exc}")
            return
        old_space = self._xr_space
        self._xr_space = new_space
        self._xr_space_pose_in_ref = new_space_in_ref
        if old_space is not None:
            try:
                xr.destroy_space(old_space)
            except Exception:
                pass
        print(f"[OpenXRViewer] Vertical reseat: desired_y={desired_y:.3f} native_y={native_head_y:.3f}")

    # ------------------------------------------------------------------
    # Seat adjust mode  (both grips toggle, thumbsticks move, save on exit)
    # ------------------------------------------------------------------

    def _seat_adjust_current_pos(self):
        """Return current (x, y, z, angle) in screen-relative coords.

        Convention:  x = horizontal offset from screen centre (right +)
                     y = distance from screen (positive = further away)
                     z = vertical offset from screen centre (up +)
                     angle = horizontal rotation in degrees (0 = face screen,
                             negative = clockwise, positive = counter-clockwise)
        """
        view = getattr(self, '_view_pose_profile', {}) or {}
        x = float(view.get('x', 0.0))
        y = float(view.get('y', 0.6))
        z = float(view.get('z', 0.0))
        angle = float(view.get('angle', 0.0))
        return x, y, z, angle

    def _enter_seat_adjust_mode(self):
        if self._seat_adjust_active:
            return
        self._seat_adjust_active = True
        self._seat_adjust_t = time.perf_counter()
        self._seat_adjust_osd_show_t = time.perf_counter()
        self._seat_adjust_osd_dirty = True
        print("[OpenXRViewer] Entered seat adjust mode")

    def _exit_seat_adjust_mode(self, save=True):
        if not self._seat_adjust_active:
            return
        self._seat_adjust_active = False
        if save:
            self._save_view_pose_to_profile()
        self._seat_adjust_osd_dirty = True
        print("[OpenXRViewer] Exited seat adjust mode (saved=%s)" % save)

    def _apply_seat_adjust_xr_space(self, x, y, z, angle):
        """Recompute XR space offset from screen-relative (x, y, z, angle)."""
        screen = getattr(self, '_screen_profile', {}) or {}
        if not isinstance(screen, dict) or not screen:
            return
        position = screen.get('position', screen.get('screen_position'))
        if not isinstance(position, (list, tuple)) or len(position) < 3:
            return
        screen_pos = np.array([float(position[0]), float(position[1]), float(position[2])],
                              dtype=np.float32)
        screen_rot = self._profile_rotation_rad(
            screen,
            ('rotation_deg', 'screen_rotation_deg'),
            ('rotation', 'screen_rotation'),
            [0.0, 0.0, 0.0],
        )
        screen_mat = _euler_to_mat4(*screen_rot).astype(np.float32)
        screen_normal = screen_mat[:3, 2].copy()
        screen_right = screen_mat[:3, 0].copy()
        screen_up = screen_mat[:3, 1].copy()
        viewer_pos = screen_pos + screen_right * x + screen_normal * y + screen_up * z
        angle_rad = math.radians(angle)
        face_screen_yaw = math.atan2(float(screen_normal[0]), float(screen_normal[2]))
        viewer_yaw = face_screen_yaw + angle_rad
        desired_head = _euler_to_mat4(viewer_yaw, 0.0, 0.0).astype(np.float32)
        desired_head[:3, 3] = viewer_pos
        views = getattr(self, '_last_located_views', None)
        if not views or views[0] is None or views[1] is None:
            return
        raw_head = self._head_model_mat4_from_views(views)
        if raw_head is None:
            return
        current_space_in_ref = getattr(self, '_xr_space_pose_in_ref', np.eye(4, dtype=np.float32))
        reference_head = current_space_in_ref @ raw_head
        leveled = self._level_head_model_mat4(reference_head)
        if leveled is not None:
            reference_head = leveled
        space_in_ref = reference_head @ np.linalg.inv(desired_head)
        try:
            new_space = xr.create_reference_space(
                self._xr_session,
                xr.ReferenceSpaceCreateInfo(
                    reference_space_type=self._xr_ref_space_type,
                    pose_in_reference_space=_mat4_to_xr_posef(space_in_ref.astype(np.float32)),
                ),
            )
        except Exception as exc:
            print(f"[OpenXRViewer] Seat adjust XR space failed: {exc}")
            return
        old_space = self._xr_space
        self._xr_space = new_space
        self._xr_space_pose_in_ref = space_in_ref.astype(np.float32)
        if old_space is not None:
            try:
                xr.destroy_space(old_space)
            except Exception:
                pass

    def _save_view_pose_to_profile(self):
        """Write current screen-relative view pose back to profile.json."""
        view = getattr(self, '_view_pose_profile', {}) or {}
        if not isinstance(view, dict):
            return
        env_name = self._active_environment or self._environment_model
        if not env_name or env_name.lower() in ('default', 'default glow'):
            return
        root = self._environment_root
        room_dir = os.path.join(root, env_name)
        profile_path = os.path.join(room_dir, 'profile.json')
        if not os.path.exists(profile_path):
            return
        try:
            with open(profile_path, 'r', encoding='utf-8-sig') as f:
                profile = json.load(f)
        except Exception as exc:
            print(f"[OpenXRViewer] Failed to read profile for save: {exc}")
            return
        vp = None
        view_poses = profile.get('view_poses')
        if isinstance(view_poses, list) and view_poses:
            idx = int(getattr(self, '_view_pose_index', 0)) % len(view_poses)
            if not isinstance(view_poses[idx], dict):
                view_poses[idx] = {}
            vp = view_poses[idx]
            profile['view_pose_index'] = idx
        if vp is None:
            if 'view_pose' not in profile:
                profile['view_pose'] = {}
            vp = profile['view_pose']
        vp['x'] = round(float(view.get('x', 0.0)), 4)
        vp['y'] = round(float(view.get('y', 0.6)), 4)
        vp['z'] = round(float(view.get('z', 0.0)), 4)
        vp['angle'] = round(float(view.get('angle', 0.0)), 1)
        profile['model_position'] = [round(float(v), 4) for v in self._env_model_pos]
        profile['screen_light_intensity'] = round(float(self._screen_light_intensity), 2)
        if 'screen' in profile and isinstance(profile['screen'], dict):
            profile['screen']['curve_axis'] = self._screen_curve_mode()
        try:
            with open(profile_path, 'w', encoding='utf-8') as f:
                json.dump(profile, f, indent=2, ensure_ascii=False)
            print(f"[OpenXRViewer] Saved view_pose to {profile_path}: x={vp['x']} y={vp['y']} z={vp['z']} angle={vp['angle']} curve={self._screen_curve_mode()}")
        except Exception as exc:
            print(f"[OpenXRViewer] Failed to save profile: {exc}")

    def _reset_screen_to_default(self, show_border=False):
        """Reset screen to upright default: 2 m ahead, perpendicular to floor.

        Screen is always vertical (pitch=0) and faces the user's current gaze
        direction. Centre height matches the headset eye height recorded at
        session start so the screen sits comfortably in front of the user.
        Called at session start and by the Y button.
        """
        if self._apply_profile_screen_layout(show_border=show_border):
            return
        old_screen_mat = self._screen_pose_mat4() if self._environment_screen_locked() else None
        RESET_DIST = 2.0
        DEFAULT_PRESET = 3
        self._reset_orientation_offsets()
        self._clear_screen_grab_anchors()
        # Cancel any in-flight glide animation so it does not fight the reset.
        self._anim_target_pan_x    = None
        self._anim_target_pan_y    = None
        self._anim_target_distance = None
        self._anim_target_yaw      = None
        self._anim_target_pitch    = None
        self._anim_target_roll     = None
        if not self._environment_screen_locked():
            self.screen_width    = 2.4
            self._screen_ref_size = 2.4
            self.screen_height   = None
            self._set_screen_curve_mode('none')
            self._preset_index   = DEFAULT_PRESET
        self.screen_pitch    = 0.0   # always vertical perpendicular to floor
        self.screen_roll     = 0.0   # always level no tilt
        if self._head_pos_w is not None and self._head_fwd_w is not None:
            hx, hy, hz = self._head_pos_w
            fx, fy, fz = self._head_fwd_w
            # Normalize full 3D forward vector so screen distance follows the
            # user's actual gaze direction, matching _apply_preset behavior.
            flen = math.sqrt(fx * fx + fy * fy + fz * fz)
            if flen > 1e-4:
                fx /= flen; fy /= flen; fz /= flen
            else:
                fx, fy, fz = 0.0, 0.0, -1.0
            # Place screen centre RESET_DIST metres in front of the head along
            # the gaze direction.  The model matrix is T @ R @ S (translate then
            # rotate), so the world-space centre is simply (pan_x, pan_y, -distance).
            if self._environment_screen_locked():
                dx = self.screen_pan_x - hx
                dy = self.screen_pan_y - hy
                dz = -self.screen_distance - hz
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                tx = hx + fx * dist
                tz = hz + fz * dist
                self.screen_distance = -tz
                self.screen_pan_x    = tx
            else:
                self.screen_pan_x    = hx + fx * RESET_DIST
                self.screen_distance = -(hz + fz * RESET_DIST)
                self.screen_pan_y    = float(self._initial_head_y)
            # Compute yaw from the horizontal projection so the screen faces the
            # user horizontally while staying vertical (pitch=0).
            horiz = math.sqrt(fx * fx + fz * fz)
            self.screen_yaw = math.atan2(-fx, -fz) if horiz > 1e-4 else 0.0
        else:
            if not self._environment_screen_locked():
                self.screen_distance = RESET_DIST
                self.screen_pan_x    = 0.0
                self.screen_pan_y    = float(self._initial_head_y)
            self.screen_yaw      = 0.0
        self._move_env_with_screen_delta(old_screen_mat)
        if show_border:
            self._border_alpha  = 1.0
            self._border_idle_t = time.perf_counter()
        if self._keyboard_visible:
            self._anchor_keyboard_below_screen()

    def _screen_uv_to_world(self, u, v):
        """Convert a screen UV in [0,1] to its world-space 3-D position on the screen plane."""
        sh = self.screen_height
        if sh is None:
            fw, fh = self.frame_size
            sh = self.screen_width * (self._effective_frame_aspect() if fw > 0 else 9.0 / 16.0)

        # Curved-screen mapping: parametric cylindrical arc. Horizontal curves
        # bend across U; vertical curves bend across V.
        if self._screen_curved:
            half_w = float(self.screen_width) / 2.0
            half_h = float(sh) / 2.0
            half_ang = min(_CURVED_HALF_ANGLE_RAD, math.pi / 2)
            if self._screen_curve_mode() == 'vertical':
                radius = half_h / max(half_ang, 1e-6)
                ang = -half_ang + 2.0 * half_ang * float(v)
                local = np.array([
                    (float(u) - 0.5) * self.screen_width,
                    radius * math.sin(ang),
                    radius * (1.0 - math.cos(ang)),
                ], dtype='f8')
            else:
                radius = half_w / max(half_ang, 1e-6)
                ang = -half_ang + 2.0 * half_ang * float(u)
                local = np.array([
                    radius * math.sin(ang),
                    (float(v) - 0.5) * sh,
                    radius * (1.0 - math.cos(ang)),
                ], dtype='f8')
            Rm, center = self._screen_effect_basis()
            return center.astype('f8') + Rm[:3, :3].astype('f8') @ local

        # Flat screen mapping (original behaviour)
        cp  = math.cos(self.screen_pitch); sp  = math.sin(self.screen_pitch)
        cy  = math.cos(self.screen_yaw);   sy_ = math.sin(self.screen_yaw)
        screen_pos = self._screen_world_pos(cy, sy_, cp, sp)
        r_ax = np.array([cy,      0.0,    -sy_    ], dtype='f8')
        u_ax = np.array([sp*sy_,  cp,      sp*cy  ], dtype='f8')
        return screen_pos + r_ax * ((u - 0.5) * self.screen_width) + u_ax * ((v - 0.5) * sh)

    def _screen_world_pos(self, cy, sy_, cp, sp):
        """Return the world-space position of the screen centre based on current pan/pitch/yaw.
        Used by both _screen_uv_to_world and _laser_screen_hit_dist, which are called frequently during aiming so we take pan/pitch/yaw as arguments to avoid redundant trig calls.
        """
        return np.array([self.screen_pan_x, self.screen_pan_y, -self.screen_distance], dtype='f8')

    def _get_target_monitor_rect(self):
        """Return (left, top, width, height) of the captured monitor in virtual-desktop pixels.

        Uses Win32 EnumDisplayMonitors.  Monitor indices follow the MSS 1-based convention
        used in utils.py / viewer.py.  Falls back to the primary monitor.
        Result is cached in self._target_mon_rect.
        """
        if self._target_mon_rect is not None:
            return self._target_mon_rect
        if sys.platform != 'win32':
            self._target_mon_rect = (0, 0, 1920, 1080)
            return self._target_mon_rect

        import ctypes
        from ctypes import wintypes

        class RECT(ctypes.Structure):
            _fields_ = [('left', wintypes.LONG), ('top', wintypes.LONG),
                        ('right', wintypes.LONG), ('bottom', wintypes.LONG)]

        class MONITORINFO(ctypes.Structure):
            _fields_ = [('cbSize', wintypes.DWORD), ('rcMonitor', RECT),
                        ('rcWork', RECT), ('dwFlags', wintypes.DWORD)]

        monitors = []
        def _callback(h_mon, hdc, p_rect, _):
            mi = MONITORINFO()
            mi.cbSize = ctypes.sizeof(MONITORINFO)
            ctypes.windll.user32.GetMonitorInfoW(h_mon, ctypes.byref(mi))
            r = mi.rcMonitor
            monitors.append((r.left, r.top, r.right - r.left, r.bottom - r.top))
            return True

        _CBFUNC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HMONITOR, wintypes.HDC,
                                    ctypes.POINTER(RECT), wintypes.LPARAM)
        ctypes.windll.user32.EnumDisplayMonitors(None, None, _CBFUNC(_callback), 0)

        idx = max(1, self._input_monitor_index) - 1  # MSS is 1-based 0-based
        if idx < len(monitors):
            self._target_mon_rect = monitors[idx]
        else:
            self._target_mon_rect = monitors[0] if monitors else (0, 0, 1920, 1080)
        return self._target_mon_rect

    def _laser_screen_hit_dist(self, ctrl_pos, fwd_w):
        """Return the distance along fwd_w where the aim ray hits the screen.

        Delegates to _laser_screen_hit_uv to avoid duplicating ray-plane math.
        Returns BEAM_MAX (30 m) on miss.
        """
        result = self._laser_screen_hit_uv(ctrl_pos, fwd_w)
        if result is not None:
            return max(0.01, result[2] - 0.005)
        return 30.0

    def _keyboard_laser_hit_dist(self, ctrl_pos, fwd_w):
        """Return the distance along fwd_w where the ray hits the keyboard quad.

        Returns BEAM_MAX if the keyboard is hidden, the ray is parallel to it,
        or the hit point falls outside the keyboard rectangle.
        """
        BEAM_MAX = 30.0
        if not self._keyboard_visible or not self._keyboard_keys:
            return BEAM_MAX
        cp = math.cos(self._keyboard_pitch); sp = math.sin(self._keyboard_pitch)
        cy = math.cos(self._keyboard_yaw);   sy = math.sin(self._keyboard_yaw)
        kb_x = np.array([ cy,      0.0,  -sy      ], dtype='f8')
        kb_y = np.array([ sy * sp,  cp,   cy * sp ], dtype='f8')
        kb_n = np.array([ sy * cp, -sp,   cy * cp ], dtype='f8')
        kb_pos = np.array([self._keyboard_pan_x,
                        self._keyboard_pan_y,
                        -self._keyboard_distance], dtype='f8')
        denom = float(np.dot(kb_n, fwd_w))
        if abs(denom) < 1e-6:
            return BEAM_MAX
        t = float(np.dot(kb_n, kb_pos - ctrl_pos)) / denom
        if t < 0.01:
            return BEAM_MAX
        hit  = ctrl_pos + fwd_w * t
        diff = hit - kb_pos
        lx = float(np.dot(diff, kb_x))
        ly = float(np.dot(diff, kb_y))
        if abs(lx) <= self._keyboard_width / 2.0 and abs(ly) <= self._keyboard_height / 2.0:
            return max(0.01, t - 0.005)
        return BEAM_MAX

    def _overlay_panel_hit(self, cp, fw):
        """
        Returns True if the controller laser hits the FPS/status panel.
        """
        if self.screen_height is None:
            return False

        # Panel placement must match _render_fps_overlay() and _overlay_panel_hit_dist()
        sh = self.screen_height
        sx = self.screen_width / 2.0
        sy = sh / 2.0
        GAP = sh * 0.02
        ow, oh = self._overlay_tex_size
        OVERLAY_H = sh / 8.0
        OVERLAY_W = OVERLAY_H * (ow / oh)

        local_cx = -sx + OVERLAY_W / 2.0
        local_cy = -sy - GAP - OVERLAY_H / 2.0

        # Build screen transform
        cy = math.cos(self.screen_yaw)
        sy = math.sin(self.screen_yaw)
        cpitch = math.cos(self.screen_pitch)
        spitch = math.sin(self.screen_pitch)

        R = (
            np.array([
                [ cy, 0, sy],
                [  0, 1,  0],
                [-sy, 0, cy],
            ], dtype='f8')
            @
            np.array([
                [1,      0,       0],
                [0, cpitch, -spitch],
                [0, spitch,  cpitch],
            ], dtype='f8')
        )

        panel_center = np.array([
            self.screen_pan_x,
            self.screen_pan_y,
            -self.screen_distance
        ], dtype='f8')

        panel_center += R @ np.array([local_cx, local_cy, 0.0], dtype='f8')

        normal = R @ np.array([0.0, 0.0, 1.0], dtype='f8')

        denom = np.dot(fw, normal)
        if abs(denom) < 1e-5:
            return False

        t = np.dot(panel_center - cp, normal) / denom
        if t <= 0:
            return False

        hit = cp + fw * t
        local = hit - panel_center

        right = R @ np.array([1.0, 0.0, 0.0], dtype='f8')
        up    = R @ np.array([0.0, 1.0, 0.0], dtype='f8')

        lx = np.dot(local, right)
        ly = np.dot(local, up)

        return (
            abs(lx) <= OVERLAY_W * 0.5 and
            abs(ly) <= OVERLAY_H * 0.5
        )

    def _overlay_panel_hit_dist(self, ctrl_pos, fwd_w):
        """Return the distance along fwd_w where the ray hits the FPS overlay panel.

        The panel sits just below the screen, shares the same yaw/pitch rotation,
        and has the same surface normal. Returns BEAM_MAX if the overlay is hidden,
        screen_height is unknown, the ray is parallel, or the hit misses the rect.
        Math matches _render_fps_overlay exactly.
        """
        BEAM_MAX = 30.0
        if not self._fps_overlay_visible or self.screen_height is None:
            return BEAM_MAX

        sh = self.screen_height
        sx = self.screen_width / 2.0
        sy = sh / 2.0
        GAP = sh * 0.02
        ow, oh = self._overlay_tex_size
        OVERLAY_H = sh / 8.0
        OVERLAY_W = OVERLAY_H * (ow / oh)

        # Panel local-space centre matches _render_fps_overlay T_local
        local_cx = -sx + OVERLAY_W / 2.0
        local_cy = -sy - GAP - OVERLAY_H / 2.0

        cp  = math.cos(self.screen_pitch); sp  = math.sin(self.screen_pitch)
        cy  = math.cos(self.screen_yaw);   sy_ = math.sin(self.screen_yaw)

        # Panel normal is identical to the screen normal
        panel_n = np.array([cp * sy_, -sp, cp * cy], dtype='f8')

        # Panel right/up axes in world space
        r_ax = np.array([cy,      0.0,    -sy_    ], dtype='f8')
        u_ax = np.array([sp*sy_,  cp,      sp*cy  ], dtype='f8')

        # Panel world position: T @ R @ [local_cx, local_cy, 0, 1]
        # R_pitch @ [local_cx, local_cy, 0] = [local_cx, local_cy*cp, local_cy*sp]
        py = local_cy * cp
        pz = local_cy * sp
        # R_yaw @ [local_cx, py, pz] + screen translation
        wx = self.screen_pan_x + local_cx * cy + pz * sy_
        wy = self.screen_pan_y + py
        wz = -self.screen_distance - local_cx * sy_ + pz * cy
        panel_pos = np.array([wx, wy, wz], dtype='f8')

        denom = float(np.dot(panel_n, fwd_w))
        if abs(denom) < 1e-6:
            return BEAM_MAX
        t = float(np.dot(panel_n, panel_pos - ctrl_pos)) / denom
        if t < 0.01:
            return BEAM_MAX

        hit  = ctrl_pos + fwd_w * t
        diff = hit - panel_pos
        loc_x = float(np.dot(diff, r_ax))
        loc_y = float(np.dot(diff, u_ax))
        if abs(loc_x) <= OVERLAY_W / 2.0 and abs(loc_y) <= OVERLAY_H / 2.0:
            return max(0.01, t - 0.005)
        return BEAM_MAX

    # Controller pose smoothing for stable cursor/laser aiming
    @staticmethod
    def _mat3_to_quat(m33):
        """ 3x3 rotation matrix to (x,y,z,w) quaternion. """
        t  = m33[0, 0] + m33[1, 1] + m33[2, 2]
        if t > 0.0:
            s  = np.sqrt(t + 1.0) * 2.0
            w  = 0.25 * s
            x  = (m33[2, 1] - m33[1, 2]) / s
            y  = (m33[0, 2] - m33[2, 0]) / s
            z  = (m33[1, 0] - m33[0, 1]) / s
        elif m33[0, 0] > m33[1, 1] and m33[0, 0] > m33[2, 2]:
            s  = np.sqrt(1.0 + m33[0, 0] - m33[1, 1] - m33[2, 2]) * 2.0
            w  = (m33[2, 1] - m33[1, 2]) / s
            x  = 0.25 * s
            y  = (m33[0, 1] + m33[1, 0]) / s
            z  = (m33[0, 2] + m33[2, 0]) / s
        elif m33[1, 1] > m33[2, 2]:
            s  = np.sqrt(1.0 + m33[1, 1] - m33[0, 0] - m33[2, 2]) * 2.0
            w  = (m33[0, 2] - m33[2, 0]) / s
            x  = (m33[0, 1] + m33[1, 0]) / s
            y  = 0.25 * s
            z  = (m33[1, 2] + m33[2, 1]) / s
        else:
            s  = np.sqrt(1.0 + m33[2, 2] - m33[0, 0] - m33[1, 1]) * 2.0
            w  = (m33[1, 0] - m33[0, 1]) / s
            x  = (m33[0, 2] + m33[2, 0]) / s
            y  = (m33[1, 2] + m33[2, 1]) / s
            z  = 0.25 * s
        q = np.array([x, y, z, w], dtype='f8')
        return q / np.linalg.norm(q)

    def _slerp_quat(self, q1, q2, t):
        """Spherical linear interpolation: t=0 -> q1, t=1 -> q2. Input/output as (x,y,z,w) numpy arrays."""
        dot = np.dot(q1, q2)
        if dot < 0.0:
            q2 = -q2
            dot = -dot
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)
        theta_0 = math.acos(min(dot, 1.0))
        theta   = theta_0 * t
        sin_t   = math.sin(theta)
        sin_t0  = math.sin(theta_0)
        s1 = math.cos(theta) - dot * sin_t / sin_t0
        s2 = sin_t / sin_t0
        return s1 * q1 + s2 * q2

    def _smooth_controller_poses(self):
        """Pre-smooth both controller poses once per frame.

        Must be called after _update_aim_poses / _update_grip_poses and
        before any consumer (grip-to-move, cursor, laser rendering).
        Stores smoothed position + forward in _smooth_ray_* attrs.
        """
        for is_left, aim_mat_attr, grip_mat_attr, pos_attr, quat_attr in [
            (True,  '_aim_mat_l', '_grip_mat_l',
             '_smooth_ray_origin_l', '_smooth_ray_quat_l'),
            (False, '_aim_mat_r', '_grip_mat_r',
             '_smooth_ray_origin_r', '_smooth_ray_quat_r'),
        ]:
            aim_mat = getattr(self, aim_mat_attr)
            if aim_mat is None:
                continue
            grip_mat = getattr(self, grip_mat_attr)
            if grip_mat is not None:
                raw_pos = (grip_mat[:3, 3] + grip_mat[:3, 1] * 0.020).astype('f8')
            else:
                raw_pos = aim_mat[:3, 3].astype('f8')
            self._apply_ray_smoothing(raw_pos, aim_mat, pos_attr, quat_attr)
            # Pre-compute forward vector from smoothed quaternion (done once
            # per frame, shared by all consumers avoids 12x 3x3 rebuilds).
            sm_quat = getattr(self, quat_attr)
            if sm_quat is not None:
                x, y, z, w = sm_quat
                fwd = np.array([
                    -(2*x*z + 2*w*y),
                    -(2*y*z - 2*w*x),
                    -(1 - 2*x*x - 2*y*y),
                ], dtype='f8')
                if is_left:
                    self._smooth_ray_fwd_l = fwd
                else:
                    self._smooth_ray_fwd_r = fwd

    def _get_smoothed_ray(self, is_left):
        """Return (smoothed_pos, smoothed_fwd) from pre-computed attrs."""
        pos_attr = '_smooth_ray_origin_l' if is_left else '_smooth_ray_origin_r'
        fwd_attr = '_smooth_ray_fwd_l' if is_left else '_smooth_ray_fwd_r'
        sm_pos = getattr(self, pos_attr)
        sm_fwd = getattr(self, fwd_attr)
        if sm_pos is None or sm_fwd is None:
            return None, None
        return sm_pos.copy(), sm_fwd.copy()

    def _pre_snap_overlay_ray(self, is_left, aim_mat, grip_mat):
        """Return (cp, fw) for overlay hit testing: smoothed direction + 12° tilt, no edge-snap.

        The regular laser direction snaps toward the screen edge when the ray misses the screen.
        The overlay panel sits below the screen, so the snapped direction points away from it.
        This returns the unsnapped tilted direction so overlay detection works correctly.
        """
        sm_pos, sm_fw = self._get_smoothed_ray(is_left)
        if sm_pos is None:
            raw_pos = (grip_mat[:3, 3] + grip_mat[:3, 1] * 0.020).astype('f8') if grip_mat is not None else aim_mat[:3, 3].astype('f8')
            fw = -aim_mat[:3, 2].astype('f8')
            return raw_pos + fw * 0.11, fw
        right = aim_mat[:3, 0].astype('f8')
        ang = math.radians(12); ca, sa = math.cos(ang), math.sin(ang)
        k = right / (np.linalg.norm(right) + 1e-10)
        fw = sm_fw * ca + np.cross(k, sm_fw) * sa + k * np.dot(k, sm_fw) * (1 - ca)
        raw_pos = (grip_mat[:3, 3] + grip_mat[:3, 1] * 0.020).astype('f8') if grip_mat is not None else aim_mat[:3, 3].astype('f8')
        return raw_pos + fw * 0.11, fw

    def _apply_ray_smoothing(self, raw_pos, aim_mat, smooth_pos_attr, smooth_quat_attr):
        """Position EMA + quaternion SLERP smoothing (with dead zone). Returns (smoothed_pos, smoothed_fwd_world)."""
        raw_quat = self._mat3_to_quat(aim_mat[:3, :3].astype('f8'))

        prev_pos  = getattr(self, smooth_pos_attr)
        prev_quat = getattr(self, smooth_quat_attr)

        # One Euro Filter position smoothing (replaces adaptive EMA)
        # Filters raw controller position at source laser beam, cursor,
        # and grip-to-move all consume the same stabilized data.
        # No dead zone needed: adaptive cutoff naturally filters micro-jitter
        # while responding quickly to intentional movement.
        _filter = self._ray_filter_l if smooth_pos_attr.endswith('_l') else self._ray_filter_r
        if prev_pos is None:
            _filter.reset()
        sm_pos = _filter.filter(raw_pos, self._last_frame_dt)
        setattr(self, smooth_pos_attr, sm_pos.copy())

        # Adaptive SLERP angle smoothing with dead zone
        if prev_quat is not None:
            _dot = abs(np.dot(raw_quat, prev_quat))
            _dot = min(_dot, 1.0)
            _ang = 2.0 * math.acos(_dot) if _dot < 1.0 else 0.0
            # Normal Deadzone + SLERP (Not affect the movement on screen)
            if _ang < self._ray_deadzone_rad:
                sm_quat = prev_quat  # Not changed in Deadzone
            else:
                # The greater the angular velocity, the larger the smoothing factor, resulting in tighter tracking. When hand motion stops, the factor decreases, leading to a smooth, gradual deceleration.
                _adaptive = self._rot_smooth * (1.0 + min(_ang * 30.0, 2.0))
                _adaptive = min(_adaptive, 0.30)
                sm_quat = self._slerp_quat(prev_quat, raw_quat, _adaptive)
        else:
            sm_quat = raw_quat
        setattr(self, smooth_quat_attr, sm_quat.copy())
        x, y, z, w = sm_quat
        R33 = np.array([
            [1-2*y*y-2*z*z,   2*x*y-2*w*z,   2*x*z+2*w*y],
            [  2*x*y+2*w*z, 1-2*x*x-2*z*z,   2*y*z-2*w*x],
            [  2*x*z-2*w*y,   2*y*z+2*w*x, 1-2*x*x-2*y*y],
        ], dtype='f8')
        fwd_w = -R33[:, 2]  # -Z = forward
        return sm_pos, fwd_w

    def _laser_beam_setup(self):
        """Ray sharing: Quaternion SLERP (direction) + Position EMA simulates VD damping."""
        now = self._frame_now
        beams = []
        for aim_mat, grip_mat, last_move_attr, ctrl_name, \
            smooth_pos_attr, smooth_quat_attr in [
            (self._aim_mat_l, self._grip_mat_l, "_laser_last_move_l", 'left',
            "_smooth_ray_origin_l", "_smooth_ray_quat_l"),
            (self._aim_mat_r, self._grip_mat_r, "_laser_last_move_r", 'right',
            "_smooth_ray_origin_r", "_smooth_ray_quat_r"),
        ]:
            if aim_mat is None:
                continue
            if (now - getattr(self, last_move_attr)) > self._LASER_HIDE_AFTER:
                setattr(self, smooth_pos_attr, None)
                setattr(self, smooth_quat_attr, None)
                continue

            # Use pre-smoothed pose (One Euro Filter applied once per frame)
            is_left = (ctrl_name == 'left')
            ctrl_pos, fwd_w = self._get_smoothed_ray(is_left)
            if ctrl_pos is None:
                continue

            # Raw origin (for edge constraint uses unsmoothed position)
            if grip_mat is not None:
                raw_pos = (grip_mat[:3, 3] + grip_mat[:3, 1] * 0.020).astype('f8')
            else:
                raw_pos = aim_mat[:3, 3].astype('f8')

            # Laser direction: apply a fixed 12° upward tilt to the smoothed forward vector, around the right axis (parallel to the screen when facing it head-on) mimics the natural resting pose of the hand and makes it easier to point at things without needing to tilt the wrist up.
            right_w = aim_mat[:3, 0].astype('f8')
            _ang = math.radians(12); _ca, _sa = math.cos(_ang), math.sin(_ang)
            _k = right_w / (np.linalg.norm(right_w) + 1e-10)
            fwd_w = fwd_w * _ca + np.cross(_k, fwd_w) * _sa + _k * np.dot(_k, fwd_w) * (1 - _ca)

            # Keyboard targeting takes precedence over screen edge-snapping.
            # When the keyboard sits just below the screen, aiming at its top rows
            # places the ray inside the screen's bottom-edge dead zone, so the
            # edge-snap below would deflect the beam (and thus the hit circle)
            # toward the screen edge away from the real keyboard intersection
            # the user is pointing at (and where the key press is registered).
            # Detect a keyboard hit on the true tilted ray and, if present, skip
            # edge-snapping so the visible cursor matches the keyboard hit exactly.
            _kb_targeted = (self._keyboard_visible and
                            self._keyboard_laser_hit_dist(raw_pos, fwd_w) < 30.0)

            # Screen edge constraint: if the smoothed ray goes off-screen, try the raw ray (with the same 12° tilt) before giving up and snapping to the edge.  This lets the laser stay more stable near edges while still allowing the user to intentionally point off-screen by moving steadily in that direction.
            if not _kb_targeted and self._laser_screen_hit_uv(raw_pos, fwd_w) is None:
                # Smoothed ray misses screen check raw ray (same tilt)
                _raw_fwd = -aim_mat[:3, 2].astype('f8')
                _raw_rw = aim_mat[:3, 0].astype('f8')
                _raw_fwd = _raw_fwd * _ca + np.cross(_k, _raw_fwd) * _sa + _k * np.dot(_k, _raw_fwd) * (1 - _ca)
                if self._laser_screen_hit_uv(raw_pos, _raw_fwd) is None:
                    # Raw ray also goes off-screen snap to edge (ring follows manual movement along the edge)
                    _plane_uv = self._laser_plane_uv(raw_pos, fwd_w)
                    if _plane_uv is not None:
                        _cu = max(0.0, min(1.0, _plane_uv[0]))
                        _cv = max(0.0, min(1.0, _plane_uv[1]))
                        _clamped_wp = self._screen_uv_to_world(_cu, _cv)
                        _edge_dir = _clamped_wp - raw_pos
                        _norm = np.linalg.norm(_edge_dir)
                        if _norm > 1e-6:
                            _edge_dir /= _norm
                            _dot2 = np.dot(_raw_fwd, _edge_dir)
                            _dot2 = max(-1.0, min(1.0, _dot2))
                            _ang2 = math.acos(_dot2)
                            if _ang2 < self._ray_edge_deadzone_rad:
                                # Blend 40% of the edge direction, keep 60% of the raw direction
                                blend = (1.0 - EDGE_STRENGTH) * _raw_fwd + EDGE_STRENGTH * _edge_dir
                                fwd_w = blend / (np.linalg.norm(blend) + 1e-10)  # edge snapping
            if ctrl_name == 'left':
                self._smooth_ray_prev_fwd_l = fwd_w.copy()
            else:
                self._smooth_ray_prev_fwd_r = fwd_w.copy()

            # Beam starts from raw controller position (aligned with model)
            ctrl_pos = raw_pos + fwd_w * 0.11

            right = aim_mat[:3, 0].astype('f4')
            fwd = fwd_w.astype('f4')
            up = np.cross(right, fwd); up = up / (np.linalg.norm(up) + 1e-10)
            right2 = np.cross(fwd, up)
            beams.append((now, ctrl_name, aim_mat, ctrl_pos, fwd_w, right2, fwd, up))
        return beams

    def _render_lasers(self, mgl_fbo, vp_mat, view_mat, blend=False, view_inv=None):
        """blend=False: opaque rainbow beam; blend=True: semi-transparent hit circles."""
        beams = getattr(self, '_cached_beams', None)
        if not beams:
            return
        if blend:
            self._render_laser_hit_circles(mgl_fbo, vp_mat, view_mat, beams, view_inv)
            return
        mgl_fbo.use()
        BEAM_MAX_LEN = 0.4
        for now, ctrl_name, aim_mat, ctrl_pos, fwd_w, right2, fwd, up in beams:
            # Auto-shorten beam when target is closer than the default length 
            # mirrors the priority logic in _render_laser_hit_circles so the
            # beam tip lines up with the hit circle (no over/undershoot).
            cursor_uv = self._cursor_uv_l if ctrl_name == 'left' else self._cursor_uv_r
            if (self._cursor_ctrl == ctrl_name and cursor_uv is not None):
                hit_dist = max(0.01, float(cursor_uv[2]))
            else:
                kb_dist = self._keyboard_laser_hit_dist(ctrl_pos, fwd_w)
                sc_dist = self._laser_screen_hit_dist(ctrl_pos, fwd_w)
                # Use pre-snap direction for overlay hit (edge-snap deflects away from panel)
                _is_left = (ctrl_name == 'left')
                _bgrip = self._grip_mat_l if _is_left else self._grip_mat_r
                _bov_cp, _bov_fw = self._pre_snap_overlay_ray(_is_left, aim_mat, _bgrip)
                ov_dist = self._overlay_panel_hit_dist(_bov_cp, _bov_fw)
                if self._keyboard_visible and kb_dist < 5.0:
                    hit_dist = kb_dist
                else:
                    hit_dist = min(sc_dist, kb_dist, ov_dist)
            draw_len = min(BEAM_MAX_LEN, max(0.01, hit_dist))
            BEAM_R = 0.006
            M = np.zeros((4, 4), dtype='f4')
            M[:3, 0] = right2 * BEAM_R
            M[:3, 1] = fwd * draw_len
            M[:3, 2] = up * BEAM_R
            M[:3, 3] = ctrl_pos.astype('f4')
            M[3, 3] = 1.0
            beam_mvp = vp_mat @ M
            self._beam_prog['u_mvp'].write(beam_mvp.T.tobytes())
            self._beam_prog['u_time'].value = float(now)
            self._beam_vao.render(moderngl.TRIANGLE_STRIP)
    def _render_laser_hit_circles(self, mgl_fbo, vp_mat, view_mat, beams, view_inv=None):
        mgl_fbo.use()
        if view_inv is None:
            view_inv = _view_mat_inv(view_mat)
        cam_r = view_inv[:3, 0].astype('f4')
        cam_u = view_inv[:3, 1].astype('f4')
        cam_pos = view_inv[:3, 3].astype('f4')
        cam_r /= np.linalg.norm(cam_r) + 1e-10
        cam_u /= np.linalg.norm(cam_u) + 1e-10

        for now, ctrl_name, aim_mat, ctrl_pos, fwd_w, right2, fwd, up in beams:
            kb_dist = self._keyboard_laser_hit_dist(ctrl_pos, fwd_w)
            sc_dist = self._laser_screen_hit_dist(ctrl_pos, fwd_w)
            # Use pre-snap direction for overlay hit so edge-snapping doesn't
            # deflect the ray away from the panel (which sits below the screen).
            is_left = (ctrl_name == 'left')
            grip_mat = self._grip_mat_l if is_left else self._grip_mat_r
            _ov_cp, _ov_fw = self._pre_snap_overlay_ray(is_left, aim_mat, grip_mat)
            _sm_pos = self._get_smoothed_ray(is_left)[0]  # None when no smoothed data yet
            ov_dist = self._overlay_panel_hit_dist(_ov_cp, _ov_fw)

            beam_len = 30.0
            hit_ray_cp  = ctrl_pos   # origin for hit_pos computation (may be overridden by overlay)
            hit_ray_fwd = fwd_w
            # Match opaque-beam target priority so the hit ring doesn't jump from
            # keyboard to screen near the keyboard's top edge when both are close.
            if self._keyboard_visible and kb_dist < 5.0:
                beam_len = kb_dist
            else:
                beam_len = min(sc_dist, kb_dist, ov_dist)
                if ov_dist <= beam_len:
                    # Use pre-snap ray origin/direction for overlay hit position
                    if _sm_pos is not None:
                        hit_ray_cp  = _ov_cp
                        hit_ray_fwd = _ov_fw

            if beam_len >= 30.0:
                continue

            HIT_OFFSET = 0.0
            hit_pos = hit_ray_cp + hit_ray_fwd * (beam_len - HIT_OFFSET)

            to_cam = cam_pos - hit_pos.astype('f4')
            _eye_dist_sq = float(to_cam[0]*to_cam[0] + to_cam[1]*to_cam[1] + to_cam[2]*to_cam[2])
            _eye_dist = math.sqrt(_eye_dist_sq) if _eye_dist_sq > 0 else 0.0
            to_cam_dir = to_cam / (_eye_dist + 1e-10)

            _scale = math.sqrt(max(_eye_dist, 0.2))
            STROKE_R = 0.0096 * _scale
            FILL_R   = 0.0064 * _scale
            M = np.zeros((4, 4), dtype='f4')
            M[3, 3] = 1.0
            for radius, color, z_bias in [
                (STROKE_R, (0.2, 0.6, 1.0, 0.75), 0.0),
                (FILL_R,   (1.0, 1.0, 1.0, 0.75), 0.0001),
            ]:
                M[:3, 0] = cam_r * radius
                M[:3, 1] = cam_u * radius
                M[:3, 3] = (hit_pos + to_cam_dir * z_bias).astype('f4')
                circle_mvp = vp_mat @ M
                self._border_prog['u_mvp'].write(circle_mvp.T.tobytes())
                self._border_prog['u_color'].value = color
                self._circle_vao.render(moderngl.TRIANGLE_FAN)

    def _render_controllers(self, mgl_fbo, vp_mat, view_mat, view_inv=None):
        """Render PICO 4 Ultra 3D controller models with Blinn-Phong lighting."""
        now = self._frame_now
        if view_inv is None:
            view_inv = _view_mat_inv(view_mat)
        cam_pos = view_inv[:3, 3].astype(np.float32)
        controllers = []
        for grip_mat, prims, last_move_attr in [
            (self._grip_mat_l, self._ctrl_prims_l, "_laser_last_move_l"),
            (self._grip_mat_r, self._ctrl_prims_r, "_laser_last_move_r"),
        ]:
            if (now - getattr(self, last_move_attr)) > self._LASER_HIDE_AFTER:
                continue
            if grip_mat is None or not prims:
                continue
            diff = grip_mat[:3, 3] - cam_pos
            dist = float(math.sqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]))
            controllers.append((dist, grip_mat, prims))

        if not controllers:
            return

        controllers.sort(key=lambda x: x[0], reverse=True)
        mgl_fbo.use()

        # light_color (diffuse) is slightly bluish to look more like the PICO 4's built-in light;
        # ambient_color is dim to avoid washing out the dark controller textures.
        light_color = np.array([0.60, 0.60, 0.65], dtype=np.float32)
        ambient_color = np.array([0.15, 0.15, 0.17], dtype=np.float32)

        _off = (self._calibration_temp_offset if self._calibration_mode
                else self._ctrl_model_offset)
        _rot = (self._calibration_temp_rot if self._calibration_mode
                else self._ctrl_model_rot_deg)
        _corr_key = (tuple(_off), _rot)
        if getattr(self, '_ctrl_corr_key', None) != _corr_key:
            T_mat = np.eye(4, dtype=np.float32)
            T_mat[0, 3] = _off[0]
            T_mat[1, 3] = _off[1]
            T_mat[2, 3] = _off[2]
            _ang = math.radians(_rot)
            _ca, _sa = math.cos(_ang), math.sin(_ang)
            R_mat = np.eye(4, dtype=np.float32)
            R_mat[1, 1] = _ca; R_mat[1, 2] = -_sa
            R_mat[2, 1] = _sa; R_mat[2, 2] = _ca
            self._ctrl_corr_mat = (R_mat @ T_mat).astype(np.float32)
            self._ctrl_corr_key = _corr_key
        _corr = self._ctrl_corr_mat

        vp_bytes = vp_mat.astype(np.float32).T.tobytes()
        light_bytes = light_color.tobytes()
        ambient_bytes = ambient_color.tobytes()
        cam_bytes = cam_pos.tobytes()

        for _dist, grip_mat, prims in controllers:
            model_mat = (grip_mat @ _corr).astype(np.float32)

            # Set common uniforms for the current controller
            # u_model: model world (grip @ _corr)
            # u_mvp: world clip (VP only, no model, because shader computes world_pos = u_model * v)
            self._controller_prog['u_mvp'].write(vp_bytes)
            self._controller_prog['u_model'].write(model_mat.T.tobytes())
            self._controller_prog['u_normal_mat'].write(model_mat[:3, :3].T.tobytes())
            self._controller_prog['u_light_color'].write(light_bytes)
            self._controller_prog['u_ambient_color'].write(ambient_bytes)
            self._controller_prog['u_camera_pos'].write(cam_bytes)

            sorted_prims = prims

            if self._use_d3d11:
                glFrontFace(GL_CW)

            for prim in sorted_prims:
                tex = self._ctrl_tex_cache.get(prim['tex_key'])
                if tex is not None:
                    tex.use(location=3)
                prim['vao'].render(moderngl.TRIANGLES)

            if self._use_d3d11:
                glFrontFace(GL_CCW)

    # Controller real-time calibration
    def _enter_calibration_mode(self):
        """Enter calibration mode: copy current offset to temporary values, disable regular controller operations."""
        self._calibration_mode = True
        self._calibration_temp_offset = list(self._ctrl_model_offset)
        self._calibration_temp_rot = self._ctrl_model_rot_deg
        _brand = self._current_brand or self._controller_model
        print(f"[Calibration] Entered mode. Brand: {_brand}")
        print(f"  L-stick U/D = Y, R-stick U/D = Z, R-stick L/R = Rotation")
        print(f"  B button = Save & exit, long Menu+A+B = Discard & exit")

    def _exit_calibration_mode(self, save=True):
        """Exit calibration mode. When save=True, write to profile.json."""
        if save:
            # Save to controllers/<brand>/profile.json
            import json as _json, os as _os
            _brand = self._current_brand or self._controller_model
            _base = _os.path.join(_MODULE_DIR, 'controllers', _brand)
            _os.makedirs(_base, exist_ok=True)
            _path = _os.path.join(_base, 'profile.json')
            # Read existing profile if exists, to preserve unrelated fields (e.g. model name, other overrides) and only update the offset/rotation overrides.
            _prof = {}
            if _os.path.isfile(_path):
                try:
                    with open(_path, 'r') as f:
                        _prof = _json.load(f)
                except Exception:
                    pass
            _prof['profileId'] = self._controller_model
            _prof['overrides'] = {
                'model_offset': self._calibration_temp_offset,
                'model_rotation_deg': self._calibration_temp_rot,
            }
            with open(_path, 'w') as f:
                _json.dump(_prof, f, indent=2, ensure_ascii=False)
            # Update the runtime values immediately so the user can see the effect without needing to re-enter calibration mode, and also update the _all_models cache so that if they later calibrate another model of the same brand, it will use the newly saved values as the starting point.
            self._ctrl_model_offset = list(self._calibration_temp_offset)
            self._ctrl_model_rot_deg = self._calibration_temp_rot
            # Sync the cache in case of multiple models of the same brand sharing the same profile overrides (common use case: same profile for both left/right controllers, or multiple models from the same brand that don't have separate profiles).
            if _brand in self._all_models:
                self._all_models[_brand]['offset'] = list(self._ctrl_model_offset)
                self._all_models[_brand]['rot_deg'] = self._ctrl_model_rot_deg
            print(f"[Calibration] Saved to {_path}: "
                f"offset={self._ctrl_model_offset}, rot={self._ctrl_model_rot_deg}")
        else:
            print("[Calibration] Discarded changes.")
        self._calibration_mode = False
        self._calib_combo_fired = False

    def _advance_glow_color(self, lerp=0.03):
        """Advance ``self._glow_color`` one step toward ``self._glow_target_color``.

        Factored out of ``_render_glow`` so it can also be called by the
        env-model path when the planar glow is gated off (so the env
        shader's cinema bias light, which reads ``self._glow_color``,
        keeps tracking the screen content).
        """
        c = self._glow_color
        t = self._glow_target_color
        self._glow_color = (
            c[0] + lerp * (t[0] - c[0]),
            c[1] + lerp * (t[1] - c[1]),
            c[2] + lerp * (t[2] - c[2]),
        )

    def _active_glow_mode(self):
        mode = str(getattr(self, '_glow_mode', '') or '').strip().lower()
        aliases = {
            'screen': 'glow',
            'surround': 'glow',
            'frost': 'frosted',
            'frosted': 'frosted',
            'veil': 'veil',
            'glow': 'glow',
            'off': 'off',
            'none': 'off',
        }
        if mode in aliases:
            return aliases[mode]
        mult = float(getattr(self, '_glow_intensity_multiplier', 0.0) or 0.0)
        return 'glow' if mult > 0.0 else 'off'

    def _refresh_active_glow_mode_cache(self):
        self._active_glow_mode_cached = self._active_glow_mode()
        return self._active_glow_mode_cached

    def _screen_effect_basis(self):
        cy = math.cos(self.screen_yaw)
        sy_ = math.sin(self.screen_yaw)
        cp = math.cos(self.screen_pitch)
        sp = math.sin(self.screen_pitch)
        cr = math.cos(self.screen_roll)
        sr = math.sin(self.screen_roll)
        R = np.array([
            [ cy * cr + sy_ * sp * sr, -cy * sr + sy_ * sp * cr,  sy_ * cp, 0],
            [ cp * sr,                  cp * cr,                 -sp,      0],
            [-sy_ * cr + cy * sp * sr,  sy_ * sr + cy * sp * cr,  cy * cp, 0],
            [ 0,                        0,                        0,       1],
        ], dtype=np.float32)
        center = np.array(
            [self.screen_pan_x, self.screen_pan_y, -self.screen_distance],
            dtype=np.float32,
        )
        return R, center

    def _screen_effect_model(self, width, height, z_offset=0.0, y_offset=0.0, z_scale=1.0):
        if self.screen_height is None:
            self._build_model_mat4()

        sx = width / 2.0
        sy = height / 2.0
        R, center = self._screen_effect_basis()
        S = np.diag([sx, sy, z_scale, 1.0]).astype(np.float32)
        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = center
        T[1, 3] += y_offset
        if z_offset:
            n = R[:3, 2]
            T[0, 3] += float(n[0]) * z_offset
            T[1, 3] += float(n[1]) * z_offset
            T[2, 3] += float(n[2]) * z_offset
        return T @ R @ S

    def _frost_front_layout(self):
        if self.screen_height is None:
            self._build_model_mat4()
        head = getattr(self, '_head_pos_w', None)
        if head is None:
            head_key = None
        else:
            head_key = (round(float(head[0]), 2), round(float(head[1]), 2), round(float(head[2]), 2))
        key = (
            round(float(self.screen_width), 6), round(float(self.screen_height), 6),
            round(float(self.screen_distance), 6), round(float(self.screen_pan_x), 6),
            round(float(self.screen_pan_y), 6), round(float(self.screen_yaw), 6),
            round(float(self.screen_pitch), 6), round(float(self.screen_roll), 6),
            head_key,
        )
        if key == getattr(self, '_frost_layout_cache_key', None):
            return self._frost_layout_cache_val

        R, center = self._screen_effect_basis()
        if head is None:
            head_w = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        else:
            head_w = np.asarray(head, dtype=np.float32)
        head_local = R[:3, :3].T @ (head_w - center)
        front_depth = max(float(head_local[2]) + 0.55, float(self.screen_distance) + 0.35, 0.75)
        front_half_w = max(float(self.screen_width) * 0.5, abs(float(head_local[0])) + 0.65, 0.65)
        front_half_h = max(float(self.screen_height) * 0.5, abs(float(head_local[1])) + 0.65, 0.65)
        val = (R, center, front_depth, front_half_w, front_half_h)
        self._frost_layout_cache_key = key
        self._frost_layout_cache_val = val
        return val

    def _build_flat_frost_verts(self, front_half_w, front_half_h):
        sx = max(float(self.screen_width) * 0.5, 1e-6)
        sy = max(float(self.screen_height) * 0.5, 1e-6)
        fx = front_half_w / sx
        fy = front_half_h / sy

        def _quad(a, b, c, d):
            return a + b + c + c + b + d

        # Each quad connects a screen edge to the expanded front plane.
        # UVs stay pinned to the screen edge pixel so veil samples only the edge.
        quads = [
            # Left edge.
            ([-1, -1, 0, 0, 1], [-1,  1, 0, 0, 0], [-fx, -fy, 1, 0, 1], [-fx,  fy, 1, 0, 0]),
            # Right edge.
            ([ 1, -1, 0, 1, 1], [ 1,  1, 0, 1, 0], [ fx, -fy, 1, 1, 1], [ fx,  fy, 1, 1, 0]),
            # Bottom edge.
            ([-1, -1, 0, 0, 1], [ 1, -1, 0, 1, 1], [-fx, -fy, 1, 0, 1], [ fx, -fy, 1, 1, 1]),
            # Top edge.
            ([-1,  1, 0, 0, 0], [ 1,  1, 0, 1, 0], [-fx,  fy, 1, 0, 0], [ fx,  fy, 1, 1, 0]),
        ]
        verts = []
        for quad in quads:
            verts.extend(_quad(*quad))
        return np.array(verts, dtype='f4')

    def _build_curved_frost_verts(self, front_depth, front_half_w, front_half_h, N=48):
        surface = self._build_curved_screen_verts(N=N).reshape((-1, 5))
        R, center = self._screen_effect_basis()
        rot = R[:3, :3]
        axis = self._screen_curve_mode()

        def _front(u, v):
            lx = (float(u) * 2.0 - 1.0) * front_half_w
            ly = (float(v) * 2.0 - 1.0) * front_half_h
            local = np.array([lx, ly, front_depth], dtype=np.float32)
            return center + rot @ local

        def _v(world, uv, local_z):
            u, v = float(uv[0]), float(uv[1])
            lx = u * 2.0 - 1.0
            ly = v * 2.0 - 1.0
            return [float(world[0]), float(world[1]), float(world[2]), u, v, lx, ly, local_z]

        def _append_quad(out, rear0, front0, rear1, front1):
            out.extend(rear0)
            out.extend(rear1)
            out.extend(front0)
            out.extend(front0)
            out.extend(rear1)
            out.extend(front1)

        verts = []
        cols = N + 1
        if axis == 'vertical':
            for side in (0, 1):
                pairs = []
                for i in range(cols):
                    rear = surface[i * 2 + side]
                    uv = (rear[3], rear[4])
                    pairs.append((_v(rear[:3], uv, 0.0), _v(_front(*uv), uv, 1.0)))
                for i in range(cols - 1):
                    _append_quad(verts, pairs[i][0], pairs[i][1], pairs[i + 1][0], pairs[i + 1][1])

            for v in (0.0, 1.0):
                rear0 = self._screen_uv_to_world(0.0, v)
                rear1 = self._screen_uv_to_world(1.0, v)
                q0 = (_v(rear0, (0.0, v), 0.0), _v(_front(0.0, v), (0.0, v), 1.0))
                q1 = (_v(rear1, (1.0, v), 0.0), _v(_front(1.0, v), (1.0, v), 1.0))
                _append_quad(verts, q0[0], q0[1], q1[0], q1[1])
        else:
            for row in (1, 0):
                pairs = []
                for i in range(cols):
                    rear = surface[i * 2 + row]
                    uv = (rear[3], rear[4])
                    pairs.append((_v(rear[:3], uv, 0.0), _v(_front(*uv), uv, 1.0)))
                for i in range(cols - 1):
                    _append_quad(verts, pairs[i][0], pairs[i][1], pairs[i + 1][0], pairs[i + 1][1])

            for col in (0, cols - 1):
                rear0 = surface[col * 2 + 0]
                rear1 = surface[col * 2 + 1]
                uv0 = (rear0[3], rear0[4])
                uv1 = (rear1[3], rear1[4])
                q0 = (_v(rear0[:3], uv0, 0.0), _v(_front(*uv0), uv0, 1.0))
                q1 = (_v(rear1[:3], uv1, 0.0), _v(_front(*uv1), uv1, 1.0))
                _append_quad(verts, q0[0], q0[1], q1[0], q1[1])

        return np.array(verts, dtype='f4')

    def _render_screen_background_effects(self, mgl_fbo, vp_mat, env_model_active=False,
                                          passthrough_active=False):
        mode = getattr(self, '_active_glow_mode_cached', None) or self._active_glow_mode()
        if mode != 'glow' or passthrough_active:
            return
        if env_model_active:
            return
        self._render_glow(mgl_fbo, vp_mat)

    def _render_screen_foreground_effects(self, mgl_fbo, vp_mat, passthrough_active=False):
        if passthrough_active:
            return
        mode = getattr(self, '_active_glow_mode_cached', None) or self._active_glow_mode()
        if mode == 'veil':
            self._render_frost_veil(mgl_fbo, vp_mat)
        elif mode == 'frosted':
            self._render_frost_glow(mgl_fbo, vp_mat)

    def _render_curved_glow(self, mgl_fbo, vp_mat, glow_intensity):
        if self.screen_height is None:
            return
        if self._curved_glow_prog is None or self._curved_glow_vao is None:
            return

        screen_long = max(self.screen_width, self.screen_height)
        glow_scale = screen_long / max(self._glow_ref_screen, 1e-6)
        glow_width = self._glow_width_m * glow_scale
        glow_range = glow_width * 0.75
        glow_margin = glow_range
        glow_w = self.screen_width + 2 * glow_margin
        glow_h = self.screen_height + 2 * glow_margin
        uv_glow_range = glow_range / max(glow_w, glow_h, 1e-6)
        inner_w = self.screen_width / glow_w
        inner_h = self.screen_height / glow_h

        params = (
            round(float(glow_w), 6), round(float(glow_h), 6),
            float(self.screen_distance), float(self.screen_pan_x), float(self.screen_pan_y),
            float(self.screen_yaw), float(self.screen_pitch), float(self.screen_roll),
            self._screen_curve_mode(),
        )
        if params != self._curved_glow_verts_params:
            verts = self._build_curved_screen_verts(
                width_override=glow_w,
                height_override=glow_h,
                normal_offset=-0.002,
            )
            self._curved_glow_vbo.write(verts.tobytes())
            self._curved_glow_verts_params = params

        self.ctx.depth_mask = False
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        self._advance_glow_color()
        self._curved_glow_prog['u_mvp'].write(vp_mat.T.astype('f4').tobytes())
        self._curved_glow_prog['u_screen_half'].value = (inner_w * 0.5, inner_h * 0.5)
        self._curved_glow_prog['u_glow_color'].value = self._glow_color
        self._curved_glow_prog['u_glow_inv_range'].value = 1.0 / max(uv_glow_range, 1e-6)
        self._curved_glow_prog['u_glow_intensity'].value = glow_intensity
        self._curved_glow_vao.render(moderngl.TRIANGLE_STRIP, vertices=(48 + 1) * 2)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.depth_mask = True
        self.ctx.enable(moderngl.DEPTH_TEST)

    def _render_glow(self, mgl_fbo, vp_mat):
        """Render a soft glow outside the screen edges using a larger quad."""
        glow_intensity = self._glow_intensity * self._glow_intensity_multiplier
        if glow_intensity <= 0.0 or self.screen_height is None:
            return
        if self._screen_curved:
            self._render_curved_glow(mgl_fbo, vp_mat, glow_intensity)
            return

        # Glow scales with screen size, referenced to a default 2.4m screen
        screen_long = max(self.screen_width, self.screen_height)
        glow_scale = screen_long / self._glow_ref_screen
        glow_width = self._glow_width_m * glow_scale

        glow_range = glow_width * 0.75      # pixel density reaches 0 at this finite tail edge
        glow_margin = glow_range
        glow_w = self.screen_width  + 2 * glow_margin
        glow_h = self.screen_height + 2 * glow_margin
        uv_glow_range = glow_range / max(glow_w, glow_h)
        inner_w = self.screen_width / glow_w
        inner_h = self.screen_height / glow_h

        band_params = (round(float(inner_w), 6), round(float(inner_h), 6))
        if self._glow_band_params != band_params and self._glow_vbo is not None:
            u0 = 0.5 - inner_w * 0.5
            u1 = 0.5 + inner_w * 0.5
            v0 = 0.5 - inner_h * 0.5
            v1 = 0.5 + inner_h * 0.5
            x0, x1 = u0 * 2.0 - 1.0, u1 * 2.0 - 1.0
            y0, y1 = v0 * 2.0 - 1.0, v1 * 2.0 - 1.0

            def _quad(ax, ay, au, av, bx, by, bu, bv, cx, cy_, cu, cv, dx, dy, du, dv):
                return [
                    ax, ay, au, av, bx, by, bu, bv, cx, cy_, cu, cv,
                    bx, by, bu, bv, dx, dy, du, dv, cx, cy_, cu, cv,
                ]

            verts = []
            verts += _quad(-1, -1, 0, 0,  1, -1, 1, 0,  -1, y0, 0, v0,  1, y0, 1, v0)
            verts += _quad(-1, y1, 0, v1,  1, y1, 1, v1,  -1, 1, 0, 1,  1, 1, 1, 1)
            verts += _quad(-1, y0, 0, v0,  x0, y0, u0, v0,  -1, y1, 0, v1,  x0, y1, u0, v1)
            verts += _quad(x1, y0, u1, v0,  1, y0, 1, v0,  x1, y1, u1, v1,  1, y1, 1, v1)
            bands = np.array(verts, dtype='f4')
            self._glow_vbo.write(bands.tobytes())
            self._glow_band_params = band_params

        self.ctx.depth_mask = False
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        # Premultiplied alpha: the shader outputs (color * glow, glow), and
        # the blend is `out = src.rgb + dst.rgb * (1 - src.a)`.  This is LINEAR
        # in `glow` and is the standard fix for the concentric-ring banding
        # that SRC_ALPHA blending produces with an alpha-weighted colour.
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA

        # Smoothly interpolate glow color towards sampled frame average.
        self._advance_glow_color()

        # Flat glow bands around the screen only; the screen-sized centre is
        # skipped by geometry to avoid wasted fill-rate.
        sx = glow_w / 2.0
        sy = glow_h / 2.0
        model_params = (
            round(float(sx), 6), round(float(sy), 6),
            float(self.screen_yaw), float(self.screen_pitch), float(self.screen_roll),
            float(self.screen_pan_x), float(self.screen_pan_y), float(self.screen_distance),
        )
        if model_params != self._glow_model_params:
            cy  = math.cos(self.screen_yaw);   sy_ = math.sin(self.screen_yaw)
            cp  = math.cos(self.screen_pitch); sp  = math.sin(self.screen_pitch)
            cr  = math.cos(self.screen_roll);  sr  = math.sin(self.screen_roll)
            rot_y = np.array([[ cy, 0, sy_, 0], [0, 1, 0, 0], [-sy_, 0, cy, 0], [0, 0, 0, 1]], dtype='f4')
            rot_x = np.array([[1, 0, 0, 0], [0, cp, -sp, 0], [0, sp, cp, 0], [0, 0, 0, 1]], dtype='f4')
            rot_z = np.array([[cr, -sr, 0, 0], [sr,  cr, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype='f4')
            R = rot_y @ rot_x @ rot_z
            S = np.diag([sx, sy, 1.0, 1.0]).astype('f4')
            T = np.eye(4, dtype='f4')
            T[0, 3] = self.screen_pan_x
            T[1, 3] = self.screen_pan_y
            T[2, 3] = -self.screen_distance - 0.002
            self._glow_model_mat = T @ R @ S
            self._glow_model_params = model_params
        glow_model = self._glow_model_mat
        mvp = vp_mat @ glow_model
        self._glow_prog['u_mvp'].write(mvp.T.astype('f4').tobytes())
        self._glow_prog['u_screen_half'].value = (inner_w * 0.5, inner_h * 0.5)
        self._glow_prog['u_glow_color'].value  = self._glow_color
        self._glow_prog['u_glow_inv_range'].value  = 1.0 / max(uv_glow_range, 1e-6)
        self._glow_prog['u_glow_intensity'].value = glow_intensity
        self._glow_vao.render(moderngl.TRIANGLES, vertices=24)

        self.ctx.disable(moderngl.BLEND)
        self.ctx.depth_mask = True
        self.ctx.enable(moderngl.DEPTH_TEST)

    def _frost_source_texture(self):
        return getattr(self, 'color_tex', None)

    def _color_mipmaps_needed(self, mode=None):
        if mode is None:
            mode = self._active_glow_mode()
        if mode == 'veil':
            return float(getattr(self, '_frost_veil_lod', 0.0) or 0.0) > 0.001
        return mode == 'frosted'

    def _maybe_generate_color_mipmaps(self, mode=None):
        needs_mips = self._color_mipmaps_needed(mode)
        if self.color_tex is not None and getattr(self, '_color_tex_mipmap_filter_active', None) != needs_mips:
            self.color_tex.filter = (
                (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
                if needs_mips else
                (moderngl.LINEAR, moderngl.LINEAR)
            )
            self._color_tex_mipmap_filter_active = needs_mips
        if needs_mips:
            glGenerateMipmap(GL_TEXTURE_2D)

    def _set_frost_uniforms(self, prog, source_tex, intensity):
        self._set_frost_uniform_values(prog, source_tex, intensity)

    def _set_frost_uniform_values(self, prog, source_tex, intensity, *, edge_inset=None, lod=None,
                                  threshold=None, alpha=None, noise_scale=None, beam_softness=0.34,
                                  blend=None, thickness=None, diffuse=None):
        source_tex.use(location=0)
        crop = self._movie_crop_render_uv_fast()
        values = (
            crop,
            float(getattr(self, '_frost_glow_inset', 0.045) if edge_inset is None else edge_inset),
            float(getattr(self, '_frost_glow_lod', 5.4) if lod is None else lod),
            float(getattr(self, '_frost_glow_threshold', 0.46) if threshold is None else threshold),
            float(intensity),
            float(getattr(self, '_frost_glow_alpha', 0.42) if alpha is None else alpha),
            float(getattr(self, '_frost_glow_noise_scale', 54.0) if noise_scale is None else noise_scale),
            float(beam_softness),
            float(getattr(self, '_frost_glow_blend', 1.35) if blend is None else blend),
            float(getattr(self, '_frost_glow_thickness', 1.6) if thickness is None else thickness),
            float(getattr(self, '_frost_glow_diffuse', 0.85) if diffuse is None else diffuse),
        )
        time_value = float(getattr(self, '_frame_now', 0.0) or time.perf_counter())
        key = id(prog)
        cache = getattr(self, '_frost_uniform_cache', None)
        if not isinstance(cache, dict):
            cache = {}
            self._frost_uniform_cache = cache
        cached = cache.get(key)
        if cached is None or cached[0] != values:
            self._set_source_crop_uniform(prog, crop)
            prog['u_edge_inset'].value = values[1]
            prog['u_lod'].value = values[2]
            prog['u_threshold'].value = values[3]
            prog['u_intensity'].value = values[4]
            prog['u_frost_alpha'].value = values[5]
            prog['u_noise_scale'].value = values[6]
            prog['u_beam_softness'].value = values[7]
            prog['u_frost_blend'].value = values[8]
            prog['u_beam_thickness'].value = values[9]
            prog['u_diffuse_scatter'].value = values[10]
            cached = (values, None)
        if cached[1] != time_value:
            prog['u_time'].value = time_value
            cached = (values, time_value)
        cache[key] = cached

    def _set_frost_veil_uniforms(self, prog, source_tex, intensity):
        source_tex.use(location=0)
        crop = self._movie_crop_render_uv_fast()
        values = (
            crop,
            0.02,
            float(intensity),
            float(getattr(self, '_frost_veil_alpha', 1.0)),
            0.34,
            3.0,
        )
        key = id(prog)
        cache = getattr(self, '_frost_uniform_cache', None)
        if not isinstance(cache, dict):
            cache = {}
            self._frost_uniform_cache = cache
        if cache.get(key) == values:
            return
        self._set_source_crop_uniform(prog, crop)
        prog['u_edge_inset'].value = values[1]
        prog['u_intensity'].value = values[2]
        prog['u_frost_alpha'].value = values[3]
        prog['u_beam_softness'].value = values[4]
        prog['u_beam_thickness'].value = values[5]
        cache[key] = values

    def _mat4_uniform_bytes(self, mat):
        arr = np.asarray(mat)
        transposed = arr.T
        if transposed.dtype == np.float32 and transposed.flags['C_CONTIGUOUS']:
            return transposed.tobytes()
        return transposed.astype('f4', copy=False).tobytes()

    def _frost_model_bytes(self, beam_len):
        key = (
            round(float(self.screen_width), 6), round(float(self.screen_height), 6),
            round(float(self.screen_distance), 6), round(float(self.screen_pan_x), 6),
            round(float(self.screen_pan_y), 6), round(float(self.screen_yaw), 6),
            round(float(self.screen_pitch), 6), round(float(self.screen_roll), 6),
            round(float(beam_len), 2),
        )
        if key == getattr(self, '_frost_model_cache_key', None):
            cached = getattr(self, '_frost_model_cache_bytes', None)
            if cached is not None:
                return cached
        model = self._screen_effect_model(self.screen_width, self.screen_height, z_scale=beam_len)
        data = model.T.astype('f4').tobytes()
        self._frost_model_cache_key = key
        self._frost_model_cache_bytes = data
        return data

    def _write_frost_model_uniform(self, prog, data):
        cache = getattr(self, '_frost_model_uniform_cache', None)
        if not isinstance(cache, dict):
            cache = {}
            self._frost_model_uniform_cache = cache
        key = id(prog)
        if cache.get(key) == data:
            return
        prog['u_model'].write(data)
        cache[key] = data

    def _render_curved_frost_with_uniforms(self, mgl_fbo, vp_mat, source_tex, intensity, uniform_setter,
                                           prog=None, vao=None):
        prog = self._curved_frost_prog if prog is None else prog
        vao = self._curved_frost_vao if vao is None else vao
        if prog is None or vao is None:
            return
        _, _, front_depth, front_half_w, front_half_h = self._frost_front_layout()
        params = (
            round(float(self.screen_width), 6), round(float(self.screen_height), 6),
            float(self.screen_distance), float(self.screen_pan_x), float(self.screen_pan_y),
            float(self.screen_yaw), float(self.screen_pitch), float(self.screen_roll),
            round(float(front_depth), 2), round(float(front_half_w), 2), round(float(front_half_h), 2),
            self._screen_curve_mode(),
        )
        if params != self._curved_frost_verts_params:
            verts = self._build_curved_frost_verts(front_depth, front_half_w, front_half_h)
            self._curved_frost_vbo.write(verts.tobytes())
            self._curved_frost_verts_params = params

        self.ctx.depth_mask = False
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        prog['u_vp'].write(self._mat4_uniform_bytes(vp_mat))
        uniform_setter(prog, source_tex, intensity)
        vao.render(moderngl.TRIANGLES, vertices=(12 * 48) + 12)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.depth_mask = True
        self.ctx.enable(moderngl.DEPTH_TEST)

    def _render_flat_frost_with_uniforms(self, mgl_fbo, vp_mat, source_tex, intensity, uniform_setter,
                                         prog=None, vao=None):
        prog = self._frost_glow_prog if prog is None else prog
        vao = self._frost_glow_vao if vao is None else vao
        if prog is None or vao is None:
            return
        _, _, beam_len, front_half_w, front_half_h = self._frost_front_layout()
        params = (
            round(float(self.screen_width), 6), round(float(self.screen_height), 6),
            round(float(front_half_w), 2), round(float(front_half_h), 2),
        )
        if params != self._frost_glow_verts_params:
            self._frost_glow_vbo.write(self._build_flat_frost_verts(front_half_w, front_half_h).tobytes())
            self._frost_glow_verts_params = params

        self.ctx.depth_mask = False
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        self._write_frost_model_uniform(prog, self._frost_model_bytes(beam_len))
        prog['u_vp'].write(self._mat4_uniform_bytes(vp_mat))
        uniform_setter(prog, source_tex, intensity)
        vao.render(moderngl.TRIANGLES, vertices=24)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.depth_mask = True
        self.ctx.enable(moderngl.DEPTH_TEST)

    def _render_curved_frost_glow(self, mgl_fbo, vp_mat, source_tex, intensity):
        self._render_curved_frost_with_uniforms(mgl_fbo, vp_mat, source_tex, intensity, self._set_frost_uniforms)

    def _render_frost_glow(self, mgl_fbo, vp_mat):
        intensity = float(getattr(self, '_frost_glow_intensity', 1.0))
        intensity *= max(float(getattr(self, '_glow_intensity_multiplier', 0.0)), 0.0)
        if intensity <= 0.0 or self.screen_height is None:
            return
        source_tex = self._frost_source_texture()
        if source_tex is None:
            return
        if self._screen_curved:
            self._render_curved_frost_with_uniforms(mgl_fbo, vp_mat, source_tex, intensity, self._set_frost_uniforms)
        else:
            self._render_flat_frost_with_uniforms(mgl_fbo, vp_mat, source_tex, intensity, self._set_frost_uniforms)

    def _render_frost_veil(self, mgl_fbo, vp_mat):
        intensity = float(getattr(self, '_frost_veil_intensity', 1.0))
        intensity *= max(float(getattr(self, '_glow_intensity_multiplier', 0.0)), 0.0)
        if intensity <= 0.0 or self.screen_height is None:
            return
        source_tex = self._frost_source_texture()
        if source_tex is None:
            return
        if self._screen_curved:
            self._render_curved_frost_with_uniforms(
                mgl_fbo, vp_mat, source_tex, intensity, self._set_frost_veil_uniforms,
                prog=self._curved_veil_prog,
                vao=self._curved_veil_vao,
            )
        else:
            self._render_flat_frost_with_uniforms(
                mgl_fbo, vp_mat, source_tex, intensity, self._set_frost_veil_uniforms,
                prog=self._frost_veil_prog,
                vao=self._frost_veil_vao,
            )

    def _render_frosted_glow(self, mgl_fbo, vp_mat):
        self._render_frost_glow(mgl_fbo, vp_mat)

    def _render_frosted_veil(self, mgl_fbo, vp_mat):
        self._render_frost_veil(mgl_fbo, vp_mat)

    def _render_border(self, mgl_fbo, vp_mat):
        """Render a thin solid-color border slightly larger than the screen.

        Flat mode: oversized quad rendered before the screen so it peeks at the edges.
        Curved mode: oversized arc strip matching the curved screen geometry.
        Color is cyan when the user is grabbing/resizing, light grey otherwise.
        """
        if self.screen_height is None:
            return
        alpha = self._border_alpha
        if alpha <= 0.0:
            return

        if self._grabbed:
            color = (0.3, 0.7, 1.0, alpha)
        else:
            color = (0.75, 0.75, 0.75, alpha * 0.9)

        BORDER = self.screen_width / 300.0   # proportional to screen width

        mgl_fbo.use()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        if self._screen_curved and self._curved_border_vao is not None:
            # Build a border arc: same arc geometry but screen_width+2B wide,
            # screen_height+2B tall, and pushed 1 mm behind the screen surface.
            bw = self.screen_width  + 2 * BORDER
            bh = self.screen_height + 2 * BORDER
            params = (bw, bh, self.screen_distance, self.screen_pan_x,
                    self.screen_pan_y, self.screen_yaw, self.screen_pitch,
                    self.screen_roll, self._screen_curve_mode())
            if self._curved_border_verts_params != params:
                border_verts = self._build_curved_screen_verts(
                    width_override=bw, height_override=bh, dist_offset=0.001,
                )
                self._curved_border_vbo.write(border_verts.tobytes())
                self._curved_border_verts_params = params
            self._curved_border_prog['u_mvp'].write(vp_mat.T.tobytes())
            self._curved_border_prog['u_color'].value = color
            n_verts = (48 + 1) * 2
            self._curved_border_vao.render(moderngl.TRIANGLE_STRIP, vertices=n_verts)
        else:
            sx = self.screen_width  / 2.0 + BORDER
            sy = self.screen_height / 2.0 + BORDER
            cy  = math.cos(self.screen_yaw);   sy_ = math.sin(self.screen_yaw)
            cp  = math.cos(self.screen_pitch); sp  = math.sin(self.screen_pitch)
            rot_y = np.array([[ cy, 0, sy_, 0], [0, 1, 0, 0], [-sy_, 0, cy, 0], [0, 0, 0, 1]], dtype='f4')
            rot_x = np.array([[1, 0, 0, 0], [0, cp, -sp, 0], [0, sp, cp, 0], [0, 0, 0, 1]], dtype='f4')
            cr = math.cos(self.screen_roll); sr = math.sin(self.screen_roll)
            rot_z = np.array([[cr, -sr, 0, 0], [sr, cr, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype='f4')
            S = np.diag([sx, sy, 1.0, 1.0]).astype('f4')
            R = rot_y @ rot_x @ rot_z
            T = np.eye(4, dtype='f4')
            T[0, 3] = self.screen_pan_x
            T[1, 3] = self.screen_pan_y
            T[2, 3] = -self.screen_distance - 0.001
            border_model = T @ R @ S
            mvp = vp_mat @ border_model
            self._border_prog['u_mvp'].write(mvp.T.astype('f4').tobytes())
            self._border_prog['u_color'].value = color
            self._border_vao.render(moderngl.TRIANGLE_STRIP)

        self.ctx.disable(moderngl.BLEND)

    def _get_panorama_texture(self):
        path = getattr(self, '_panorama_background_path', None)
        if not path:
            return None
        path = os.path.abspath(path)
        if self._panorama_tex is not None and self._panorama_tex_path == path:
            return self._panorama_tex
        if self._panorama_tex is not None:
            try:
                self._panorama_tex.release()
            except Exception:
                pass
            self._panorama_tex = None
            self._panorama_tex_path = None
        try:
            img = Image.open(path).convert('RGB')
            max_tex = int(getattr(self.ctx, 'info', {}).get('GL_MAX_TEXTURE_SIZE', 8192) or 8192)
            if max(img.size) > max_tex:
                scale = float(max_tex) / float(max(img.size))
                new_size = (
                    max(1, int(round(img.size[0] * scale))),
                    max(1, int(round(img.size[1] * scale))),
                )
                resample = getattr(getattr(Image, 'Resampling', Image), 'LANCZOS', Image.BICUBIC)
                img = img.resize(new_size, resample)
            arr = np.asarray(img, dtype=np.uint8)
            tex = self.ctx.texture(img.size, 3, arr.tobytes())
            tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            try:
                tex.repeat_x = True
                tex.repeat_y = False
            except Exception:
                pass
            try:
                tex.build_mipmaps()
            except Exception:
                tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._panorama_tex = tex
            self._panorama_tex_path = path
            print(f"[OpenXRViewer] Panorama background loaded: {path} ({img.size[0]}x{img.size[1]})")
            return tex
        except Exception as exc:
            print(f"[OpenXRViewer] Panorama background load failed: {exc}")
            return None

    def _render_panorama_background(self, mgl_fbo, view_mat, proj_mat):
        if self._panorama_prog is None or self._panorama_vao is None:
            return
        tex = self._get_panorama_texture()
        if tex is None:
            return
        settings = getattr(self, '_panorama_background_settings', {}) or {}
        try:
            yaw_offset = math.radians(float(settings.get('yaw_offset_deg', 0.0))) / (2.0 * math.pi)
        except (TypeError, ValueError):
            yaw_offset = 0.0
        try:
            exposure = float(settings.get('exposure', 1.0))
        except (TypeError, ValueError):
            exposure = 1.0
        flip_y = 1 if bool(settings.get('flip_y', False)) else 0

        view_rot = np.array(view_mat, dtype=np.float32, copy=True)
        view_rot[:3, 3] = 0.0
        try:
            inv_proj = np.linalg.inv(proj_mat.astype(np.float32))
            inv_view_rot = np.linalg.inv(view_rot)
        except Exception:
            return

        mgl_fbo.use()
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.depth_mask = False
        self.ctx.disable(moderngl.BLEND)
        tex.use(location=8)
        self._panorama_prog['u_inv_proj'].write(inv_proj.T.astype('f4').tobytes())
        self._panorama_prog['u_inv_view_rot'].write(inv_view_rot.T.astype('f4').tobytes())
        self._panorama_prog['u_yaw_offset'].value = float(yaw_offset)
        self._panorama_prog['u_exposure'].value = max(0.0, float(exposure))
        self._panorama_prog['u_flip_y'].value = flip_y
        self._panorama_vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.depth_mask = True
        self.ctx.enable(moderngl.DEPTH_TEST)

    def _render_eye(self, eye_index, mgl_fbo, view_mat, proj_mat, flip_y=False):
        """Render one eye's parallax view into the swapchain FBO using world-space MVP.

        Left eye:  u_eye_offset = -ipd/2
        Right eye: u_eye_offset = +ipd/2

        If flip_y is True the projection is Y-flipped so glReadPixels produces
        top-down rows for D3D11, eliminating the CPU row-reversal copy.
        """
        if flip_y:
            proj_mat = proj_mat.copy()
            proj_mat[1, :] = -proj_mat[1, :]
        sc_w, sc_h = self._swapchain_sizes[eye_index]

        # The swapchain is GL_SRGB8_ALPHA8, but the desktop capture texture is already
        # gamma-encoded.  Disabling GL_FRAMEBUFFER_SRGB prevents AMD (and compliant
        # drivers) from applying a second sRGB encoding pass on write, which would
        # cause pale/washed-out colours.
        glDisable(GL_FRAMEBUFFER_SRGB)

        mgl_fbo.use()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.depth_mask = True
        self.ctx.disable(moderngl.BLEND)
        self.ctx.viewport = (0, 0, sc_w, sc_h)
        bg_a = 1.0
        bg_r, bg_g, bg_b = _BG_COLORS[self._bg_color_idx]
        mgl_fbo.clear(bg_r, bg_g, bg_b, bg_a, depth=1.0)

        if not self._screen_visible:
            self.ctx.screen.use()
            return

        if self.color_tex is None or self.depth_tex is None:
            self.ctx.screen.use()
            return

        vp_mat = proj_mat @ view_mat
        view_inv = None

        # -3. Environment model (glTF 3D scene). Locked room profiles keep
        # depth through the desktop draw so nearer furniture can occlude the
        # screen; other environments retain the legacy backdrop-only behavior.
        passthrough_active = self._bg_color_idx == 1
        if not passthrough_active and self._panorama_background_path:
            self._render_panorama_background(mgl_fbo, view_mat, proj_mat)
        env_model_active = bool(
            not passthrough_active
            and self._env_model_visible
            and self._env_model_prims
        )
        env_depth_occlusion = bool(env_model_active and self._environment_screen_locked())
        if env_model_active:
            view_inv = _view_mat_inv(view_mat)
            self._render_env_model(mgl_fbo, vp_mat, view_mat, view_inv)
            mgl_fbo.use()
            if not env_depth_occlusion:
                glClear(GL_DEPTH_BUFFER_BIT)

        # 0. Screen effects behind the desktop quad.
        if self._active_glow_mode_cached != 'off':
            self._render_screen_background_effects(
                mgl_fbo,
                vp_mat,
                env_model_active=env_model_active,
                passthrough_active=passthrough_active,
            )

        # 1. Main screen (flat quad or cylindrical curved arc)
        mgl_fbo.use()
        self.color_tex.use(location=0)
        self.depth_tex.use(location=1)
        source_crop = self._movie_crop_render_uv_fast()

        eye_sign = -1.0 if eye_index == 0 else 1.0

        if self._screen_curved and self._curved_prog is not None:
            # Curved path: build world-space arc verts, upload, draw with vp_mat only
            prog = self._curved_prog
            screen_depth_bias = _SCREEN_ENV_DEPTH_BIAS_M if env_depth_occlusion else 0.0
            params = (self.screen_width, self.screen_height, self.screen_distance,
                    self.screen_pan_x, self.screen_pan_y, self.screen_yaw,
                    self.screen_pitch, self.screen_roll, screen_depth_bias,
                    self._screen_curve_mode())
            if self._curved_verts_params != params:
                arc_verts = self._build_curved_screen_verts(normal_offset=screen_depth_bias)
                self._curved_vbo.write(arc_verts.tobytes())
                self._curved_verts_params = params
            prog['u_mvp'].write(vp_mat.T.tobytes())
            prog['u_eye_offset'].value     = eye_sign * self.ipd_uv / 2.0
            prog['u_depth_strength'].value = self.depth_strength * self.depth_ratio
            prog['u_convergence'].value    = float(self.convergence)
            prog['u_roll'].value           = self.screen_roll
            prog['u_corner_radius'].value  = self._corner_radius
            self._set_source_crop_uniform(prog, source_crop)
            n_verts = (48 + 1) * 2
            self._curved_vao.render(moderngl.TRIANGLE_STRIP, vertices=n_verts)
        else:
            # Flat path: standard MVP quad
            screen_depth_bias = _SCREEN_ENV_DEPTH_BIAS_M if env_depth_occlusion else 0.0
            model = self._build_model_mat4(normal_offset=screen_depth_bias)
            mvp   = vp_mat @ model
            self.prog['u_mvp'].write(mvp.T.tobytes())
            self.prog['u_eye_offset'].value     = eye_sign * self.ipd_uv / 2.0
            self.prog['u_depth_strength'].value = self.depth_strength * self.depth_ratio
            # Keep convergence in sync user-driven divergence input updates self.convergence
            # at runtime; pushing it here ensures any external change is reflected per-frame.
            self.prog['u_convergence'].value = float(self.convergence)
            self.prog['u_roll'].value      = self.screen_roll
            self.prog['u_corner_radius'].value = self._corner_radius
            self._set_source_crop_uniform(self.prog, source_crop)
            # Render screen WITHOUT alpha blending so the shader's edge alpha is written
            # directly into the swapchain framebuffer. The XR compositor composites those
            # near-zero-alpha edge pixels against the VR background producing a clean soft
            # edge. With SRC_ALPHA blending the edge pixels would blend against the FBO's
            # opaque black clear, creating a persistent dark halo visible at all times.
            self.quad_vao.render(moderngl.TRIANGLE_STRIP)

        # 2. Screen effects on and around the desktop quad.
        if self._active_glow_mode_cached not in ('off', 'glow'):
            self._render_screen_foreground_effects(
                mgl_fbo,
                vp_mat,
                passthrough_active=passthrough_active,
            )

        # Clear depth so UI overlays (FPS panel, OSD, keyboard, controllers,
        # lasers) are never occluded by environment geometry (e.g. cinema floor).
        if env_depth_occlusion:
            mgl_fbo.use()
            glClear(GL_DEPTH_BUFFER_BIT)

        # 3. Keyboard
        if self._keyboard_visible and self._keyboard_tex is not None:
            self._render_keyboard(mgl_fbo, vp_mat)

        # 5. Depth OSD (floating panel, always checked method handles its own alpha)
        if self._depth_osd_tex is not None:
            self._render_depth_osd(eye_index, mgl_fbo, vp_mat)

        # 5b. Screen-info OSD (size + distance, shown while right grip + stick adjusts)
        if self._screen_osd_tex is not None:
            self._render_screen_osd(eye_index, mgl_fbo, vp_mat)

        # 5c. Preset OSD (name, shown briefly after cycling presets with Y button)
        if self._preset_osd_tex is not None:
            self._render_preset_osd(eye_index, mgl_fbo, vp_mat)

        # 5d. Seat adjust OSD (position indicator, shown during/after seat adjust)
        if self._seat_adjust_osd_tex is not None:
            self._render_seat_adjust_osd(eye_index, mgl_fbo, vp_mat)

        # 6b. Brand OSD (controller model indicator, attached to right controller)
        if self._brand_osd_tex is not None and self._grip_mat_r is not None:
            self._render_brand_osd(eye_index, mgl_fbo, vp_mat)

        # Update laser beam cache once per frame (shared across both eyes).
        if getattr(self, '_beams_frame', -1) != self._frame_count:
            self._cached_beams = self._laser_beam_setup()
            self._beams_frame = self._frame_count
        has_beams = bool(getattr(self, '_cached_beams', None))

        # 7. Laser beam (opaque rainbow)
        if has_beams:
            self._render_lasers(mgl_fbo, vp_mat, view_mat, blend=False)
        needs_view_inv = bool(self._ctrl_prims_l or self._ctrl_prims_r) or has_beams
        if needs_view_inv and view_inv is None:
            view_inv = _view_mat_inv(view_mat)

        # 8. VR Controller models
        if self._ctrl_prims_l or self._ctrl_prims_r:
            self._render_controllers(mgl_fbo, vp_mat, view_mat, view_inv)

        # 9. FPS overlay
        if self._fps_overlay_visible and self._overlay_tex is not None:
            self._render_fps_overlay(eye_index, mgl_fbo, vp_mat)

        # 10. Laser hit circles (semi-transparent)
        if has_beams:
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            self._render_lasers(mgl_fbo, vp_mat, view_mat, blend=True, view_inv=view_inv)
            self.ctx.disable(moderngl.BLEND)
            self.ctx.enable(moderngl.DEPTH_TEST)

        # 11. Calibration panel
        if self._calibration_mode:
            self._render_calibration_panel(mgl_fbo, vp_mat)

        # 12. Help/shortcut panel
        if self._fps_overlay_visible and self._help_panel_visible and self._help_tex is not None:
            self._render_help_panel(mgl_fbo, vp_mat)

        self.ctx.screen.use()
    
    # OpenXR event loop
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
                    print("[OpenXRViewer] Session READY rendering started")

                elif state in (
                    xr.SessionState.STOPPING,
                    xr.SessionState.LOSS_PENDING,
                    xr.SessionState.EXITING,
                ):
                    xr.end_session(self._xr_session)
                    self._session_running = False
                    print(f"[OpenXRViewer] Session state {state.name}; rendering paused")

            elif event_type == xr.StructureType.EVENT_DATA_REFERENCE_SPACE_CHANGE_PENDING:
                # Long-pressing the Oculus/home button triggers a playspace recenter,
                # which emits this event.  Defer the re-seat to the next frame so
                # the Quest's own recenter is reflected in the views first.
                self._pending_recenter = True

            elif event_type == xr.StructureType.EVENT_DATA_INSTANCE_LOSS_PENDING:
                print("[OpenXRViewer] Instance loss pending shutting down")
                shutdown_event.set()
                break

    def _read_bool_action(self, action, hand_path_str="/user/hand/left"):
        """Return True if the boolean action is currently pressed on the given hand."""
        if action is None:
            return False
        try:
            path = (self._path_left
                    if hand_path_str == "/user/hand/left" else self._path_right)
            if path is None:
                path = xr.string_to_path(self._xr_instance, hand_path_str)
            state = xr.get_action_state_boolean(
                self._xr_session,
                xr.ActionStateGetInfo(action=action, subaction_path=path),
            )
            return bool(state.is_active and state.current_state)
        except Exception:
            return False

    def _read_bool_edge(self, action, hand_path_str, prev_state):
        """Return True on the rising edge of a boolean action.

        Tries to use the OpenXR runtime's `changed` flag via the raw ctypes struct
        (pyopenxr may not expose it as a Python attribute).  Falls back to manual
        frame-to-frame comparison if the ctypes path fails.
        """
        if action is None:
            return False
        try:
            path = (self._path_left
                    if hand_path_str == "/user/hand/left" else self._path_right)
            if path is None:
                path = xr.string_to_path(self._xr_instance, hand_path_str)
            state = xr.get_action_state_boolean(
                self._xr_session,
                xr.ActionStateGetInfo(action=action, subaction_path=path),
            )
            pressed = bool(state.is_active and state.current_state)

            # pyopenxr wraps XrActionStateBoolean. Try the Python attribute first,
            # then fall back to reading the underlying ctypes struct.
            changed = False
            if hasattr(state, 'changed'):
                changed = bool(state.changed)
            else:
                # The struct is [isActive:i4, currentState:i4, changed:i4, ...]
                # changed is at byte offset 8 (after two 4-byte fields).
                try:
                    ptr = ctypes.cast(ctypes.byref(state), ctypes.POINTER(ctypes.c_int32))
                    changed = bool(ptr[2])  # offset 2 × 4 bytes
                except Exception:
                    pass

            if changed:
                return pressed   # runtime-confirmed edge
            # Fallback: manual rising-edge detection
            return pressed and not prev_state
        except Exception:
            return False

    def _update_trackpad_button_emu(self):
        """Compute per-frame Vive trackpad button emulation flags.

        On controllers with a clickable trackpad, the physical click position
        emulates face buttons:
          center (|y| <= 0.5) thumbstick click
          top    (y > 0.5)   B (right) / Y (left)
          bottom (y < -0.5)  A (right) / X (left)

        On controllers with real buttons the emulation is harmless: the real
        reads dominate via OR, and the thumbstick self-centres near (0,0) so
        only the center flag fires (matching the raw stick-click).
        """
        for hand, stick_act, click_act, attr_top, attr_bot, attr_ctr in [
            ("/user/hand/left",  self._act_left_stick,  self._act_left_stick_click,
             '_emu_y', '_emu_x', '_emu_lsc'),
            ("/user/hand/right", self._act_right_stick, self._act_right_stick_click,
             '_emu_b', '_emu_a', '_emu_rsc'),
        ]:
            clicked = self._read_bool_action(click_act, hand)
            if not clicked:
                setattr(self, attr_top, False)
                setattr(self, attr_bot, False)
                setattr(self, attr_ctr, False)
                continue
            try:
                path = self._path_left if hand == "/user/hand/left" else self._path_right
                state = xr.get_action_state_vector2f(
                    self._xr_session,
                    xr.ActionStateGetInfo(action=stick_act, subaction_path=path),
                )
                py = float(state.current_state.y) if state.is_active else 0.0
            except Exception:
                py = 0.0
            if py > _VIVE_TB_Y:
                setattr(self, attr_top, True)
                setattr(self, attr_bot, False)
                setattr(self, attr_ctr, False)
            elif py < -_VIVE_TB_Y:
                setattr(self, attr_top, False)
                setattr(self, attr_bot, True)
                setattr(self, attr_ctr, False)
            else:
                setattr(self, attr_top, False)
                setattr(self, attr_bot, False)
                setattr(self, attr_ctr, True)

    def _read_float_action(self, action, hand_path_str="/user/hand/left"):
        """Return the float value [0,1] of a trigger/squeeze action."""
        if action is None:
            return 0.0
        try:
            path = (self._path_left
                    if hand_path_str == "/user/hand/left" else self._path_right)
            if path is None:
                path = xr.string_to_path(self._xr_instance, hand_path_str)
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
            sh = self.screen_width * (self._effective_frame_aspect() if fw > 0 else 9.0 / 16.0)
        safe_w = max(self.screen_width, 1e-6)
        safe_h = max(sh, 1e-6)

        # Curved screen: solve ray <-> cylindrical arc intersection in the
        # screen-local frame. Horizontal curves bend across U (x/z cylinder);
        # vertical curves bend across V (y/z cylinder).
        if self._screen_curved:
            BEAM_MAX = 30.0
            half_w = self.screen_width / 2.0
            half_h = sh / 2.0
            half_ang = min(_CURVED_HALF_ANGLE_RAD, math.pi / 2)
            axis = self._screen_curve_mode()
            radius = (half_h if axis == 'vertical' else half_w) / max(half_ang, 1e-6)
            Rm, center = self._screen_effect_basis()
            inv_rot = Rm[:3, :3].astype('f8').T
            ro = inv_rot @ (ctrl_pos - center.astype('f8'))
            rd = inv_rot @ fwd_w

            if axis == 'vertical':
                a = float(rd[1] * rd[1] + rd[2] * rd[2])
                b = float(2.0 * (ro[1] * rd[1] + (ro[2] - radius) * rd[2]))
                c = float(ro[1] * ro[1] + (ro[2] - radius) * (ro[2] - radius) - radius * radius)
            else:
                a = float(rd[0] * rd[0] + rd[2] * rd[2])
                b = float(2.0 * (ro[0] * rd[0] + (ro[2] - radius) * rd[2]))
                c = float(ro[0] * ro[0] + (ro[2] - radius) * (ro[2] - radius) - radius * radius)
            if abs(a) < 1e-9:
                return None
            disc = b * b - 4.0 * a * c
            if disc < 0.0:
                return None
            sqrt_disc = math.sqrt(disc)
            hits = sorted(((-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)))
            for t_hit in hits:
                if t_hit < 0.01 or t_hit > BEAM_MAX:
                    continue
                vl = ro + rd * t_hit
                lx = float(vl[0])
                ly = float(vl[1])
                lz = float(vl[2])
                if axis == 'vertical':
                    if abs(lx) > half_w + 1e-6:
                        continue
                    ang = math.atan2(ly, radius - lz)
                    if ang < -half_ang - 1e-6 or ang > half_ang + 1e-6:
                        continue
                    u = (lx + half_w) / (2.0 * half_w) if half_w > 1e-9 else 0.5
                    v = (ang + half_ang) / (2.0 * half_ang) if half_ang > 1e-9 else 0.5
                else:
                    if abs(ly) > half_h + 1e-6:
                        continue
                    ang = math.atan2(lx, radius - lz)
                    if ang < -half_ang - 1e-6 or ang > half_ang + 1e-6:
                        continue
                    u = (ang + half_ang) / (2.0 * half_ang) if half_ang > 1e-9 else 0.5
                    v = (ly + half_h) / (2.0 * half_h) if half_h > 1e-9 else 0.5
                return float(u), float(v), float(t_hit)
            return None

        # Flat-screen (plane) fallback
        cp  = math.cos(self.screen_pitch); sp  = math.sin(self.screen_pitch)
        cy  = math.cos(self.screen_yaw);   sy_ = math.sin(self.screen_yaw)
        screen_n   = np.array([cp * sy_, -sp, cp * cy], dtype='f8')
        screen_pos = self._screen_world_pos(cy, sy_, cp, sp)
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
        if abs(loc_x) <= safe_w / 2.0 and abs(loc_y) <= safe_h / 2.0:
            u = 0.5 + loc_x / safe_w
            v = 0.5 + loc_y / safe_h
            return u, v, t
        return None

    def _laser_plane_uv(self, ctrl_pos, fwd_w):
        """Return (u, v) where the aim ray hits the infinite plane of the screen, ignoring bounds."""
        sh = self.screen_height
        if sh is None:
            fw, fh = self.frame_size
            sh = self.screen_width * (self._effective_frame_aspect() if fw > 0 else 9.0 / 16.0)
        safe_w = max(self.screen_width, 1e-6)
        safe_h = max(sh, 1e-6)
        cp = math.cos(self.screen_pitch); sp = math.sin(self.screen_pitch)
        cy = math.cos(self.screen_yaw); sy_ = math.sin(self.screen_yaw)
        screen_n = np.array([cp * sy_, -sp, cp * cy], dtype='f8')
        screen_pos = self._screen_world_pos(cy, sy_, cp, sp)
        denom = float(np.dot(screen_n, fwd_w))
        if abs(denom) < 1e-6:
            return None
        t = float(np.dot(screen_n, screen_pos - ctrl_pos)) / denom
        if t < 0.01:
            return None
        hit = ctrl_pos + fwd_w * t
        diff = hit - screen_pos
        r_ax = np.array([cy, 0.0, -sy_], dtype='f8')
        u_ax = np.array([sp * sy_, cp, sp * cy], dtype='f8')
        loc_x = float(np.dot(diff, r_ax))
        loc_y = float(np.dot(diff, u_ax))
        u = 0.5 + loc_x / safe_w
        v = 0.5 + loc_y / safe_h
        return u, v

    def _keyboard_laser_hit(self, ctrl_pos, fwd_w):
        """Return (key_index, t) if the aim ray hits a key on the virtual keyboard, else (None, None).

        Accounts for the keyboard's tilt (`_keyboard_pitch`) and yaw the plane is no
        longer axis-aligned, so we project the world hit point back into the keyboard's
        local 2D frame using the rotated basis vectors before testing rect_local bounds.
        """
        if not self._keyboard_keys:
            return None, None
        cp = math.cos(self._keyboard_pitch); sp = math.sin(self._keyboard_pitch)
        cy = math.cos(self._keyboard_yaw);   sy = math.sin(self._keyboard_yaw)
        # Local axes in world (columns of rot_y(yaw) ∘ rot_x(pitch)):
        #   X_local ( cy,        0,    -sy)
        #   Y_local ( sy*sp,    cp,     cy*sp)
        #   Z_local ( sy*cp,   -sp,     cy*cp)   surface normal
        kb_x = np.array([ cy,          0.0,    -sy        ], dtype='f8')
        kb_y = np.array([ sy * sp,     cp,      cy * sp   ], dtype='f8')
        kb_n = np.array([ sy * cp,    -sp,      cy * cp   ], dtype='f8')
        kb_pos = np.array([self._keyboard_pan_x,
                        self._keyboard_pan_y,
                        -self._keyboard_distance], dtype='f8')
        denom = float(np.dot(kb_n, fwd_w))
        if abs(denom) < 1e-6:
            return None, None
        t = float(np.dot(kb_n, kb_pos - ctrl_pos)) / denom
        if t < 0.05:
            return None, None
        hit  = ctrl_pos + fwd_w * t
        diff = hit - kb_pos
        lx = float(np.dot(diff, kb_x))
        ly = float(np.dot(diff, kb_y))
        for i, key in enumerate(self._keyboard_keys):
            x0, y0, x1, y1 = key.rect_local
            if x0 <= lx <= x1 and y0 <= ly <= y1:
                return i, t
        return None, None

    def _handle_cursor(self):
        """Move the Windows mouse cursor when a controller laser is pointing at the screen.

        Cursor jitter at long laser distances comes from natural hand tremor: a 0.5°
        wrist wobble at 2 m laser length is ~17 mm at the screen, which is hundreds
        of cursor pixels. We low-pass-filter the UV with an exponential moving average
        to take the high-frequency edge off without adding perceptible lag.

        Controller cursor control is only active when the laser actually intersects the
        screen quad. When no laser hits the screen, the physical mouse has full control.
        """
        PHYS_TIMEOUT = 3.0

        def _beam_origin_dir(aim_mat, grip_mat, smooth_pos_attr, smooth_quat_attr):
            """Same logic as _laser_beam_setup for smoothing and edge constraints."""
            is_left = smooth_pos_attr.endswith('_l')
            cp, fw = self._get_smoothed_ray(is_left)
            if cp is None:
                if grip_mat is not None:
                    raw_pos = (grip_mat[:3, 3] + grip_mat[:3, 1] * 0.020).astype('f8')
                else:
                    raw_pos = aim_mat[:3, 3].astype('f8')
                fw = -aim_mat[:3, 2].astype('f8')
                cp = raw_pos + fw * 0.11
                return cp, fw
            # raw_pos for edge constraint (unsmoothed origin)
            if grip_mat is not None:
                raw_pos = (grip_mat[:3, 3] + grip_mat[:3, 1] * 0.020).astype('f8')
            else:
                raw_pos = aim_mat[:3, 3].astype('f8')
            right = aim_mat[:3, 0].astype('f8')
            ang = math.radians(12); ca, sa = math.cos(ang), math.sin(ang)
            k = right / (np.linalg.norm(right) + 1e-10)
            fw = fw * ca + np.cross(k, fw) * sa + k * np.dot(k, fw) * (1 - ca)
            # Keyboard targeting takes precedence over screen edge-snapping 
            # mirrors _laser_beam_setup so the screen cursor isn't deflected onto
            # the screen edge (and thus stolen away from the keyboard) when the
            # keyboard sits close below the screen.
            _kb_targeted = (self._keyboard_visible and
                            self._keyboard_laser_hit_dist(raw_pos, fw) < 30.0)
            # Screen edge constraint: if the smoothed ray misses the screen but the raw ray is close, clamp to the edge.
            if not _kb_targeted and self._laser_screen_hit_uv(raw_pos, fw) is None:
                _raw_fw = -aim_mat[:3, 2].astype('f8')
                _raw_rw = aim_mat[:3, 0].astype('f8')
                _raw_k = _raw_rw / (np.linalg.norm(_raw_rw) + 1e-10)
                _raw_fw = _raw_fw * ca + np.cross(_raw_k, _raw_fw) * sa + _raw_k * np.dot(_raw_k, _raw_fw) * (1 - ca)
                if self._laser_screen_hit_uv(raw_pos, _raw_fw) is None:
                    _plane_uv = self._laser_plane_uv(raw_pos, fw)
                    if _plane_uv is not None:
                        _cu = max(0.0, min(1.0, _plane_uv[0]))
                        _cv = max(0.0, min(1.0, _plane_uv[1]))
                        _clamped_wp = self._screen_uv_to_world(_cu, _cv)
                        _edge_dir = _clamped_wp - raw_pos
                        _norm = np.linalg.norm(_edge_dir)
                        if _norm > 1e-6:
                            _edge_dir /= _norm
                            _dot2 = np.dot(_raw_fw, _edge_dir)
                            _dot2 = max(-1.0, min(1.0, _dot2))
                            _ang2 = math.acos(_dot2)
                            if _ang2 < self._ray_edge_deadzone_rad:
                                fw = _edge_dir
            # Beam starts from raw grip position (matches _laser_beam_setup)
            cp = raw_pos + fw * 0.11
            return cp, fw

        # Physical mouse detection: if the physical mouse has moved recently,
        # suppress VR cursor control unconditionally so the user can use the
        # physical mouse without fighting the VR cursor.  VR resumes after a
        # quiet period with no physical mouse movement.
        # Check FIRST before expensive ray casting to skip work when mouse is active.
        if sys.platform == "win32":
            if (time.perf_counter() - self._phys_mouse_last_move) < PHYS_TIMEOUT:
                self._cursor_ctrl = None
                self._cursor_smooth_uv = None
                # Invalidate touch positions so any held contact is released by
                # _handle_triggers when physical mouse takes over.
                self._touch_valid_l = False
                self._touch_valid_r = False
                return
            # Throttle GetCursorPos to every ~50ms (3-4 frames at 72Hz) 
            # per-frame polling is wasteful; physical mouse detection doesn't need sub-frame precision.
            _now = time.perf_counter()
            if _now - getattr(self, '_last_get_cursor_pos_time', 0.0) >= 0.05:
                class _POINT(ctypes.Structure):
                    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
                _pt = _POINT()
                ctypes.windll.user32.GetCursorPos(ctypes.byref(_pt))
                self._last_get_cursor_pos_time = _now
                _cur_pos = (_pt.x, _pt.y)
                if self._phys_mouse_pos is not None and _cur_pos != self._phys_mouse_pos:
                    vcp = self._vr_cursor_screen_pos
                    if vcp is None or abs(_cur_pos[0] - vcp[0]) > 4 or abs(_cur_pos[1] - vcp[1]) > 4:
                        self._phys_mouse_last_move = _now
                self._phys_mouse_pos = _cur_pos
            if (time.perf_counter() - self._phys_mouse_last_move) < PHYS_TIMEOUT:
                self._cursor_ctrl = None
                self._cursor_smooth_uv = None
                self._touch_valid_l = False
                self._touch_valid_r = False
                return

        hit_l = hit_r = None
        ov_hit_l = ov_hit_r = False
        # Read both triggers every frame.  The keyboard typing-lock below only
        # needs them when the keyboard is visible, but the cursor-ownership rule
        # ("latest click / tap wins") needs them every frame to know which hand
        # most recently clicked so we always poll.
        ltrig_now = self._read_float_action(self._act_left_trigger,  "/user/hand/left")
        rtrig_now = self._read_float_action(self._act_right_trigger, "/user/hand/right")
        # Beam origin/direction per hand stashed for the touch-publish block
        # below so it can clamp the off-screen drag position to the screen
        # edge instead of letting `_touch_px_l/r` freeze at the last on-screen
        # pixel.
        cp_l = fw_l = cp_r = fw_r = None
        if self._aim_mat_l is not None:
            cp, fw = _beam_origin_dir(self._aim_mat_l, self._grip_mat_l,
                                    "_smooth_ray_origin_l", "_smooth_ray_quat_l")
            cp_l, fw_l = cp, fw
            # Compute both keyboard and screen hit distances, only interact with closer one
            _kb_idx_l, _kb_t_l = self._keyboard_laser_hit(cp, fw)
            hit_l = self._laser_screen_hit_uv(cp, fw)
            kb_dist_l = self._keyboard_laser_hit_dist(cp, fw)
            sc_dist_l = hit_l[2] if hit_l is not None else float('inf')
            # `_keyboard_laser_hit_dist` returns the BEAM_MAX sentinel (30 m)
            # when the laser misses the keyboard rectangle.  We must use this
            # sentinel not `< float('inf')` to test "laser is on keyboard",
            # otherwise typing_lock would falsely engage whenever the user
            # pulls the trigger on the screen with the keyboard merely
            # visible, blocking touch clicks on the screen entirely.
            _KB_BEAM_MAX = 30.0
            kb_actually_hit_l = kb_dist_l < _KB_BEAM_MAX
            # Keep keyboard priority stable when it is very near the screen
            # (e.g. top rows close to screen bottom) to avoid visual cursor dropouts.
            typing_lock_l = (
                self._keyboard_visible
                and kb_actually_hit_l
                and (
                    self._kb_held_key_l is not None
                    or (ltrig_now >= 0.55)
                )
            )
            if typing_lock_l or (kb_actually_hit_l and kb_dist_l <= (sc_dist_l + KB_CURSOR_PRIORITY_BIAS)):
                # Refresh the hold timer while the keyboard owns the cursor so the
                # post-release grace measures time since the LAST owned frame.
                self._kb_cursor_owned_t_l = time.perf_counter()
                hit_l = None  # keyboard is closer, suppress screen cursor
            elif (self._keyboard_visible and
                  (time.perf_counter() - self._kb_cursor_owned_t_l) < self._KB_RELEASE_GRACE):
                # Post-release grace: keep the screen cursor suppressed briefly while
                # the user lifts off the keyboard toward the screen (smooths the swap).
                hit_l = None
            else:
                self._kb_hover_l = None  # screen is closer, suppress keyboard hover
            ov_cp_l, ov_fw_l = self._pre_snap_overlay_ray(True, self._aim_mat_l, self._grip_mat_l)
            ov_dist_l = self._overlay_panel_hit_dist(ov_cp_l, ov_fw_l)
            ov_hit_l = ov_dist_l < 5.0
            if ov_hit_l and hit_l is not None and ov_dist_l < hit_l[2]:
                hit_l = None  # suppress cursor when overlay is closer than screen
        if self._aim_mat_r is not None:
            cp, fw = _beam_origin_dir(self._aim_mat_r, self._grip_mat_r,
                                    "_smooth_ray_origin_r", "_smooth_ray_quat_r")
            cp_r, fw_r = cp, fw
            _kb_idx_r, _kb_t_r = self._keyboard_laser_hit(cp, fw)
            hit_r = self._laser_screen_hit_uv(cp, fw)
            kb_dist_r = self._keyboard_laser_hit_dist(cp, fw)
            sc_dist_r = hit_r[2] if hit_r is not None else float('inf')
            _KB_BEAM_MAX = 30.0
            kb_actually_hit_r = kb_dist_r < _KB_BEAM_MAX
            typing_lock_r = (
                self._keyboard_visible
                and kb_actually_hit_r
                and (
                    self._kb_held_key_r is not None
                    or (rtrig_now >= 0.55)
                )
            )
            if typing_lock_r or (kb_actually_hit_r and kb_dist_r <= (sc_dist_r + KB_CURSOR_PRIORITY_BIAS)):
                # Refresh the hold timer while the keyboard owns the cursor so the
                # post-release grace measures time since the LAST owned frame.
                self._kb_cursor_owned_t_r = time.perf_counter()
                hit_r = None  # keyboard is closer, suppress screen cursor
            elif (self._keyboard_visible and
                  (time.perf_counter() - self._kb_cursor_owned_t_r) < self._KB_RELEASE_GRACE):
                # Post-release grace: keep the screen cursor suppressed briefly while
                # the user lifts off the keyboard toward the screen (smooths the swap).
                hit_r = None
            else:
                self._kb_hover_r = None  # screen is closer, suppress keyboard hover
            ov_cp_r, ov_fw_r = self._pre_snap_overlay_ray(False, self._aim_mat_r, self._grip_mat_r)
            ov_dist_r = self._overlay_panel_hit_dist(ov_cp_r, ov_fw_r)
            ov_hit_r = ov_dist_r < 5.0
            if ov_hit_r and hit_r is not None and ov_dist_r < hit_r[2]:
                hit_r = None  # suppress cursor when overlay is closer than screen

        self._cursor_uv_l = hit_l if hit_l else None   # (u, v, t) or None
        self._cursor_uv_r = hit_r if hit_r else None   # (u, v, t) or None
        self._overlay_hit_l = ov_hit_l
        self._overlay_hit_r = ov_hit_r
        self._ray_prev_uv_l = self._cursor_uv_l
        self._ray_prev_uv_r = self._cursor_uv_r

        # Publish per-hand desktop pixel positions for the multi-touch injector.
        # Each hand is tracked independently so two simultaneous triggers become
        # two simultaneous touch contacts (Windows multi-touch pinch/zoom,
        # two-finger pan, press-and-hold right-click, etc.).
        #
        # When the laser is OFF-SCREEN we project the ray onto the screen plane
        # and clamp the UV to [0, 1] so a drag-in-progress can keep updating at
        # the screen edge in the direction the laser is pointing.  Without this
        # clamp, `_touch_px_l/r` froze at the last on-screen pixel fast drags
        # that briefly grazed off-screen left the touch contact stuck at the
        # edge with the cursor visibly lagging behind the laser, then jumping
        # when the laser returned.  Edge-clamping makes the drag continue at
        # the edge so the OS sees uninterrupted motion data.  `valid` is still
        # False off-screen so no NEW touch can fire there only an active
        # drag uses the clamped position.
        try:
            mon_left, mon_top, mon_w, mon_h = self._get_target_monitor_rect()
        except Exception:
            mon_left = mon_top = 0; mon_w = mon_h = 0
        def _uv_to_px(uv):
            if uv is None or mon_w <= 0 or mon_h <= 0:
                return None
            u, v = float(uv[0]), float(uv[1])
            self._movie_crop_note_cursor_uv(u, v)
            src_u, src_top_v = self._screen_uv_to_source_top_uv(u, v)
            return (mon_left + int(src_u * mon_w),
                    mon_top + int(src_top_v * mon_h))
        def _edge_px(cp, fw):
            """Project ray onto screen plane, clamp UV to [0,1], return pixels."""
            if cp is None or fw is None or mon_w <= 0 or mon_h <= 0:
                return None
            uv = self._laser_plane_uv(cp, fw)
            if uv is None:
                return None
            u = max(0.0, min(1.0, float(uv[0])))
            v = max(0.0, min(1.0, float(uv[1])))
            src_u, src_top_v = self._screen_uv_to_source_top_uv(u, v)
            return (mon_left + int(src_u * mon_w),
                    mon_top + int(src_top_v * mon_h))
        _pl = _uv_to_px(hit_l)
        _pr = _uv_to_px(hit_r)
        if _pl is not None:
            self._touch_px_l = _pl
            self._touch_valid_l = True
        else:
            # Off-screen / keyboard-claimed / overlay-claimed: clamp to edge so
            # an active drag stays alive; mark invalid so no new DOWN fires.
            edge_l = _edge_px(cp_l, fw_l)
            if edge_l is not None:
                self._touch_px_l = edge_l
            self._touch_valid_l = False
        if _pr is not None:
            self._touch_px_r = _pr
            self._touch_valid_r = True
        else:
            edge_r = _edge_px(cp_r, fw_r)
            if edge_r is not None:
                self._touch_px_r = edge_r
            self._touch_valid_r = False

        # ▶▶Pick the active cursor controller: LATEST CLICK / TAP WINS ▶▶
        # When BOTH lasers are on the screen we give the cursor to whichever
        # controller most recently pulled its trigger (a click / tap), so the
        # user can hand control back and forth between hands just by clicking.
        # The old rule was "right always wins", which made the off-hand unable
        # to ever take over while the right laser merely grazed the screen.
        #
        # Rising-edge stamp: record the click time the instant the trigger
        # crosses PRESS while THAT hand's laser is on the screen.  Gating on
        # `hit_*` means a press on the keyboard / overlay (where `hit_*` is
        # already None) never steals the screen cursor.  Pure hover never
        # changes the stamps, so two resting lasers never ping-pong the last
        # hand to click keeps control until the other hand clicks.
        now_pc = time.perf_counter()
        _CURSOR_PRESS = 0.55
        if hit_l and ltrig_now >= _CURSOR_PRESS and self._cursor_trig_prev_l < _CURSOR_PRESS:
            self._cursor_click_ts_l = now_pc
        if hit_r and rtrig_now >= _CURSOR_PRESS and self._cursor_trig_prev_r < _CURSOR_PRESS:
            self._cursor_click_ts_r = now_pc
        self._cursor_trig_prev_l = ltrig_now
        self._cursor_trig_prev_r = rtrig_now

        if hit_l and hit_r:
            # Both lasers on screen newer click owns the cursor.
            if self._cursor_click_ts_r > self._cursor_click_ts_l:
                ctrl = 'right'
            elif self._cursor_click_ts_l > self._cursor_click_ts_r:
                ctrl = 'left'
            else:
                # Neither has clicked yet (equal stamps, e.g. both 0.0) keep
                # the current owner so the cursor doesn't ping-pong; default to
                # right on the very first frame.
                ctrl = self._cursor_ctrl if self._cursor_ctrl in ('left', 'right') else 'right'
        elif hit_r:
            ctrl = 'right'
        elif hit_l:
            ctrl = 'left'
        else:
            self._cursor_ctrl = None
            self._cursor_smooth_uv = None   # reset so next entry doesn't drag
            return

        # On an ownership swap, drop the stale smoothing anchor so the new hand's
        # cursor starts exactly where ITS laser points instead of sliding across
        # the screen from the previous owner's last position that slide was the
        # "lag" the user saw when control changed hands.
        if ctrl != self._cursor_ctrl:
            self._cursor_smooth_uv = None

        self._cursor_ctrl = ctrl
        u, v = (hit_r[0], hit_r[1]) if ctrl == 'right' else (hit_l[0], hit_l[1])

        # IMPORTANT no second smoothing stage here.
        # The controller ray is already low-pass filtered upstream by the
        # One-Euro filter in `_get_smoothed_ray`.  A second EMA at this layer
        # (the old `ALPHA = 0.35` cursor smoother) stacked latency on top of
        # that and was a root cause of the drag problems the user reported:
        #   fast drags moved the window only a short distance (the smoother
        #     never caught up before release),
        #   the window/cursor visibly trailed behind the beam,
        #   the cursor kept gliding after the trigger was released (the EMA
        #     was still converging to the final position), and
        #   drag felt "sticky".
        # It also disagreed with the touch-contact position (published above
        # from the RAW `hit_*` UV), so Windows which pins the cursor to the
        # active touch contact fought this SetCursorPos call.
        # Use the raw UV directly: what the beam points at is where the cursor
        # (and the touch contact) goes, with zero added lag.  This is also a
        # few ops cheaper per frame, so it's FPS-neutral-to-positive.
        self._cursor_smooth_uv = (u, v)

        mon_left, mon_top, mon_w, mon_h = self._get_target_monitor_rect()
        self._movie_crop_note_cursor_uv(u, v)
        src_u, src_top_v = self._screen_uv_to_source_top_uv(u, v)
        px = mon_left + int(src_u * mon_w)
        py = mon_top + int(src_top_v * mon_h)
        # Always track the VR cursor position so the physical-mouse detector
        # doesn't falsely fire when grip ends and the cursor resumes moving.
        self._vr_cursor_screen_pos = (px, py)
        # Suppress cursor movement while gripping or during a two-finger
        # touch gesture.  When both touch contacts are active, SetCursorPos
        # fights the multi-touch subsystem and breaks pinch/zoom/pan because
        # Windows pins the cursor to the active contact moving it to the
        # "winning" hand's position mid-gesture collapses the two-contact
        # interaction into a single-point one.
        both_touch_down = (self._touch_state_l == 'down'
                           and self._touch_state_r == 'down')
        if not self._grabbed and not both_touch_down:
            _set_cursor_pos(px, py)

    def _handle_triggers(self):
        """Map controller triggers to Windows multi-touch contacts (preferred)
        or mouse clicks (fallback).

        Per-hand contact lifecycle when ``_TOUCH_AVAILABLE``:

        * Trigger ≥PRESS_THRESH on a valid screen target touch DOWN.
        * Trigger held touch UPDATE every frame (drives drag, incl. window
          title-bar drag Windows treats it as one continuous interaction).
        * Trigger < RELEASE_THRESH (or laser leaves a usable target) touch UP.

        Both controllers active simultaneously become two-contact multi-touch,
        enabling the gestures documented at
        https://support.microsoft.com/en-us/windows/touch-gestures-for-windows-a9d28305-4818-a5df-4e2b-e5590f850741
        (tap = click, drag = drag, two-finger pan/zoom, press-and-hold =
        right-click, edge swipes for notification center / widgets, etc.).

        If a trigger fires while the laser hits the FPS/status panel, that
        trigger toggles the shortcuts/help panel instead of generating a click.
        """
        PRESS_THRESH   = 0.55   # rising edge
        RELEASE_THRESH = 0.30   # falling edge (hysteresis)
        HOLD_TIME      = 0.22   # seconds trigger must stay held to enter drag mode (mouse fallback only)

        # While gripping (user repositioning the screen/keyboard), release any
        # active touch contacts cleanly so the next press starts fresh and
        # skip all further click processing for this frame.
        if self._grabbed:
            if _TOUCH_AVAILABLE and _touch_injector is not None:
                if self._touch_state_l == 'down':
                    _touch_injector.set(_TOUCH_CONTACT_ID_LEFT,
                                        self._touch_px_l[0], self._touch_px_l[1],
                                        want_down=False)
                    self._touch_state_l = 'idle'
                    self._touch_smooth_l = None
                if self._touch_state_r == 'down':
                    _touch_injector.set(_TOUCH_CONTACT_ID_RIGHT,
                                        self._touch_px_r[0], self._touch_px_r[1],
                                        want_down=False)
                    self._touch_state_r = 'idle'
                    self._touch_smooth_r = None
                _touch_injector.flush()
            # Seed the per-hand prior-trigger trackers to the current readings so
            # that on the frame the user releases the grip, the rising-edge gate
            # in the touch path still requires a true release-then-press before
            # firing a new touch DOWN (avoids "drop the grip instant phantom
            # click" if the trigger happens to be high at grip-release time).
            self._touch_trig_prev_l = self._read_float_action(
                self._act_left_trigger,  "/user/hand/left")
            self._touch_trig_prev_r = self._read_float_action(
                self._act_right_trigger, "/user/hand/right")
            return

        now = self._frame_now
        lt  = self._read_float_action(self._act_left_trigger,  "/user/hand/left")
        rt  = self._read_float_action(self._act_right_trigger, "/user/hand/right")

        # Trigger-click on FPS/status panel toggles shortcut help panel.
        # Uses _overlay_hit_l/r (same hit test as cursor suppression) so the
        # toggle and click-block are consistent with what the user sees.
        ov_claim_l = False
        ov_claim_r = False

        if self._fps_overlay_visible:
            for hit_overlay, trig_now, is_left in (
                (self._overlay_hit_l, lt, True),
                (self._overlay_hit_r, rt, False),
            ):
                trig_prev_held = self._ov_ltrig_held if is_left else self._ov_rtrig_held

                if hit_overlay and trig_now >= PRESS_THRESH and not trig_prev_held:
                    self._help_panel_visible = not self._help_panel_visible

                if is_left:
                    # Only track held state while the laser is on the overlay;
                    # reset when the laser leaves so the next press fires cleanly.
                    self._ov_ltrig_held = hit_overlay and trig_now >= PRESS_THRESH
                    if hit_overlay:
                        ov_claim_l = True
                else:
                    self._ov_rtrig_held = hit_overlay and trig_now >= PRESS_THRESH
                    if hit_overlay:
                        ov_claim_r = True

        left_on_kb  = self._kb_hover_l is not None
        right_on_kb = self._kb_hover_r is not None
        # Treat the keyboard as "claiming" the hand whenever a virtual key is
        # currently being typed by that hand, even if the laser drifted off the
        # key this frame.  Without this, releasing a key by sliding the laser
        # off it (while the trigger is still held) could leak into a touch DOWN
        # on the screen behind/below the keyboard.
        left_kb_typing  = self._kb_held_key_l is not None
        right_kb_typing = self._kb_held_key_r is not None

        # ▶Touch path (preferred): clean DOWN/UPDATE/UP per hand ▶        # Reliable click: a single DOWN-UP pair is unambiguous to Windows,
        # unlike the mouse-flag pulse that occasionally raced with cursor
        # positioning and produced missed clicks.
        if _TOUCH_AVAILABLE and _touch_injector is not None:
            touch_updates = []
            for (trig, valid, px_attr, smooth_attr, state_attr, trig_prev_attr,
                 contact_id, ov_claim, on_kb, kb_typing) in (
                (lt, self._touch_valid_l, '_touch_px_l', '_touch_smooth_l',
                 '_touch_state_l', '_touch_trig_prev_l',
                 _TOUCH_CONTACT_ID_LEFT, ov_claim_l, left_on_kb, left_kb_typing),
                (rt, self._touch_valid_r, '_touch_px_r', '_touch_smooth_r',
                 '_touch_state_r', '_touch_trig_prev_r',
                 _TOUCH_CONTACT_ID_RIGHT, ov_claim_r, right_on_kb, right_kb_typing),
            ):
                state    = getattr(self, state_attr)
                raw_px   = getattr(self, px_attr)
                trig_prev = getattr(self, trig_prev_attr)
                # Keyboard-claimed hand: no touch this frame, period.  Covers
                # both "laser on a key" and "still holding a typed key" cases.
                kb_claim = on_kb or kb_typing

                # Determine the desired contact state this frame.
                if state == 'down':
                    # Stay down until trigger releases OR the hand loses its
                    # screen target (laser left screen / moved to keyboard /
                    # claimed by overlay).  We use the looser RELEASE_THRESH so
                    # the contact survives small trigger dips during drag.
                    want_down = (trig > RELEASE_THRESH
                                 and not ov_claim
                                 and not kb_claim)
                else:
                    # idle only fire on a TRUE rising edge against a valid
                    # screen target.  Requiring the prior frame's trigger to be
                    # below PRESS_THRESH prevents a phantom click when the user:
                    #   * slides the laser off the virtual keyboard onto the
                    #     screen while still holding the trigger, or
                    #   * toggles the keyboard off mid-press.
                    # In both cases the trigger never released, so no new touch
                    # should fire on the screen until the user lets go.
                    want_down = (trig >= PRESS_THRESH
                                 and trig_prev < PRESS_THRESH
                                 and valid
                                 and not ov_claim
                                 and not kb_claim)

                touch_updates.append({
                    'want_down': want_down,
                    'state': state,
                    'raw_px': raw_px,
                    'smooth_attr': smooth_attr,
                    'state_attr': state_attr,
                    'trig_prev_attr': trig_prev_attr,
                    'contact_id': contact_id,
                    'trig': trig,
                })

            if (len(touch_updates) == 2
                    and touch_updates[0]['want_down']
                    and touch_updates[1]['want_down']
                    and _TOUCH_PINCH_SPREAD_GAIN > 1.0):
                try:
                    mon_left, mon_top, mon_w, mon_h = self._get_target_monitor_rect()
                except Exception:
                    mon_left = mon_top = mon_w = mon_h = 0

                def _clamp_px(x, y):
                    if mon_w > 0 and mon_h > 0:
                        x = max(mon_left, min(mon_left + mon_w - 1, x))
                        y = max(mon_top, min(mon_top + mon_h - 1, y))
                    return int(round(x)), int(round(y))

                p0 = touch_updates[0]['raw_px']
                p1 = touch_updates[1]['raw_px']
                cx = (float(p0[0]) + float(p1[0])) * 0.5
                cy = (float(p0[1]) + float(p1[1])) * 0.5
                gain = float(_TOUCH_PINCH_SPREAD_GAIN)
                touch_updates[0]['raw_px'] = _clamp_px(
                    cx + (float(p0[0]) - cx) * gain,
                    cy + (float(p0[1]) - cy) * gain,
                )
                touch_updates[1]['raw_px'] = _clamp_px(
                    cx + (float(p1[0]) - cx) * gain,
                    cy + (float(p1[1]) - cy) * gain,
                )

            for upd in touch_updates:
                want_down = upd['want_down']
                state = upd['state']
                raw_px = upd['raw_px']
                smooth_attr = upd['smooth_attr']
                state_attr = upd['state_attr']
                trig_prev_attr = upd['trig_prev_attr']
                contact_id = upd['contact_id']
                trig = upd['trig']

                if want_down:
                    # Send touch DOWN/UPDATE at the *current* laser-mapped
                    # pixel position no EMA, no snap-distance teleport.
                    #
                    # The controller pose is already smoothed upstream by
                    # `_get_smoothed_ray`, so any extra EMA here is pure
                    # latency: an α=0.45 second-stage EMA caused
                    #   a visible cursor jump after release (UP fired at
                    #     the lagging smoothed position; the cursor then
                    #     teleported to the actual laser position on the
                    #     next frame), and
                    #   a "fast drag = tiny window move" symptom
                    #     (smoother was catching up over ~10 frames so the
                    #     OS only saw ~70% of the user's drag distance), and
                    #   a "cursor hang" symptom (Windows pins the cursor
                    #     to the touch contact, which was crawling along the
                    #     EMA tail far behind the laser).
                    # Going direct-to-raw also drops ~14 ops/frame/hand, so
                    # if anything this is FPS-positive.
                    _touch_injector.set(contact_id, raw_px[0], raw_px[1],
                                        want_down=True)
                    setattr(self, smooth_attr, raw_px)  # last sent pos
                    setattr(self, state_attr, 'down')
                else:
                    if state == 'down':
                        # Release at the *current* laser position.  The
                        # injector promotes a moving-UP into UPDATE-then-UP
                        # internally so the OS still sees a clean transition.
                        # Releasing at the laser position (rather than at
                        # the last-sent position) also means there is no
                        # post-release cursor snap: the touch UP and the
                        # subsequent `_set_cursor_pos` call from
                        # `_handle_cursor` agree on where the cursor ends up.
                        _touch_injector.set(contact_id, raw_px[0], raw_px[1],
                                            want_down=False)
                    setattr(self, smooth_attr, None)
                    setattr(self, state_attr, 'idle')

                # Remember this frame's trigger value so the next frame can
                # detect a true rising edge.
                setattr(self, trig_prev_attr, trig)

            _touch_injector.flush()
            # If touch turned out to be unavailable (mid-session injection
            # failure), fall through to the mouse path so clicks still work.
            if _touch_injector.available:
                return

        # ▶Mouse fallback: original click+drag state machine ▶
        left_laser_usable  = (not left_on_kb and not ov_claim_l and
                            (self._cursor_uv_l is not None or
                            self._cursor_ctrl == 'left'))
        right_laser_usable = (not right_on_kb and not ov_claim_r and
                            (self._cursor_uv_r is not None or
                            self._cursor_ctrl == 'right'))

        any_drag = False

        for trig, usable, state_attr, press_t_attr in (
            (lt, left_laser_usable,  '_ltrig_state', '_ltrig_press_t'),
            (rt, right_laser_usable, '_rtrig_state', '_rtrig_press_t'),
        ):
            state = getattr(self, state_attr)

            if state == 'idle':
                if trig >= PRESS_THRESH and usable:
                    # Rising edge: fire an immediate click pulse so the OS can
                    # accumulate it toward double-click detection, then start timer.
                    _send_mouse_flags(_MOUSEEVENTF_LEFTDOWN)
                    _send_mouse_flags(_MOUSEEVENTF_LEFTUP)
                    setattr(self, state_attr,   'pressed')
                    setattr(self, press_t_attr, now)

            elif state == 'pressed':
                if not usable or trig <= RELEASE_THRESH:
                    # Released quickly click already delivered, return to idle.
                    setattr(self, state_attr, 'idle')
                elif (now - getattr(self, press_t_attr)) >= HOLD_TIME:
                    # Held long enough begin drag (send LEFTDOWN if not already down).
                    if not self._left_btn_down:
                        _send_mouse_flags(_MOUSEEVENTF_LEFTDOWN)
                        self._left_btn_down = True
                    setattr(self, state_attr, 'dragging')
                    any_drag = True

            elif state == 'dragging':
                if not usable or trig <= RELEASE_THRESH:
                    setattr(self, state_attr, 'idle')
                else:
                    any_drag = True

        # Send LEFTUP once both triggers leave drag state.
        if not any_drag and self._left_btn_down:
            _send_mouse_flags(_MOUSEEVENTF_LEFTUP)
            self._left_btn_down = False

    def _press_key(self, key, key_idx, held_key_attr, held_mods_attr):
        """Press and hold a regular key on the virtual keyboard (key-down only)."""
        kbd = ctypes.windll.user32.keybd_event
        VK_SHIFT = 0x10; VK_CTRL = 0x11; VK_ALT = 0x12; VK_WIN = 0x5B
        sh = self._mod_state['shift']
        ct = self._mod_state['ctrl']
        al = self._mod_state['alt']
        wn = self._mod_state['win']
        shift_on = sh[0] or sh[1]
        ctrl_on  = ct[0] or ct[1]
        alt_on   = al[0] or al[1]
        win_on   = wn[0] or wn[1]
        use_shift = shift_on ^ self._caps_lock
        vk_to_use = key.shifted_vk if use_shift else key.vk
        need_shift = use_shift and vk_to_use == key.vk
        if ctrl_on:     kbd(VK_CTRL,  0, 0, 0)
        if need_shift:  kbd(VK_SHIFT, 0, 0, 0)
        if alt_on:      kbd(VK_ALT,   0, 0, 0)
        if win_on:      kbd(VK_WIN,   0, 0, 0)
        kbd(vk_to_use, 0, 0, 0)
        setattr(self, held_key_attr, key_idx)
        setattr(self, held_mods_attr, (need_shift, ctrl_on, alt_on, win_on, vk_to_use))

    def _handle_keyboard_input(self):
        """Send Windows keystrokes when a controller trigger fires on a keyboard key.

        Regular keys use press-and-hold: key-down on trigger pull, key-up on release.
        Modifier keys (Shift/Ctrl/Alt/Win) use tap/lock toggles.  Caps toggles caps-lock.
        """
        if not self._keyboard_visible:
            self._kb_hover_l = None
            self._kb_hover_r = None
            return
        CLICK_THRESH  = 0.7
        RELEASE_THRESH = 0.3
        VK_SHIFT      = 0x10
        VK_CAPS       = 0x14
        VK_CTRL       = 0x11
        VK_ALT        = 0x12
        VK_WIN        = 0x5B
        kbd           = ctypes.windll.user32.keybd_event

        # Suppress keystrokes while any grip is held the user is repositioning
        # the keyboard, not typing.  Still update hover for the laser cursor /
        # grip-to-move logic, but force trigger inputs to "released" so no
        # rising-edge fires and any held key is released cleanly.
        gripping = bool(self._grip_l_now or self._grip_r_now)

        if gripping:
            lt = 0.0
            rt = 0.0
        else:
            lt = self._read_float_action(self._act_left_trigger,  "/user/hand/left")
            rt = self._read_float_action(self._act_right_trigger, "/user/hand/right")

        for trig_now, trig_prev_attr, hover_attr, held_key_attr, held_mods_attr, aim_mat in [
            (lt, '_kb_trig_prev_l', '_kb_hover_l', '_kb_held_key_l', '_kb_held_mods_l', self._aim_mat_l),
            (rt, '_kb_trig_prev_r', '_kb_hover_r', '_kb_held_key_r', '_kb_held_mods_r', self._aim_mat_r),
        ]:
            trig_prev = getattr(self, trig_prev_attr)
            held_key  = getattr(self, held_key_attr)
            held_mods = getattr(self, held_mods_attr)
            if aim_mat is not None:
                # Calculate the laser beam's origin point and forward direction in world space
                grip_mat = self._grip_mat_l if aim_mat is self._aim_mat_l else self._grip_mat_r
                fw = -aim_mat[:3, 2].astype('f8')
                right = aim_mat[:3, 0].astype('f8')
                _ang = math.radians(12); _ca, _sa = math.cos(_ang), math.sin(_ang)
                _k = right / (np.linalg.norm(right) + 1e-10)
                fw = fw * _ca + np.cross(_k, fw) * _sa + _k * np.dot(_k, fw) * (1 - _ca)
                if grip_mat is not None:
                    cp = (grip_mat[:3, 3] + grip_mat[:3, 1] * 0.020).astype('f8')
                else:
                    cp = aim_mat[:3, 3].astype('f8')
                cp = cp + fw * 0.11
                idx, kb_t = self._keyboard_laser_hit(cp, fw)
                # Only interact with keyboard if it's closer than the screen
                if idx is not None:
                    sc_t = self._laser_screen_hit_dist(cp, fw)
                    if sc_t is not None and sc_t < kb_t:
                        idx = None  # screen is closer, suppress keyboard interaction
            else:
                idx = None
            setattr(self, hover_attr, idx)

            # ▶Release held regular key when trigger drops or laser leaves the key ▶
            if held_key is not None:
                release = False
                if trig_now < RELEASE_THRESH:
                    release = True
                elif idx != held_key:
                    release = True
                if release:
                    shift_dn, ctrl_dn, alt_dn, win_dn, vk_held = held_mods
                    kbd(vk_held, 0, _KEYEVENTF_KEYUP, 0)
                    if win_dn:   kbd(VK_WIN,   0, _KEYEVENTF_KEYUP, 0)
                    if alt_dn:   kbd(VK_ALT,   0, _KEYEVENTF_KEYUP, 0)
                    if shift_dn: kbd(VK_SHIFT, 0, _KEYEVENTF_KEYUP, 0)
                    if ctrl_dn:  kbd(VK_CTRL,  0, _KEYEVENTF_KEYUP, 0)
                    # Auto-release one-shot modifiers that were armed when the key went down
                    for name in ('shift', 'ctrl', 'alt', 'win'):
                        self._mod_state[name][0] = False
                    setattr(self, held_key_attr, None)
                    setattr(self, held_mods_attr, None)
                    held_key  = None
                    held_mods = None

            # ▶Rising edge: modifier / caps toggles, or start holding a regular key ▶
            if trig_now >= CLICK_THRESH and trig_prev < CLICK_THRESH and idx is not None:
                key = self._keyboard_keys[idx]
                mod_name = {VK_SHIFT: 'shift', VK_CTRL: 'ctrl',
                            VK_ALT: 'alt', VK_WIN: 'win'}.get(key.vk)
                if mod_name is not None:
                    DOUBLE_TAP_WINDOW = 0.4
                    now_t = time.monotonic()
                    state = self._mod_state[mod_name]
                    if state[1]:
                        state[0] = False
                        state[1] = False
                    elif state[0]:
                        state[0] = False
                        _send_key(key.vk)
                    elif (now_t - state[2]) < DOUBLE_TAP_WINDOW:
                        state[0] = False
                        state[1] = True
                    else:
                        state[0] = True
                    state[2] = now_t
                elif key.vk == VK_CAPS:
                    self._caps_lock = not self._caps_lock
                else:
                    self._press_key(key, idx, held_key_attr, held_mods_attr)

            # ▶Slide to new key: trigger already held, laser moved to another regular key ▶
            if (held_key is None and trig_now >= CLICK_THRESH
                    and idx is not None and trig_prev >= CLICK_THRESH):
                key = self._keyboard_keys[idx]
                if key.vk not in (VK_SHIFT, VK_CTRL, VK_ALT, VK_WIN, VK_CAPS):
                    self._press_key(key, idx, held_key_attr, held_mods_attr)

            setattr(self, trig_prev_attr, trig_now)

        # After processing triggers, check if shift/caps state changed and
        # rebuild the keyboard texture so labels update visually.
        sh = self._mod_state['shift']
        cur_shifted = bool(sh[0] or sh[1] or self._caps_lock)
        if cur_shifted != self._kb_show_shifted:
            self._kb_show_shifted = cur_shifted
            self._build_keyboard_texture()

    def _accum_scroll(self, x_axis, y_axis, dt):
        """Accumulate thumbstick deflection into accelerated mouse wheel events.

        Uses a cubic acceleration curve so small deflections are precise while
        full push is dramatically faster eliminates the \"stuck\" feeling.

        Fires WHEEL_DELTA-granular (120) scroll so every event is a full
        hardware notch that applications process reliably.
        """
        WHEEL_DELTA        = 100     # Windows: one wheel notch
        SCROLL_BASE_NOTCH  = 2.0     # notches/s just above dead zone
        SCROLL_MAX_NOTCH   = 35.0    # notches/s at full deflection
        ACCEL_EXPONENT     = 2.8     # >1 = soft centre, aggressive at edges

        for axis_val, accum_attr, send_fn in [
            (x_axis, '_scroll_accum_x', _send_hscroll),
            (y_axis, '_scroll_accum_y', _send_vscroll),
        ]:
            mag = abs(axis_val)
            if mag <= DEAD:
                continue
            # Normalise [DEAD .. 1.0] [0 .. 1]
            t = (mag - DEAD) / (1.0 - DEAD)
            # Cubic acceleration + base offset ensures fine control near centre
            speed = SCROLL_BASE_NOTCH + (SCROLL_MAX_NOTCH - SCROLL_BASE_NOTCH) * (t ** ACCEL_EXPONENT)
            accum = getattr(self, accum_attr) + float(axis_val) * speed * dt
            # Fire whole notches; keep leftover for next frame
            whole = int(accum)
            if whole:
                send_fn(whole * WHEEL_DELTA)
                accum -= whole
            setattr(self, accum_attr, accum)

    def _poll_controller_input(self, dt):
        """Controller interaction mapping:
        Left stick (no grip)       Mouse wheel
        Left grip + Left stick X/Y Screen pan horizontally/vertically (always parallel to ground)
        Left grip + Right stick X  Screen yaw rotation around its center
        Left grip + Right stick Y  Screen pitch tilt forward/backward
        Right grip + Left stick Y  Depth intensity
        Right stick (no grip)      Mouse wheel
        Right grip + Right stick X Screen width adjustment
        Right grip + Right stick Y Screen distance (with acceleration curve)
        Left grip + Left stick (when keyboard visible) Keyboard pan (preserved)
        Left stick press hold 1s    Toggle FPS/shortcut panel
        Left stick short press      Cycle environment
        Right stick press hold 1s   Reset screen direction (when no grip)
        Right stick short press     Cycle horizontal curve / vertical curve / flat
        Both sticks press 0.5s       Toggle FPS/help panel
        A/B/X/Y/Menu/Triggers      Original functions unchanged
        """
        if self._action_set is None:
            return

        self._update_trackpad_button_emu()

        def vec2(action, hand):
            try:
                path = (self._path_left
                        if hand == "/user/hand/left" else self._path_right)
                state = xr.get_action_state_vector2f(
                    self._xr_session,
                    xr.ActionStateGetInfo(action=action, subaction_path=path),
                )
                if state.is_active:
                    return state.current_state.x, state.current_state.y
            except Exception:
                pass
            return 0.0, 0.0

        lx, ly = vec2(self._act_left_stick,  "/user/hand/left")
        rx, ry = vec2(self._act_right_stick, "/user/hand/right")

        # Controller real-time calibration
        menu_now = self._read_bool_action(self._act_menu_btn, "/user/hand/left")
        a_now = self._read_bool_action(self._act_a_btn, "/user/hand/right") or self._emu_a
        b_now = self._read_bool_action(self._act_b_btn, "/user/hand/right") or self._emu_b

        # Calibration combo: Left Menu + Right A + Right B held for 1 second
        calib_combo = menu_now and a_now and b_now
        if calib_combo and not self._calib_combo_fired:
            if self._calib_combo_start == 0.0:
                self._calib_combo_start = time.perf_counter()
            elif time.perf_counter() - self._calib_combo_start >= 1.0:
                if self._calibration_mode:
                    self._exit_calibration_mode(save=False)
                else:
                    self._enter_calibration_mode()
                self._calib_combo_fired = True
                self._menu_long_fired = True  # suppress status-panel toggle on menu release
        if not calib_combo:
            self._calib_combo_start = 0.0
            self._calib_combo_fired = False

        # calibration mode: use sticks to adjust screen position/rotation with visual feedback, no cursor control
        if self._calibration_mode:
            CALIB_STEP = 0.005   # step size in meters for position adjustment
            ROT_STEP   = 0.5     # rotation step size in degrees
            if abs(ly) > 0.15:
                self._calibration_temp_offset[1] += ly * CALIB_STEP * 0.5
            if abs(ry) > 0.15:
                self._calibration_temp_offset[2] += ry * CALIB_STEP * 0.5
            if abs(rx) > 0.15:
                self._calibration_temp_rot += rx * ROT_STEP * 0.5
            # Button B to save and exit calibration mode on press (rising edge)
            b_edge = self._read_bool_edge(self._act_b_btn, "/user/hand/right", self._b_last)
            if not b_edge:
                b_edge = b_now and not self._b_last
            if b_edge:
                self._exit_calibration_mode(save=True)
            # calibration mode: disable normal stick operations
            lx, ly, rx, ry = 0.0, 0.0, 0.0, 0.0

        # Brand switching: Right A+B held for 1 second. Ensure ab_held is always defined for later A/B handling
        ab_held = False
        if not self._calibration_mode and self._available_brands:
            a_now2 = self._read_bool_action(self._act_a_btn, "/user/hand/right") or self._emu_a
            b_now2 = self._read_bool_action(self._act_b_btn, "/user/hand/right") or self._emu_b
            ab_held = a_now2 and b_now2
            if ab_held and not getattr(self, '_brand_sw_fired', False):
                _t = getattr(self, '_brand_sw_start', 0.0)
                if _t == 0.0:
                    self._brand_sw_start = time.perf_counter()
                elif time.perf_counter() - _t >= 0.5:
                    _idx = self._available_brands.index(self._current_brand) if self._current_brand in self._available_brands else 0
                    _next = self._available_brands[(_idx + 1) % len(self._available_brands)]
                    self._switch_brand(_next)
                    self._brand_sw_fired = True
            if not ab_held:
                self._brand_sw_start = 0.0
                self._brand_sw_fired = False

        grip_l = self._read_bool_action(self._act_left_grip,  "/user/hand/left")
        grip_r = self._read_bool_action(self._act_right_grip, "/user/hand/right")
        # Stash for _handle_keyboard_input used to suppress typing while
        # the user is repositioning the keyboard with a grip press.
        self._grip_l_now = grip_l
        self._grip_r_now = grip_r

        # Compute fresh cursor UVs + keyboard hits before grip-to-move
        # so laser-on-screen checks use current-frame data (no 1-frame lag).
        self._handle_keyboard_input()
        self._handle_cursor()
        laser_l_on_screen = self._cursor_uv_l is not None
        laser_r_on_screen = self._cursor_uv_r is not None
        laser_on_screen = laser_l_on_screen or laser_r_on_screen
        active = (grip_l or grip_r) and laser_on_screen and not self._environment_screen_locked()
        self._grabbed  = active
        self._resizing = False
        screen_locked = self._environment_screen_locked()
        locked_old_screen_mat = self._screen_pose_mat4() if screen_locked else None

        # Grip-to-move: screen follows laser beam origin 1:1, Skip while stick is actively used (outside deadzone) stick takes priority.
        stick_active_l = abs(lx) > DEAD or abs(ly) > DEAD
        stick_active_r = abs(rx) > DEAD or abs(ry) > DEAD
        # Laser-anchored grip dragging

        both_grips = grip_l and grip_r
        GRIP_LONG = 3.0
        if both_grips and not self._both_grips_last:
            self._both_grips_hold_t = time.perf_counter()
            self._both_grips_long_fired = False
        if both_grips and not self._both_grips_long_fired and screen_locked:
            if time.perf_counter() - self._both_grips_hold_t >= GRIP_LONG:
                self._both_grips_long_fired = True
                if self._seat_adjust_active:
                    self._exit_seat_adjust_mode(save=True)
                else:
                    self._enter_seat_adjust_mode()
        self._both_grips_last = both_grips
        seat_adjust_active = self._seat_adjust_active

        # Latch per-hand grip target on rising edge
        # "Only grip one item at a time": once a grip press locks onto the
        # screen or keyboard, it stays on that target until the grip is
        # released.  Decided on rising edge by what the laser is hitting:
        # keyboard takes priority if hovered, else screen.
        for grip_now, target_attr, kb_hover in [
            (grip_l, '_grip_target_l', self._kb_hover_l),
            (grip_r, '_grip_target_r', self._kb_hover_r),
        ]:
            if not grip_now:
                setattr(self, target_attr, None)
            elif getattr(self, target_attr) is None:
                if self._keyboard_visible and kb_hover is not None:
                    setattr(self, target_attr, 'keyboard')
                elif (target_attr == '_grip_target_l' and laser_l_on_screen) \
                        or (target_attr == '_grip_target_r' and laser_r_on_screen):
                    setattr(self, target_attr, 'screen')
                # else: laser on nothing leave unlatched; next frame may catch it

        # Per-controller laser requirement: grip only moves screen when
        # its corresponding laser hits the screen.
        for grip_now, aim_mat, grab_attr, local_attr, stick_active, laser_on, grip_target in [
            (grip_l, self._aim_mat_l,
            '_screen_grab_grip_l', '_screen_grab_local_l', stick_active_l,
             laser_l_on_screen, self._grip_target_l),

            (grip_r, self._aim_mat_r,
            '_screen_grab_grip_r', '_screen_grab_local_r', stick_active_r,
             laser_r_on_screen, self._grip_target_r),
        ]:
            if grip_now and not stick_active and not both_grips and laser_on \
                    and grip_target == 'screen' and not screen_locked:

                if aim_mat is None:
                    continue

                # Use pre-smoothed controller pose (shared with laser/cursor).
                # One Euro Filter applied once per frame no double-filtering.
                is_left = (grab_attr == '_screen_grab_grip_l')
                grip_mat = self._grip_mat_l if is_left else self._grip_mat_r
                # Raw grip position for responsive drag (no smoothing lag)
                if grip_mat is not None:
                    ray_origin = (grip_mat[:3, 3] + grip_mat[:3, 1] * 0.020).astype('f8')
                else:
                    ray_origin = aim_mat[:3, 3].astype(np.float64)
                # Smoothed forward direction for jitter-free pointing
                _, ray_dir = self._get_smoothed_ray(is_left)
                if ray_dir is None:
                    ray_dir = -aim_mat[:3, 2].astype(np.float64)
                ray_dir /= np.linalg.norm(ray_dir) + 1e-10

                # Screen orientation
                cp = math.cos(self.screen_pitch)
                sp = math.sin(self.screen_pitch)

                cy = math.cos(self.screen_yaw)
                sy = math.sin(self.screen_yaw)

                screen_normal = np.array([
                    cp * sy,
                    -sp,
                    cp * cy
                ], dtype=np.float64)

                screen_center = self._screen_world_pos(cy, sy, cp, sp).astype(np.float64)

                denom = np.dot(screen_normal, ray_dir)

                if abs(denom) < 1e-6:
                    continue

                t = np.dot(screen_normal, screen_center - ray_origin) / denom

                if t < 0.0:
                    continue

                hit_world = ray_origin + ray_dir * t

                # Screen local axes
                right_axis = np.array([cy, 0.0, -sy], dtype=np.float64)
                up_axis = np.array([sp * sy, cp, sp * cy], dtype=np.float64)

                local_x = np.dot(hit_world - screen_center, right_axis)
                local_y = np.dot(hit_world - screen_center, up_axis)

                saved_local = getattr(self, local_attr)

                # Initial grab
                if saved_local is None:

                    setattr(self, local_attr, np.array([local_x, local_y], dtype=np.float64))

                else:

                    target_local = saved_local

                    # Move screen center so laser continues hitting same local point
                    desired_center = (
                        hit_world
                        - right_axis * target_local[0]
                        - up_axis * target_local[1]
                    )

                    # Project onto sphere around head preserves 3D Euclidean
                    # distance so movement in any direction doesn't change
                    # apparent screen size.
                    if self._head_pos_w is not None:
                        hx, hy, hz = self._head_pos_w

                        # 3D vector from head to desired center
                        dx = desired_center[0] - hx
                        dy = desired_center[1] - hy
                        dz = desired_center[2] - hz

                        # Current 3D Euclidean distance from head to screen
                        csx = self.screen_pan_x - hx
                        csy = self.screen_pan_y - hy
                        csz = -self.screen_distance - hz
                        R3 = math.sqrt(csx * csx + csy * csy + csz * csz)

                        if R3 > 0.01:
                            d_len = math.sqrt(dx * dx + dy * dy + dz * dz)
                            if d_len > 0.001:
                                dx /= d_len
                                dy /= d_len
                                dz /= d_len

                            self.screen_pan_x    = float(hx + dx * R3)
                            self.screen_pan_y    = float(hy + dy * R3)
                            self.screen_distance = float(-(hz + dz * R3))

                            # Baseline orientation faces head; manual offset on top
                            base_yaw   = math.atan2(-dx, -dz)
                            base_pitch = math.asin(max(-1.0, min(1.0, dy)))
                            self.screen_yaw   = base_yaw   + self._yaw_offset
                            self.screen_pitch = base_pitch + self._pitch_offset

            elif both_grips and not grip_now:
                # One grip released while both were held clear grab state
                setattr(self, local_attr, None)
            elif not grip_now:
                # Grip released clear grab state so next press starts fresh
                setattr(self, local_attr, None)
            elif stick_active:
                # Stick fine-tuning is moving the screen old anchor is stale.
                # Clear it so grip-to-move re-acquires at the current pose
                # when the stick returns to neutral (avoids snap-back).
                setattr(self, local_attr, None)
            # else: grip held but laser transiently off screen keep anchor so
            # the grab resumes as soon as the laser re-enters the screen quad.

        # Keyboard grip-to-move: panel follows laser hit on head-sphere 
        # Mirrors the screen grip-to-move loop: laser stays anchored on the
        # same local key point while the panel slides along the sphere
        # centred on the head.  Preserves Euclidean head→panel distance so
        # apparent size doesn't change, and auto-orients the panel toward
        # the head with standalone yaw/pitch offsets applied on top.
        if self._keyboard_visible:
            for grip_now, aim_mat, kb_local_attr, stick_active, grip_target in [
                (grip_l, self._aim_mat_l, '_kb_grab_local_l',
                 stick_active_l, self._grip_target_l),
                (grip_r, self._aim_mat_r, '_kb_grab_local_r',
                 stick_active_r, self._grip_target_r),
            ]:
                # Don't require a per-key hover during drag: a fast-moving
                # laser often falls in gaps between keys, which would drop
                # the grip.  Once latched onto the keyboard, just keep
                # following the ray against the keyboard plane.
                if grip_now and not stick_active and not both_grips \
                        and grip_target == 'keyboard':
                    if aim_mat is None:
                        continue
                    is_left = (kb_local_attr == '_kb_grab_local_l')
                    grip_mat = self._grip_mat_l if is_left else self._grip_mat_r
                    if grip_mat is not None:
                        ray_origin = (grip_mat[:3, 3] + grip_mat[:3, 1] * 0.020).astype('f8')
                    else:
                        ray_origin = aim_mat[:3, 3].astype(np.float64)
                    _, ray_dir = self._get_smoothed_ray(is_left)
                    if ray_dir is None:
                        ray_dir = -aim_mat[:3, 2].astype(np.float64)
                    ray_dir /= np.linalg.norm(ray_dir) + 1e-10

                    # Keyboard plane axes (match _keyboard_laser_hit)
                    cp = math.cos(self._keyboard_pitch); sp = math.sin(self._keyboard_pitch)
                    cy = math.cos(self._keyboard_yaw);   sy = math.sin(self._keyboard_yaw)
                    kb_x_ax = np.array([cy,        0.0,  -sy      ], dtype=np.float64)
                    kb_y_ax = np.array([sy * sp,   cp,    cy * sp ], dtype=np.float64)
                    kb_n    = np.array([sy * cp,  -sp,    cy * cp ], dtype=np.float64)
                    kb_pos  = np.array([self._keyboard_pan_x,
                                        self._keyboard_pan_y,
                                        -self._keyboard_distance], dtype=np.float64)
                    denom = float(np.dot(kb_n, ray_dir))
                    if abs(denom) < 1e-6:
                        continue
                    t_hit = float(np.dot(kb_n, kb_pos - ray_origin)) / denom
                    if t_hit < 0.05:
                        continue
                    hit_world = ray_origin + ray_dir * t_hit
                    diff = hit_world - kb_pos
                    local_x = float(np.dot(diff, kb_x_ax))
                    local_y = float(np.dot(diff, kb_y_ax))

                    saved_local = getattr(self, kb_local_attr)
                    if saved_local is None:
                        # First frame of grab anchor the local hit point
                        setattr(self, kb_local_attr,
                                np.array([local_x, local_y], dtype=np.float64))
                    else:
                        target_local = saved_local

                        # Desired panel centre so laser still hits same local key
                        desired_center = (
                            hit_world
                            - kb_x_ax * target_local[0]
                            - kb_y_ax * target_local[1]
                        )

                        # Sphere projection around the head preserves the
                        # Euclidean head→keyboard distance so the panel doesn't
                        # grow/shrink as the user drags it through space.
                        if self._head_pos_w is not None:
                            hx, hy, hz = self._head_pos_w
                            dx = desired_center[0] - hx
                            dy = desired_center[1] - hy
                            dz = desired_center[2] - hz
                            ksx = self._keyboard_pan_x - hx
                            ksy = self._keyboard_pan_y - hy
                            ksz = -self._keyboard_distance - hz
                            R3 = math.sqrt(ksx * ksx + ksy * ksy + ksz * ksz)
                            if R3 > 0.01:
                                d_len = math.sqrt(dx * dx + dy * dy + dz * dz)
                                if d_len > 0.001:
                                    dx /= d_len; dy /= d_len; dz /= d_len
                                self._keyboard_pan_x    = float(hx + dx * R3)
                                self._keyboard_pan_y    = float(hy + dy * R3)
                                self._keyboard_distance = float(max(0.2, -(hz + dz * R3)))

                                # Auto-orient toward head; standalone offsets on top
                                base_yaw   = math.atan2(-dx, -dz)
                                base_pitch = math.asin(max(-1.0, min(1.0, dy)))
                                self._keyboard_yaw   = base_yaw   + self._kb_yaw_offset
                                self._keyboard_pitch = base_pitch + self._kb_pitch_offset
                        else:
                            # No head pose yet fall back to direct translation
                            self._keyboard_pan_x    = float(desired_center[0])
                            self._keyboard_pan_y    = float(desired_center[1])
                            self._keyboard_distance = float(max(0.2, -desired_center[2]))
                elif both_grips and not grip_now:
                    setattr(self, kb_local_attr, None)
                elif not grip_now:
                    setattr(self, kb_local_attr, None)
                elif stick_active:
                    # Stick fine-tuning the keyboard clear stale anchor so
                    # grip-to-move re-acquires at current pose on neutral.
                    setattr(self, kb_local_attr, None)
                # else: grip held but laser transiently off keyboard keep
                # anchor so the grab resumes when laser re-enters the panel.
                # Cache the keyboard position after any grip-move update.
                if grip_now and grip_target == 'keyboard':
                    self._kb_cached_position = {
                        'pan_x': self._keyboard_pan_x, 'pan_y': self._keyboard_pan_y,
                        'distance': self._keyboard_distance, 'width': self._keyboard_width,
                        'yaw': self._keyboard_yaw, 'pitch': self._keyboard_pitch,
                    }
        else:
            # Keyboard hidden clear any stale anchors
            self._kb_grab_local_l = None
            self._kb_grab_local_r = None

        # Both grips held: system move
        # Average the two laser hit positions, project onto sphere around
        # head, and move the screen as a single rigid system.  This allows
        # two-handed repositioning without individual grip fighting.
        if both_grips and not seat_adjust_active and not stick_active_l and not stick_active_r \
                and laser_l_on_screen and laser_r_on_screen and not screen_locked:
            mats = []
            for aim_mat, grab_attr, local_attr in [
                (self._aim_mat_l, '_screen_grab_grip_l', '_screen_grab_local_l'),
                (self._aim_mat_r, '_screen_grab_grip_r', '_screen_grab_local_r'),
            ]:
                if aim_mat is None:
                    continue
                is_left = (grab_attr == '_screen_grab_grip_l')
                grip_mat = self._grip_mat_l if is_left else self._grip_mat_r
                if grip_mat is not None:
                    ray_origin = (grip_mat[:3, 3] + grip_mat[:3, 1] * 0.020).astype('f8')
                else:
                    ray_origin = aim_mat[:3, 3].astype(np.float64)
                _, ray_dir = self._get_smoothed_ray(is_left)
                if ray_dir is None:
                    ray_dir = -aim_mat[:3, 2].astype(np.float64)
                ray_dir /= np.linalg.norm(ray_dir) + 1e-10
                cp = math.cos(self.screen_pitch)
                sp = math.sin(self.screen_pitch)
                cy = math.cos(self.screen_yaw)
                sy = math.sin(self.screen_yaw)
                screen_normal = np.array([cp * sy, -sp, cp * cy], dtype=np.float64)
                screen_center = self._screen_world_pos(cy, sy, cp, sp).astype(np.float64)
                denom = np.dot(screen_normal, ray_dir)
                if abs(denom) < 1e-6:
                    continue
                t_hit = np.dot(screen_normal, screen_center - ray_origin) / denom
                if t_hit < 0.0:
                    continue
                hit_world = ray_origin + ray_dir * t_hit
                right_axis = np.array([cy, 0.0, -sy], dtype=np.float64)
                up_axis = np.array([sp * sy, cp, sp * cy], dtype=np.float64)
                local_x = np.dot(hit_world - screen_center, right_axis)
                local_y = np.dot(hit_world - screen_center, up_axis)
                saved_local = getattr(self, local_attr)
                if saved_local is None:
                    setattr(self, local_attr, np.array([local_x, local_y], dtype=np.float64))
                    saved_local = np.array([local_x, local_y], dtype=np.float64)
                target_local = saved_local
                dc = hit_world - right_axis * target_local[0] - up_axis * target_local[1]
                mats.append(dc)

            if mats and self._head_pos_w is not None:
                # Average desired centers from both controllers
                avg_center = sum(mats) / len(mats)
                hx, hy, hz = self._head_pos_w
                dx = avg_center[0] - hx
                dy = avg_center[1] - hy
                dz = avg_center[2] - hz
                csx = self.screen_pan_x - hx
                csy = self.screen_pan_y - hy
                csz = -self.screen_distance - hz
                R3 = math.sqrt(csx * csx + csy * csy + csz * csz)
                if R3 > 0.01:
                    d_len = math.sqrt(dx * dx + dy * dy + dz * dz)
                    if d_len > 0.001:
                        dx /= d_len; dy /= d_len; dz /= d_len
                    self.screen_pan_x    = float(hx + dx * R3)
                    self.screen_pan_y    = float(hy + dy * R3)
                    self.screen_distance = float(-(hz + dz * R3))
                    base_yaw   = math.atan2(-dx, -dz)
                    base_pitch = math.asin(max(-1.0, min(1.0, dy)))
                    self.screen_yaw   = base_yaw   + self._yaw_offset
                    self.screen_pitch = base_pitch + self._pitch_offset

        # Seat adjust: thumbsticks move viewer position relative to screen
        # With a single grip held: thumbsticks shift the env model position
        # (room moves around the user/screen).
        if seat_adjust_active:
            single_grip = (grip_l or grip_r) and not both_grips
            self._seat_adjust_grip_move = single_grip
            if single_grip:
                ENV_MOVE_SPEED = 1.0
                mp = self._env_model_pos
                env_changed = False
                if abs(lx) > DEAD:
                    mp[0] += lx * ENV_MOVE_SPEED * dt
                    env_changed = True
                if abs(ly) > DEAD:
                    mp[2] += ly * ENV_MOVE_SPEED * dt
                    env_changed = True
                if abs(rx) > DEAD:
                    mp[0] += rx * ENV_MOVE_SPEED * dt
                    env_changed = True
                if abs(ry) > DEAD:
                    mp[1] += ry * ENV_MOVE_SPEED * dt
                    env_changed = True
                if env_changed:
                    self._env_model_pos = [round(v, 4) for v in mp]
                    self._cached_env_model_mat4_frame = -1
                    self._seat_adjust_osd_dirty = True
            else:
                SEAT_MOVE_SPEED = 0.3
                SEAT_ANGLE_SPEED = 30.0
                view = getattr(self, '_view_pose_profile', {}) or {}
                sa_x = float(view.get('x', 0.0))
                sa_y = float(view.get('y', 0.6))
                sa_z = float(view.get('z', 0.0))
                sa_angle = float(view.get('angle', 0.0))
                changed = False
                if abs(lx) > DEAD:
                    sa_x += lx * SEAT_MOVE_SPEED * dt
                    changed = True
                if abs(ly) > DEAD:
                    sa_y += -ly * SEAT_MOVE_SPEED * dt
                    sa_y = max(0.1, sa_y)
                    changed = True
                if abs(rx) > DEAD:
                    sa_angle += -rx * SEAT_ANGLE_SPEED * dt
                    sa_angle = max(-90.0, min(90.0, sa_angle))
                    sa_angle = round(sa_angle)
                    changed = True
                if abs(ry) > DEAD:
                    sa_z += ry * SEAT_MOVE_SPEED * dt
                    changed = True
                if changed:
                    view['x'] = round(sa_x, 4)
                    view['y'] = round(sa_y, 4)
                    view['z'] = round(sa_z, 4)
                    view['angle'] = round(sa_angle, 1)
                    self._view_pose_profile = view
                    self._apply_seat_adjust_xr_space(sa_x, sa_y, sa_z, sa_angle)
                    self._seat_adjust_osd_dirty = True

        # Grip + stick fine-tuning (pan, resize, rotate, depth)
        KB_MOVE_SPEED = 0.4    # m/s at full deflection
        # Depth-ratio stick control: replaces A/B + right-grip mapping
        DEPTH_RATIO_SPEED = 0.5   # units/s at full deflection
        DEPTH_RATIO_MIN   = 0.0
        DEPTH_RATIO_MAX   = 10.0
        if grip_l and not seat_adjust_active:
            if self._keyboard_visible and self._grip_target_l == 'keyboard':
                # Left grip latched to keyboard + left stick standalone
                # translation: orbit the keyboard around the head on a sphere
                # of preserved Euclidean radius, then re-aim at the head with
                # the user yaw/pitch offsets layered on top (mirrors the
                # yaw/pitch standalone pattern).
                if (abs(lx) > DEAD or abs(ly) > DEAD) and self._head_pos_w is not None:
                    hx, hy, hz = self._head_pos_w
                    ckx = self._keyboard_pan_x - hx
                    cky = self._keyboard_pan_y - hy
                    ckz = -self._keyboard_distance - hz
                    R3 = math.sqrt(ckx * ckx + cky * cky + ckz * ckz)
                    if R3 > 0.01:
                        az = math.atan2(ckx, -ckz)
                        el = math.asin(max(-1.0, min(1.0, cky / R3)))
                        # Convert linear speed at object distance to angular rate.
                        ang = KB_MOVE_SPEED * dt / max(R3, 0.3)
                        if abs(lx) > DEAD:
                            az += lx * ang
                        if abs(ly) > DEAD:
                            el = max(-1.4, min(1.4, el + ly * ang))
                        cel = math.cos(el)
                        ux = cel * math.sin(az)
                        uy = math.sin(el)
                        uz = -cel * math.cos(az)
                        self._keyboard_pan_x   = float(hx + ux * R3)
                        self._keyboard_pan_y   = float(hy + uy * R3)
                        self._keyboard_distance = float(-(hz + uz * R3))
                        base_yaw   = math.atan2(-ux, -uz)
                        base_pitch = math.asin(max(-1.0, min(1.0, uy)))
                        self._keyboard_yaw   = base_yaw   + self._kb_yaw_offset
                        self._keyboard_pitch = base_pitch + self._kb_pitch_offset
            else:
                # Left grip + left stick standalone screen translation:
                # orbit the screen around the head on a sphere of preserved
                # Euclidean radius, then re-aim at the head with the user
                # yaw/pitch offsets layered on top.
                if laser_on_screen and (abs(lx) > DEAD or abs(ly) > DEAD) \
                        and self._head_pos_w is not None and not screen_locked:
                    hx, hy, hz = self._head_pos_w
                    csx = self.screen_pan_x - hx
                    csy = self.screen_pan_y - hy
                    csz = -self.screen_distance - hz
                    R3 = math.sqrt(csx * csx + csy * csy + csz * csz)
                    if R3 > 0.01:
                        az = math.atan2(csx, -csz)
                        el = math.asin(max(-1.0, min(1.0, csy / R3)))
                        ang = self._pan_speed * dt / max(R3, 0.3)
                        if abs(lx) > DEAD:
                            az += lx * ang
                        if abs(ly) > DEAD:
                            el = max(-1.4, min(1.4, el + ly * ang))
                        cel = math.cos(el)
                        ux = cel * math.sin(az)
                        uy = math.sin(el)
                        uz = -cel * math.cos(az)
                        self.screen_pan_x    = float(hx + ux * R3)
                        self.screen_pan_y    = float(hy + uy * R3)
                        self.screen_distance = float(-(hz + uz * R3))
                        base_yaw   = math.atan2(-ux, -uz)
                        base_pitch = math.asin(max(-1.0, min(1.0, uy)))
                        self.screen_yaw   = base_yaw   + self._yaw_offset
                        self.screen_pitch = base_pitch + self._pitch_offset
        elif grip_r and not seat_adjust_active:
            # Right grip + left stick Y depth_ratio (no laser required)
            if abs(ly) > DEAD:
                old_val = self.depth_ratio
                self.depth_ratio = max(DEPTH_RATIO_MIN,
                                    min(DEPTH_RATIO_MAX,
                                        self.depth_ratio + ly * DEPTH_RATIO_SPEED * dt))
                if self.depth_ratio != old_val:
                    self._depth_osd_show_t = time.perf_counter()
                    self._mark_runtime_settings_dirty()
            # Don't send desktop scroll events while screen grabbed/manipulated
            if not (self._grabbed or grip_l or grip_r):
                self._accum_scroll(lx, 0.0, dt)
        else:
            if not (self._grabbed or grip_l or grip_r):
                self._accum_scroll(lx, ly, dt)

        # Right grip + right stick X: resize screen width 
        # Right grip + right stick Y: screen distance (acceleration curve) 
        # Left grip + right stick X: rotate screen yaw around its centre 
        # Left grip + right stick Y: depth strength (unchanged) 
        # Right stick (no grip): mouse scroll
        RESIZE_SPEED = 1.2    # m/s of width change at full deflection
        if grip_r and not grip_l:
            if self._keyboard_visible and self._grip_target_r == 'keyboard':
                if abs(rx) > abs(ry) and abs(rx) > DEAD:
                    self._keyboard_width = max(0.3,
                        self._keyboard_width + rx * RESIZE_SPEED * dt)
                    self._keyboard_height = None  # recalc from aspect
                elif abs(ry) > abs(rx) and abs(ry) > DEAD:
                    # Right-grip + right-stick Y radial distance along head→keyboard ray
                    _t = (abs(ry) - DEAD) / (1.0 - DEAD)
                    _speed = (self._dist_speed_base
                            + (self._dist_speed_max - self._dist_speed_base) * (_t ** self._dist_speed_exp))
                    if self._head_pos_w is not None:
                        hx, hy, hz = self._head_pos_w
                        ckx = self._keyboard_pan_x - hx
                        cky = self._keyboard_pan_y - hy
                        ckz = -self._keyboard_distance - hz
                        R3 = math.sqrt(ckx * ckx + cky * cky + ckz * ckz)
                        if R3 > 0.01:
                            ux = ckx / R3
                            uy = cky / R3
                            uz = ckz / R3
                            delta = _speed * (1.0 if ry > 0 else -1.0) * dt
                            R3_new = max(0.2, R3 + delta)
                            self._keyboard_pan_x   = float(hx + ux * R3_new)
                            self._keyboard_pan_y   = float(hy + uy * R3_new)
                            self._keyboard_distance = float(-(hz + uz * R3_new))
                            # Re-aim keyboard at the head (auto-orientation),
                            # layering user-configured offsets on top.
                            base_yaw   = math.atan2(-ux, -uz)
                            base_pitch = math.asin(max(-1.0, min(1.0, uy)))
                            self._keyboard_yaw   = base_yaw   + self._kb_yaw_offset
                            self._keyboard_pitch = base_pitch + self._kb_pitch_offset
                    else:
                        self._keyboard_distance = max(0.2,
                            self._keyboard_distance + ry * self._dist_speed_base * dt)
            else:
                if abs(rx) > abs(ry) and abs(rx) > DEAD and not self._environment_screen_locked():
                    self._screen_ref_size = max(0.8,
                                            self._screen_ref_size + rx * RESIZE_SPEED * dt)
                    self.screen_height = None
                    self._resizing = True
                    self._screen_osd_show_t = time.perf_counter()
                elif abs(ry) > abs(rx) and abs(ry) > DEAD and not screen_locked:
                    # Right-grip + right-stick Y -> radial distance along head->screen ray
                    _t = (abs(ry) - DEAD) / (1.0 - DEAD)
                    _speed = (self._dist_speed_base
                            + (self._dist_speed_max - self._dist_speed_base) * (_t ** self._dist_speed_exp))
                    # Move screen along the head→screen radial direction
                    if self._head_pos_w is not None:
                        hx, hy, hz = self._head_pos_w
                        csx = self.screen_pan_x - hx
                        csy = self.screen_pan_y - hy
                        csz = -self.screen_distance - hz
                        R3 = math.sqrt(csx * csx + csy * csy + csz * csz)
                        if R3 > 0.01:
                            ux = csx / R3
                            uy = csy / R3
                            uz = csz / R3
                            delta = _speed * (1.0 if ry > 0 else -1.0) * dt
                            R3_new = max(0.3, R3 + delta)
                            self.screen_pan_x    = float(hx + ux * R3_new)
                            self.screen_pan_y    = float(hy + uy * R3_new)
                            self.screen_distance = float(-(hz + uz * R3_new))
                    self._screen_osd_show_t = time.perf_counter()
        elif grip_l and not grip_r:
            if self._keyboard_visible and self._grip_target_l == 'keyboard':
                # Left grip latched onto keyboard + right stick standalone
                # keyboard yaw / pitch offsets (auto-orientation still aims at
                # head, these offsets are layered on top).
                if abs(rx) > DEAD:
                    self._kb_yaw_offset   -= rx * self._rot_speed * dt
                    self._keyboard_yaw    -= rx * self._rot_speed * dt
                if abs(ry) > DEAD:
                    self._kb_pitch_offset += ry * self._rot_speed * dt
                    self._keyboard_pitch  += ry * self._rot_speed * dt
            else:
                # Left grip + right stick X screen yaw rotation
                if laser_on_screen and abs(rx) > DEAD and not screen_locked:
                    self._yaw_offset -= rx * self._rot_speed * dt
                    self.screen_yaw  -= rx * self._rot_speed * dt
                # Left grip + right stick Y screen pitch rotation
                if laser_on_screen and abs(ry) > DEAD and not screen_locked:
                    self._pitch_offset += ry * self._rot_speed * dt
                    self.screen_pitch  += ry * self._rot_speed * dt
        self._grip_r_prev = grip_r
        # Re-sync grab offsets after stick fine-tuning.  Grip-to-move is paused
        # while the stick is active, so full XYZ re-sync is safe here.
        for grip_now, grip_mat, grab_attr in [
            (grip_l, self._grip_mat_l, '_screen_grab_grip_l'),
            (grip_r, self._grip_mat_r, '_screen_grab_grip_r'),
        ]:
            if grip_now and grip_mat is not None:
                saved = getattr(self, grab_attr)
                if saved is not None:
                    grip_pos = grip_mat[:3, 3].astype('f8')
                    screen_center = np.array([
                        self.screen_pan_x, self.screen_pan_y,
                        -self.screen_distance], dtype='f8')
                    setattr(self, grab_attr, screen_center - grip_pos)
        if not grip_r:
            # Accelerated scroll: higher stick deflection -> disproportionately faster scroll
            if not (self._grabbed or grip_l):
                self._accum_scroll(rx, ry, dt)

        # Rebuild keyboard geometry if width changed
        if (self._keyboard_visible and self._keyboard_tex is not None
                and abs(self._keyboard_width - self._kb_last_build_width) > 0.001):
            self._keyboard_height = (self._keyboard_width
                                    * _KB_TEX_H / float(_KB_TEX_W))
            self._build_keyboard_texture()
            self._kb_last_build_width = self._keyboard_width

        if not seat_adjust_active:
            self._move_env_with_screen_delta(locked_old_screen_mat)

        # Menu (left): short press toggle status/FPS panel
        menu_now = self._read_bool_action(self._act_menu_btn, "/user/hand/left")
        MENU_LONG = 0.6  # seconds long press reserved for calibration combo
        if menu_now and not self._menu_pressed_last:
            self._menu_press_t = time.perf_counter()
            self._menu_long_fired = False
        if not menu_now and self._menu_pressed_last:
            if not self._menu_long_fired and (time.perf_counter() - self._menu_press_t) < MENU_LONG:
                self._fps_overlay_visible = not self._fps_overlay_visible
        self._menu_pressed_last = menu_now

        # A / B (right):
        #   Previously: right-grip + A/B adjusted depth_ratio. Now: right-grip +
        #   right-stick Y adjusts depth_ratio. When A+B are held together (brand
        #   switch combo) A/B's other functions are suppressed.
        a_now = self._read_bool_action(self._act_a_btn, "/user/hand/right") or self._emu_a
        b_now = self._read_bool_action(self._act_b_btn, "/user/hand/right") or self._emu_b

        if not ab_held:
            # When A+B are not held together, A/B keep their normal behaviour.
            # If right-grip is held we intentionally do not perform A/B immediate
            # actions here (depth_ratio is adjusted via right-stick Y below).
            if not grip_r:
                # Use XR runtime's `changed` flag when available more reliable than
                # manual frame-to-frame tracking when a button sits under a resting thumb.
                # Fall back to manual edge detection if pyopenxr doesn't expose it.
                a_edge = self._read_bool_edge(self._act_a_btn, "/user/hand/right", self._a_last)
                b_edge = self._read_bool_edge(self._act_b_btn, "/user/hand/right", self._b_last)
                if not a_edge:
                    a_edge = a_now and not self._a_last
                if not b_edge:
                    b_edge = b_now and not self._b_last
                # Only send OS mouse clicks if the right controller laser is
                # currently intersecting the virtual screen. This prevents A/B
                # from clicking when pointing off-screen or at the overlay panel.
                is_gripping = self._grabbed
                if a_edge and laser_r_on_screen and not is_gripping:
                    _send_mouse_flags(_MOUSEEVENTF_LEFTDOWN)
                    _send_mouse_flags(_MOUSEEVENTF_LEFTUP)
                if b_edge and laser_r_on_screen and not is_gripping:
                    _send_mouse_flags(_MOUSEEVENTF_RIGHTDOWN)
                    _send_mouse_flags(_MOUSEEVENTF_RIGHTUP)

        self._a_last = a_now
        self._b_last = b_now

        # Y (left):
        #   short press  reset screen to upright default (same as session start)
        #   long press   reset screen to face current head gaze (same as home long-press)
        Y_LONG = 0.6   # seconds to trigger long-press action
        y_now = self._read_bool_action(self._act_y_btn, "/user/hand/left") or self._emu_y
        if y_now and not self._y_last:
            self._y_press_t    = time.perf_counter()
            self._y_long_fired = False
        if y_now and not self._y_long_fired and not seat_adjust_active:
            if time.perf_counter() - self._y_press_t >= Y_LONG:
                if screen_locked:
                    if not (self._env_uses_view_pose_cycle() and self._cycle_view_pose()):
                        self._cycle_lighting_preset()
                else:
                    nxt = (self._preset_index + 1) % len(self._screen_presets)
                    self._apply_preset(nxt)
                self._y_long_fired = True
        if not y_now and self._y_last and not self._y_long_fired and not seat_adjust_active:
            if screen_locked:
                self._reset_seating_vertical()
            else:
                self._reset_screen_to_default(show_border=True)
        self._y_last = y_now

        # X (left):
        #   release <1s              toggle virtual keyboard
        #   release 1~4s             cycle glow mode (glow -> veil -> frosted -> off)
        #   hold >4s (release)       toggle VDXR green passthrough backdrop
        X_GLOW_HOLD = 1.0
        X_PASSTHROUGH_HOLD = 4.0
        x_now = self._read_bool_action(self._act_x_btn, "/user/hand/left") or self._emu_x

        if x_now and not self._x_last:                     # rising edge
            self._x_press_t = time.perf_counter()
            self._x_long_fired = False

        if x_now and not self._x_long_fired:               # still held, not yet fired
            held = time.perf_counter() - self._x_press_t
            if held >= X_PASSTHROUGH_HOLD:
                self._toggle_passthrough_backdrop()
                self._x_long_fired = True
                # Prevent the glow action from also firing on release
                self._x_glow_fired = True

        if not x_now and self._x_last:                     # falling edge
            held = time.perf_counter() - self._x_press_t
            if not self._x_long_fired:                     # passthrough was already triggered
                if not getattr(self, '_x_glow_fired', False):
                    # 1s <= release < 4s: toggle light
                    if held >= X_GLOW_HOLD:
                        self._cycle_light_from_x()
                    else:
                        # <1s: toggle keyboard
                        self._keyboard_visible = not self._keyboard_visible
                        if self._keyboard_visible:
                            if self._keyboard_tex is None:
                                self._init_keyboard()
                            cached = getattr(self, '_kb_cached_position', None)
                            if cached is not None:
                                self._kb_restore_cached_position(cached)
                            else:
                                self._kb_cached_position = self._anchor_keyboard_below_screen()
        elif not x_now:
            self._x_glow_fired = False  # reset when button wasn't pressed

        self._x_last = x_now

        # Thumbstick buttons: distinguish between single and double thumbstick presses
        lsc_now = self._read_bool_action(self._act_left_stick_click, "/user/hand/left")
        rsc_now = self._read_bool_action(self._act_right_stick_click, "/user/hand/right")
        # Trackpad emulation: suppress stick-click when a button region (top/bottom)
        # was hit, so clicking the top fires B/Y but not stick-click.
        # Center-click emulation is OR'd in so it still acts as stick-click.
        if self._emu_x or self._emu_y:
            lsc_now = False
        else:
            lsc_now = lsc_now or self._emu_lsc
        if self._emu_a or self._emu_b:
            rsc_now = False
        else:
            rsc_now = rsc_now or self._emu_rsc
        BOTH_LONG  = 0.5   # Duration to trigger FPS/help panel toggle with both sticks
        SINGLE_LONG = 1.0  # Duration for single thumbstick long press to switch shortcut panels

        # Both thumbsticks pressed toggle FPS/help panel after 0.5 seconds (unchanged)
        both_clicked = lsc_now and rsc_now
        if both_clicked and not self._both_stick_fired:
            if self._both_stick_start == 0.0:
                self._both_stick_start = time.perf_counter()
            elif time.perf_counter() - self._both_stick_start >= BOTH_LONG:
                self._fps_overlay_visible = not self._fps_overlay_visible
                self._both_stick_fired = True
                # Mark single-stick long-fired to suppress their short actions
                self._lsc_long_fired = True
                self._rsc_long_fired = True
        if not both_clicked:
            self._both_stick_start = 0.0
            self._both_stick_fired = False

        # Non-double-press case: handle each thumbstick separately
        # - Long press (>= SINGLE_LONG) triggers shortcut panel (same as double press);
        # - Short press (released when < SINGLE_LONG) performs background/curve mode switching or grip modifier actions;
        if not both_clicked:
            now = self._frame_now

            # Left thumbstick: long-press toggle status/shortcut panel;
            #                  short-press cycle environment
            if lsc_now and not self._left_stick_click_prev:
                self._lsc_press_t = now
                self._lsc_long_fired = False
            if lsc_now and not self._lsc_long_fired:
                if now - getattr(self, '_lsc_press_t', 0.0) >= SINGLE_LONG:
                    self._fps_overlay_visible = not self._fps_overlay_visible
                    self._lsc_long_fired = True
            if not lsc_now and self._left_stick_click_prev:
                # Released if long-press wasn't fired, treat as short press
                if not self._lsc_long_fired:
                    if grip_r:
                        if self.depth_strength > 0.0:
                            self._depth_strength_saved = self.depth_strength
                            self.depth_strength = 0.0
                        else:
                            self.depth_strength = getattr(self, '_depth_strength_saved', 0.1)
                    else:
                        self._cycle_environment()

            # Right thumbstick: long-press reset screen direction (keep distance + size);
            #          short-press cycle horizontal curve -> vertical curve -> flat
            if rsc_now and not self._right_stick_click_prev:
                self._rsc_press_t = now
                self._rsc_long_fired = False
            if rsc_now and not self._rsc_long_fired:
                if now - getattr(self, '_rsc_press_t', 0.0) >= SINGLE_LONG:
                    self._reset_screen_direction()
                    self._rsc_long_fired = True
            if not rsc_now and self._right_stick_click_prev:
                if not self._rsc_long_fired:
                    if not grip_r and not grip_l and (
                            not self._environment_screen_locked() or self._env_allow_curve):
                        self._cycle_screen_curve_mode()
                    elif grip_r:
                        old_val = self.depth_ratio
                        self.depth_ratio = 2.0
                        if self.depth_ratio != old_val:
                            self._mark_runtime_settings_dirty()

        self._left_stick_click_prev = lsc_now
        self._right_stick_click_prev = rsc_now

        # Border fade: snap to 1 while the user is actively re-positioning, fade out
        # when idle. `active` is computed at the top of this method.
        FADE_DELAY = 1.5   # seconds before starting to fade
        FADE_DUR   = 0.8   # fade-out duration in seconds
        if active:
            self._border_alpha  = 1.0
            self._border_idle_t = time.perf_counter()
        else:
            idle = time.perf_counter() - self._border_idle_t
            if idle > FADE_DELAY:
                self._border_alpha = max(0.0, 1.0 - (idle - FADE_DELAY) / FADE_DUR)

        # Keyboard border fade: show while gripping AND laser on keyboard (not just off-screen)
        kb_hit = self._kb_hover_l is not None or self._kb_hover_r is not None
        kb_active = self._keyboard_visible and (grip_l or grip_r) and kb_hit
        if kb_active:
            self._kb_border_alpha  = 1.0
            self._kb_border_idle_t = time.perf_counter()
        else:
            idle = time.perf_counter() - self._kb_border_idle_t
            if idle > FADE_DELAY:
                self._kb_border_alpha = max(0.0, 1.0 - (idle - FADE_DELAY) / FADE_DUR)

        # Trigger input fires mouse clicks (skips keys claimed by keyboard)
        self._handle_triggers()

        # Debounced screen state persistence (default / non-env mode only)
        if not screen_locked:
            cur = (self.screen_width, self.screen_distance, self.screen_pan_x,
                   self.screen_pan_y, self.screen_yaw, self.screen_pitch,
                   self._screen_curve_mode(), self._preset_index)
            prev = getattr(self, '_prev_screen_snapshot', None)
            if prev != cur:
                self._screen_state_dirty = True
                self._screen_state_save_t = time.perf_counter()
                self._prev_screen_snapshot = cur
            if self._screen_state_dirty and time.perf_counter() - self._screen_state_save_t > 1.0:
                self._screen_state_dirty = False
                self._persist_screen_state()
        self._flush_runtime_settings_if_idle()

    # Main blocking loop
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

        screen_h = self.screen_height if self.screen_height else (
            self.screen_width * self._effective_frame_aspect() if self.frame_size else 1.35
        )
        curve_mode = self._screen_curve_mode()
        print(f"[OpenXRViewer] Screen: {self.screen_width:.2f}m x {screen_h:.2f}m @ {self.screen_distance:.2f}m"
              f"{f' (curved {curve_mode})' if curve_mode != 'none' else ''}")
        print(f"[OpenXRViewer] IPD: {self.ipd_uv:.3f}m, Depth: {self.depth_ratio}, Frames: {self.frame_size}")

        try:
            if not self._wait_for_openxr_device(shutdown_event):
                print("[OpenXRViewer] OpenXR startup cancelled while waiting for XR device")
                self.cleanup()
                return
        except Exception as e:
            import traceback
            print(f"[OpenXRViewer] OpenXR init failed: {e}")
            print(traceback.format_exc())
            self.cleanup()
            raise

        # Optional desktop mirror of the left eye, useful for checking XR output
        # without wearing the headset. The GUI controls this via settings.yaml.
        try:
            glfw.set_window_size(self.window, 960, 540)
            glfw.set_window_pos(self.window, 100, 100)
            if self._show_preview_window:
                glfw.show_window(self.window)
                self._preview_active = True
                print("[OpenXRViewer] Desktop preview window opened")
            else:
                self._preview_active = False
                print("[OpenXRViewer] Desktop preview window disabled")
        except Exception as e:
            print(f"[OpenXRViewer] Preview window setup failed: {e}")

        # Widen the system double-click window so VR trigger taps have more time.
        # Default Windows value is 500 ms too tight for controller triggers.
        # We restore the original value in cleanup() so no permanent side-effects.
        if sys.platform == "win32" and self._saved_dclick_time is None:
            self._saved_dclick_time = _U32.GetDoubleClickTime()
            _U32.SystemParametersInfoW(0x0020, 1200, None, 0)  # SPI_SETDOUBLECLICKTIME

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
            self._frame_now = now   # shared per-frame timestamp (overlays/input reuse)
            dt = now - last_input_t
            last_input_t = now
            self._last_frame_dt = dt
            self._frame_count += 1

            glfw.poll_events()
            self._poll_xr_events()

            if not self._session_running:
                time.sleep(0.01)
                continue

            # Lazy init: load environment model after OpenXR session is running
            if not self._env_model_init_done:
                self._env_model_init_done = True
                self._init_dark_room()
                self._configure_environment_profile()
                self._configure_profile_view_layout()
                self._init_env_model()
                self._apply_profile_screen_layout()
                print(f"[OpenXRViewer] Lazy env model: {len(self._env_model_prims)} prims")

            # Wait for the runtime to signal frame timing 
            frame_state = xr.wait_frame(self._xr_session, self._xr_frame_wait_info)
            xr.begin_frame(self._xr_session, self._xr_frame_begin_info)

            # sync_actions must happen before xr.locate_space for action spaces.
            # Do it here so _update_aim_poses gets fresh locations this frame.
            if self._xr_actions_sync_info is not None:
                try:
                    xr.sync_actions(self._xr_session, self._xr_actions_sync_info)
                except Exception:
                    pass

            # Locate controller spaces (now valid after sync_actions)
            self._update_aim_poses(frame_state.predicted_display_time)
            self._update_grip_poses(frame_state.predicted_display_time)

            # Skip smoothing + input polling when no controllers are tracked,
            # but keep locating poses so we detect reconnection immediately.
            # Wait 30 frames (~0.4s at 72Hz) before throttling avoids
            # toggling on transient tracking loss.
            both_missing = (self._aim_mat_l is None and self._aim_mat_r is None)
            if both_missing:
                self._controller_miss_frames += 1
            else:
                self._controller_miss_frames = 0

            if self._controller_miss_frames < 30:
                self._smooth_controller_poses()
                self._poll_controller_input(dt)
            else:
                # No controllers clear cursor/grab state so downstream code
                # (grip-to-move, trigger handling) sees no laser on screen.
                self._cursor_uv_l = None
                self._cursor_uv_r = None
                self._cursor_ctrl = None
                self._cursor_smooth_uv = None
                self._grabbed = False

            composition_layers = []

            if frame_state.should_render:
                # Drain depth_q non-blocking keep only the newest frame
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

                # Cache head pose for the next frame's input handlers (left-stick
                # orbit pivot + keyboard anchoring). One-frame stale is imperceptible
                # at 90 Hz and avoids needing a second xr.locate_views call.
                if views and views[0] is not None and views[1] is not None:
                    try:
                        p0 = views[0].pose.position
                        p1 = views[1].pose.position
                        self._head_pos_w = (
                            (p0.x + p1.x) / 2.0,
                            (p0.y + p1.y) / 2.0,
                            (p0.z + p1.z) / 2.0,
                        )
                        qm = _xr_quat_to_mat4(views[0].pose.orientation)
                        # Forward = -Z column of the head rotation matrix.
                        self._head_fwd_w = (
                            float(-qm[0, 2]),
                            float(-qm[1, 2]),
                            float(-qm[2, 2]),
                        )
                    except Exception:
                        pass

                self._last_located_views = views

                if self._pending_recenter and views and views[0] is not None:
                    self._pending_recenter = False
                    if self._environment_screen_locked() and self._active_environment:
                        self._reset_locked_environment_to_profile(show_border=False)
                        self._recenter_profile_view_pose()
                    else:
                        self._reset_screen_direction()

                # On the first valid frame, place the screen in front of the user's
                # current gaze identical to pressing Y so startup matches reset.
                # EXCEPT when a profile screen layout has already been applied
                # (see _apply_profile_screen_layout).  In that case the profile
                # layout is already live and we must NOT overwrite it; otherwise
                # the GUI Stop / Restart round-trip silently loses the layout.
                if not self._screen_eye_init and views and views[0] is not None:
                    try:
                        ey = (views[0].pose.position.y + views[1].pose.position.y) / 2.0
                        self._initial_head_y = float(ey)
                    except Exception:
                        pass
                    self._apply_profile_view_pose_to_xr_space(views)
                    if not self._profile_loaded:
                        if not self._restore_screen_state():
                            self._reset_screen_to_default(show_border=False)
                    self._screen_eye_init = True

                eye_layer_views = []

                if self._use_d3d11:
                    # D3D11 rendering
                    #
                    # Prefer GPU interop (NV_DX_interop2 or EXT_memory_object)
                    # which avoids the CPU round-trip entirely.
                    # Fall back to the optimized PBO readback path otherwise.

                    if self._interop_mode == 'nv_dx':
                        # NV_DX_interop2: render directly into swapchain textures
                        for eye_index in range(2):
                            swapchain = self._xr_swapchains[eye_index]
                            img_index = xr.acquire_swapchain_image(swapchain, self._xr_sc_acquire_info)
                            xr.wait_swapchain_image(swapchain, self._xr_sc_wait_info)
                            sc_image = self._swapchain_images[eye_index][img_index]
                            sc_w, sc_h = self._swapchain_sizes[eye_index]
                            view = views[eye_index] if views and views[eye_index] else None
                            view_mat = _pose_to_view_mat4(view.pose) if view else np.eye(4, dtype=np.float32)
                            proj_mat = _fov_to_proj_mat4_cached(view.fov)   if view else _default_proj

                            mgl_fbo, raw_fbo = self._get_or_create_nv_interop_fbo(
                                eye_index, img_index, sc_image.texture, sc_w, sc_h,
                            )
                            # Lock the registered D3D11 texture for GL access
                            _, _, dx_obj = self._nv_dx_objects[(eye_index, img_index)]
                            _render._wglDXLockObjectsNV(self._nv_dx_device, 1, ctypes.byref(dx_obj))
                            try:
                                self._render_eye(eye_index, mgl_fbo, view_mat, proj_mat, flip_y=True)
                            finally:
                                _render._wglDXUnlockObjectsNV(self._nv_dx_device, 1, ctypes.byref(dx_obj))

                            xr.release_swapchain_image(swapchain, self._xr_sc_release_info)
                            eye_layer_views.append(xr.CompositionLayerProjectionView(
                                pose=view.pose if view else xr.Posef(),
                                fov=view.fov   if view else _default_fov,
                                sub_image=xr.SwapchainSubImage(
                                    swapchain=swapchain,
                                    image_rect=xr.Rect2Di(
                                        offset=xr.Offset2Di(x=0, y=0),
                                        extent=xr.Extent2Di(width=sc_w, height=sc_h),
                                    ),
                                ),
                            ))

                    elif self._interop_mode == 'ext_mem':
                        # EXT_memory_object two-phase loop. Phase 1 submits all
                        # GL work and fences per eye; phase 2 waits/copies to
                        # D3D11, avoiding a full-pipeline glFinish per eye.
                        ext_pending = []  # (eye_index, fence, d3d11_tex, sc_w, sc_h, swapchain, view)
                        for eye_index in range(2):
                            swapchain = self._xr_swapchains[eye_index]
                            img_index = xr.acquire_swapchain_image(swapchain, self._xr_sc_acquire_info)
                            xr.wait_swapchain_image(swapchain, self._xr_sc_wait_info)
                            sc_image = self._swapchain_images[eye_index][img_index]
                            sc_w, sc_h = self._swapchain_sizes[eye_index]
                            view = views[eye_index] if views and views[eye_index] else None
                            view_mat = _pose_to_view_mat4(view.pose) if view else np.eye(4, dtype=np.float32)
                            proj_mat = _fov_to_proj_mat4_cached(view.fov)   if view else _default_proj

                            if self._swapchain_is_bgra:
                                offs_mgl_fbo, _ = self._get_or_create_offscreen_fbo(eye_index, img_index, sc_w, sc_h)
                                self._render_eye(eye_index, offs_mgl_fbo, view_mat, proj_mat, flip_y=True)
                                offs_tex = self._offscreen_fbo_cache[(eye_index, img_index)][2]
                                self._swizzle_blit_to_shared(eye_index, offs_tex)
                            else:
                                mgl_fbo = self._ext_shared_tex[eye_index][3]
                                self._render_eye(eye_index, mgl_fbo, view_mat, proj_mat, flip_y=True)

                            fence = self._make_gl_fence()
                            ext_pending.append((eye_index, fence, sc_image.texture,
                                                sc_w, sc_h, swapchain, view))

                        for eye_index, fence, d3d11_tex, sc_w, sc_h, swapchain, view in ext_pending:
                            self._wait_and_blit_ext(eye_index, d3d11_tex, fence)
                            xr.release_swapchain_image(swapchain, self._xr_sc_release_info)
                            eye_layer_views.append(xr.CompositionLayerProjectionView(
                                pose=view.pose if view else xr.Posef(),
                                fov=view.fov   if view else _default_fov,
                                sub_image=xr.SwapchainSubImage(
                                    swapchain=swapchain,
                                    image_rect=xr.Rect2Di(
                                        offset=xr.Offset2Di(x=0, y=0),
                                        extent=xr.Extent2Di(width=sc_w, height=sc_h),
                                    ),
                                ),
                            ))

                    else:
                        # PBO fallback: two-phase loop to overlap GPU DMA with rendering.
                        d3d11_pending = []   # (eye_index, pbo_id, d3d11_tex, sc_w, sc_h, swapchain, view)

                        for eye_index in range(2):
                            swapchain = self._xr_swapchains[eye_index]
                            img_index = xr.acquire_swapchain_image(swapchain, self._xr_sc_acquire_info)
                            xr.wait_swapchain_image(swapchain, self._xr_sc_wait_info)
                            sc_image = self._swapchain_images[eye_index][img_index]
                            sc_w, sc_h = self._swapchain_sizes[eye_index]
                            view = views[eye_index] if views and views[eye_index] else None
                            view_mat = _pose_to_view_mat4(view.pose) if view else np.eye(4, dtype=np.float32)
                            proj_mat = _fov_to_proj_mat4_cached(view.fov)   if view else _default_proj

                            mgl_fbo, raw_fbo_id = self._get_or_create_offscreen_fbo(eye_index, img_index, sc_w, sc_h)
                            self._render_eye(eye_index, mgl_fbo, view_mat, proj_mat, flip_y=True)

                            pbo_id = self._get_or_create_d3d11_pbo(eye_index, img_index, sc_w, sc_h)
                            self._submit_pbo_readback(raw_fbo_id, pbo_id, sc_w, sc_h)
                            d3d11_pending.append((eye_index, pbo_id, sc_image.texture, sc_w, sc_h, swapchain, view))

                        # Phase 2: map PBOs (DMA should be done), upload, release.
                        for eye_index, pbo_id, d3d11_tex, sc_w, sc_h, swapchain, view in d3d11_pending:
                            self._upload_pbo_to_d3d11(pbo_id, d3d11_tex, sc_w, sc_h)
                            xr.release_swapchain_image(swapchain, self._xr_sc_release_info)
                            eye_layer_views.append(xr.CompositionLayerProjectionView(
                                pose=view.pose if view else xr.Posef(),
                                fov=view.fov   if view else _default_fov,
                                sub_image=xr.SwapchainSubImage(
                                    swapchain=swapchain,
                                    image_rect=xr.Rect2Di(
                                        offset=xr.Offset2Di(x=0, y=0),
                                        extent=xr.Extent2Di(width=sc_w, height=sc_h),
                                    ),
                                ),
                            ))

                else:
                    for eye_index in range(2):
                        swapchain = self._xr_swapchains[eye_index]

                        img_index = xr.acquire_swapchain_image(
                            swapchain, self._xr_sc_acquire_info
                        )
                        xr.wait_swapchain_image(swapchain, self._xr_sc_wait_info)

                        sc_image = self._swapchain_images[eye_index][img_index]
                        sc_w, sc_h = self._swapchain_sizes[eye_index]

                        view = views[eye_index] if views and views[eye_index] else None
                        view_mat = _pose_to_view_mat4(view.pose) if view else np.eye(4, dtype=np.float32)
                        proj_mat = _fov_to_proj_mat4_cached(view.fov)   if view else _default_proj

                        raw_fbo, mgl_fbo = self._get_or_create_fbo(eye_index, img_index, sc_image.image, sc_w, sc_h)
                        self._render_eye(eye_index, mgl_fbo, view_mat, proj_mat)

                        # Desktop preview: blit left eye to GLFW window
                        if self._preview_active and eye_index == 0:
                            pw, ph = glfw.get_window_size(self.window)
                            if pw > 0 and ph > 0:
                                glBindFramebuffer(GL_READ_FRAMEBUFFER, raw_fbo)
                                glReadBuffer(GL_COLOR_ATTACHMENT0)
                                glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
                                glBlitFramebuffer(0, 0, sc_w, sc_h, 0, 0, pw, ph,
                                                  GL_COLOR_BUFFER_BIT, GL_LINEAR)
                                glfw.swap_buffers(self.window)

                        xr.release_swapchain_image(swapchain, self._xr_sc_release_info)

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

            # Timestamp-ring FPS: (N-1) frames / (last_ts - first_ts) exact, O(1)
            t_now = time.perf_counter()
            self._frame_ts_ring.append(t_now)
            fps_sample = 0.0
            try:
                period_ns = int(getattr(frame_state, 'predicted_display_period', 0))
                if period_ns > 0:
                    fps_sample = 1_000_000_000.0 / float(period_ns)
            except Exception:
                fps_sample = 0.0
            if fps_sample <= 0.0:
                n = len(self._frame_ts_ring)
                if n >= 2:
                    span = t_now - self._frame_ts_ring[0]
                    if span > 0:
                        fps_sample = (n - 1) / span
            if fps_sample > 0.0:
                if self._fps_display_ema <= 0.0:
                    self._fps_display_ema = fps_sample
                else:
                    self._fps_display_ema += 0.12 * (fps_sample - self._fps_display_ema)
                self.actual_fps = self._fps_display_ema

        self.cleanup()

    # Cleanup
    def cleanup(self):
        """Release all OpenXR and OpenGL resources."""
        try:
            self._persist_screen_state()
        except Exception:
            pass
        self._persist_runtime_settings()

        if sys.platform == "win32" and self._saved_dclick_time is not None:
            _U32.SystemParametersInfoW(0x0020, self._saved_dclick_time, None, 0)
            self._saved_dclick_time = None

        # Release any held multi-touch contacts so apps don't see a stuck
        # finger after we exit (e.g., a half-completed window drag).
        if _TOUCH_AVAILABLE and _touch_injector is not None:
            try:
                _touch_injector.cancel_all()
            except Exception:
                pass

        self._cleanup_interop()

        raw_ids = [raw_id for raw_id, _ in self._fbo_cache.values()]
        if raw_ids:
            try:
                glDeleteFramebuffers(len(raw_ids), raw_ids)
            except Exception:
                pass
        self._fbo_cache.clear()

        depth_rbs = list(self._depth_rb_cache.values())
        if depth_rbs:
            try:
                glDeleteRenderbuffers(len(depth_rbs), depth_rbs)
            except Exception:
                pass
        self._depth_rb_cache.clear()

        # Release D3D11-path PBOs used for async pixel readback
        if self._d3d11_pbo_cache:
            try:
                glDeleteBuffers(len(self._d3d11_pbo_cache), [v[0] for v in self._d3d11_pbo_cache.values()])
            except Exception:
                pass
            self._d3d11_pbo_cache.clear()

        # Release D3D11-path offscreen FBOs and their backing textures
        offscreen_raw_ids = [entry[1] for entry in self._offscreen_fbo_cache.values()]
        if offscreen_raw_ids:
            try:
                glDeleteFramebuffers(len(offscreen_raw_ids), offscreen_raw_ids)
            except Exception:
                pass
        for entry in self._offscreen_fbo_cache.values():
            try:
                entry[2].release()   # mgl Texture
            except Exception:
                pass
            try:
                glDeleteRenderbuffers(1, [entry[5]])  # depth_rb
            except Exception:
                pass
        self._offscreen_fbo_cache.clear()

        # Release D3D11 device/context (COM objects call Release via vtable)
        for d3d_obj in (self._d3d11_context, self._d3d11_device):
            if d3d_obj is not None:
                try:
                    vtbl = ctypes.cast(d3d_obj, ctypes.POINTER(ctypes.c_void_p)).contents.value
                    release_fn = ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(
                        ctypes.cast(vtbl + 2 * ctypes.sizeof(ctypes.c_void_p),
                                    ctypes.POINTER(ctypes.c_void_p)).contents.value
                    )
                    release_fn(d3d_obj.value)
                except Exception:
                    pass
        self._d3d11_device = None
        self._d3d11_context = None

        # Release GPU interop PBOs
        if self._pbo_color is not None and self._cuda_gl:
            try:
                self._cuda_gl.unregister_resource(self._cuda_res_color)
                self._cuda_gl.unregister_resource(self._cuda_res_depth)
                glDeleteBuffers(2, [self._pbo_color, self._pbo_depth])
            except Exception:
                pass
        self._pbo_color = self._pbo_depth = None

        for tex in (self._overlay_tex, self._depth_osd_tex, self._screen_osd_tex,
                    self._preset_osd_tex, self._brand_osd_tex, self._seat_adjust_osd_tex,
                    self._panorama_tex, self.color_tex, self.depth_tex):
            if tex:
                try:
                    tex.release()
                except Exception:
                    pass
        self._overlay_tex = self._depth_osd_tex = self._screen_osd_tex = self._preset_osd_tex = self._brand_osd_tex = self._seat_adjust_osd_tex = None
        self._panorama_tex = None
        self._panorama_tex_path = None
        if self._calib_tex:
            try:
                self._calib_tex.release()
            except Exception:
                pass
            self._calib_tex = None
        if self._help_tex:
            try:
                self._help_tex.release()
            except Exception:
                pass
            self._help_tex = None
        for attr in ('_panorama_vao', '_panorama_vbo', '_panorama_prog'):
            obj = getattr(self, attr, None)
            if obj:
                try:
                    obj.release()
                except Exception:
                    pass
                setattr(self, attr, None)
        self.color_tex = self.depth_tex = None

        # Release controller model GL resources
        for prims in (self._ctrl_prims_l, self._ctrl_prims_r):
            for prim in prims:
                for key in ('vao', 'vbo', 'ibo'):
                    obj = prim.get(key)
                    if obj:
                        try:
                            obj.release()
                        except Exception:
                            pass
        self._ctrl_prims_l.clear()
        self._ctrl_prims_r.clear()
        for tex in self._ctrl_tex_cache.values():
            try:
                tex.release()
            except Exception:
                pass
        self._ctrl_tex_cache.clear()
        if self._controller_prog:
            try:
                self._controller_prog.release()
            except Exception:
                pass
            self._controller_prog = None
        self._grip_mat_l = None
        self._grip_mat_r = None

        for swapchain in self._xr_swapchains.values():
            try:
                xr.destroy_swapchain(swapchain)
            except Exception:
                pass
        self._xr_swapchains.clear()

        for space_attr in ("_aim_space_l", "_aim_space_r", "_grip_space_l", "_grip_space_r", "_xr_space"):
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

def _run_standalone_test():
    # Standalone smoke test: feed the viewer a single blank-white RGB frame
    # (with a zero depth map) so you can put on the headset and verify rendering,
    # controller input, keyboard, etc. without needing the
    # full main.py capture pipeline running.
    if not OPENXR_AVAILABLE:
        print("[TEST] pyopenxr not available cannot run standalone test")
        sys.exit(1)

    import queue as _q
    W, H = 1280, 720
    white_rgb = np.full((H, W, 3), 255, dtype=np.uint8)
    zero_depth = np.zeros((H, W), dtype=np.float32)

    depth_q = _q.Queue(maxsize=2)
    # Pre-seed with one frame so the run loop has something to display immediately.
    depth_q.put((white_rgb, zero_depth, time.perf_counter()))

    viewer = OpenXRViewer(
        frame_size=(W, H),
        fps=60,
        depth_q=depth_q,
        show_fps=True,
    )

    try:
        viewer.run(first_rgb=white_rgb, first_depth=zero_depth)
    except KeyboardInterrupt:
        print("[TEST] Interrupted")
    finally:
        viewer.cleanup()


if __name__ == "__main__":
    _run_standalone_test()
