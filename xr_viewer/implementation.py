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
from .frame import FrameMixin
from .xr_session import XRSessionMixin
from .crop import CropMixin
from .screen import ScreenMixin
from .laser import LaserMixin
from .effects import EffectsMixin
from .input_handler import InputHandlerMixin

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


class OpenXRViewer(
    D3D11BackendMixin, EnvironmentMixin, OverlayMixin,
    FrameMixin, XRSessionMixin, CropMixin, ScreenMixin,
    LaserMixin, EffectsMixin, InputHandlerMixin,
):
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
        self._last_click_ts       = 0.0   # perf_counter of last emitted left click
        self._last_click_px       = None  # (px, py) of last emitted left click

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
        try:
            from utils import OS_NAME, crop_icon
            if OS_NAME != "Darwin":
                glfw.set_window_icon(self.window, 1, [crop_icon(Image.open("icon2.ico"))])
        except Exception as e:
            print(f"[OpenXRViewer] window icon load failed: {e}")
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
        # 4 walls, each an 8x8 grid strip (see _build_flat_frost_verts):
        # per-wall verts = 8*2*(8+1) + (8-1)*2 = 158; total 632 verts * 5f * 4B
        # = 12640 B. Reserve ~1.3x headroom.
        self._frost_glow_vbo = self.ctx.buffer(reserve=16384, dynamic=True)
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
