# xrviewer_final.py
# Desktop2Stereo OpenXR viewer: no-room profile with built-in screen effects.

from xrviewer_core import *


class ScreenEffectsMixin:
    """Screen effect layer for the normal no-room viewer.

    The environment viewer does not inherit this mixin, so room mode keeps a
    plain screen and does not pay the maintenance cost of no-room visual effects.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._screen_effects_enabled = bool(kwargs.get('screen_effects_enabled', True))
        self._metallic_border_enabled = bool(kwargs.get('metallic_border_enabled', True))
        self._glow_enabled = bool(kwargs.get('glow_enabled', True))
        self._shadow_enabled = bool(kwargs.get('shadow_enabled', True))
        self._ground_light_enabled = bool(kwargs.get('ground_light_enabled', False))

        self._border_alpha = max(float(getattr(self, '_border_alpha', 0.0)), 0.0)
        self._glow_intensity = float(kwargs.get('glow_intensity', 0.22))
        self._glow_width_m = float(kwargs.get('glow_width_m', 0.035))
        self._glow_ref_screen = float(kwargs.get('glow_ref_screen', 2.4))
        self._glow_color = tuple(kwargs.get('glow_color', (0.30, 0.55, 1.0)))
        self._glow_target_color = self._glow_color
        self._shadow_opacity = float(kwargs.get('shadow_opacity', 0.22))
        self._ground_light_color = tuple(kwargs.get('ground_light_color', (0.25, 0.45, 1.0)))
        self._ground_light_intensity = float(kwargs.get('ground_light_intensity', 0.10))
        self._breath_dx = self._breath_dy = self._breath_dz = 0.0

    def _make_key_callback(self):
        base_cb = super()._make_key_callback()
        viewer = self

        def _cb(window, key, scancode, action, mods):
            base_cb(window, key, scancode, action, mods)
            if action not in (glfw.PRESS, glfw.REPEAT):
                return
            # Normal-mode visual effect shortcuts. These do not exist in env mode.
            if key == glfw.KEY_B:
                viewer._metallic_border_enabled = not viewer._metallic_border_enabled
                viewer._border_alpha = 1.0
                viewer._border_idle_t = time.perf_counter()
            elif key == glfw.KEY_V:
                viewer._glow_enabled = not viewer._glow_enabled
            elif key == glfw.KEY_H:
                viewer._shadow_enabled = not viewer._shadow_enabled
            elif key == glfw.KEY_J:
                viewer._ground_light_enabled = not viewer._ground_light_enabled
        return _cb

    def _screen_effect_model(self, width, height, z_offset=0.0, y_offset=0.0):
        sx = width / 2.0
        sy = height / 2.0
        cy = math.cos(self.screen_yaw);   sy_ = math.sin(self.screen_yaw)
        cp = math.cos(self.screen_pitch); sp = math.sin(self.screen_pitch)
        cr = math.cos(self.screen_roll);  sr = math.sin(self.screen_roll)
        rot_y = np.array([[ cy, 0, sy_, 0], [0, 1, 0, 0], [-sy_, 0, cy, 0], [0, 0, 0, 1]], dtype='f4')
        rot_x = np.array([[1, 0, 0, 0], [0, cp, -sp, 0], [0, sp, cp, 0], [0, 0, 0, 1]], dtype='f4')
        rot_z = np.array([[cr, -sr, 0, 0], [sr,  cr, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype='f4')
        R = rot_y @ rot_x @ rot_z
        S = np.diag([sx, sy, 1.0, 1.0]).astype('f4')
        T = np.eye(4, dtype='f4')
        T[0, 3] = self.screen_pan_x + getattr(self, '_breath_dx', 0.0)
        T[1, 3] = self.screen_pan_y + y_offset + getattr(self, '_breath_dy', 0.0)
        T[2, 3] = -self.screen_distance + z_offset + getattr(self, '_breath_dz', 0.0)
        return T @ R @ S

    def _render_screen_background_effects(self, mgl_fbo, vp_mat):
        if not getattr(self, '_screen_effects_enabled', True):
            return
        if self.screen_height is None:
            return
        self._render_shadow(mgl_fbo, vp_mat)
        self._render_ground_light(mgl_fbo, vp_mat)
        self._render_glow(mgl_fbo, vp_mat)

    def _render_screen_foreground_effects(self, mgl_fbo, vp_mat):
        if not getattr(self, '_screen_effects_enabled', True):
            return
        if self.screen_height is None:
            return
        self._render_metallic_border(mgl_fbo, vp_mat)

    def _render_glow(self, mgl_fbo, vp_mat):
        if not getattr(self, '_glow_enabled', True):
            return
        if getattr(self, '_glow_prog', None) is None or getattr(self, '_glow_vao', None) is None:
            return
        intensity = float(getattr(self, '_glow_intensity', 0.0))
        if intensity <= 0.0:
            return

        screen_long = max(self.screen_width, self.screen_height)
        glow_scale = screen_long / max(float(getattr(self, '_glow_ref_screen', 2.4)), 1e-6)
        glow_width = float(getattr(self, '_glow_width_m', 0.035)) * glow_scale
        glow_margin = glow_width * 6.0
        glow_w = self.screen_width + 2.0 * glow_margin
        glow_h = self.screen_height + 2.0 * glow_margin
        uv_glow_width = glow_width / max(glow_w, glow_h, 1e-6)

        self.ctx.depth_mask = False
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        model = self._screen_effect_model(glow_w, glow_h, z_offset=-0.002)
        mvp = vp_mat @ model
        self._glow_prog['u_mvp'].write(mvp.T.astype('f4').tobytes())
        self._glow_prog['u_screen_half'].value = (self.screen_width / glow_w / 2.0, self.screen_height / glow_h / 2.0)
        self._glow_prog['u_glow_color'].value = tuple(getattr(self, '_glow_color', (0.30, 0.55, 1.0)))
        self._glow_prog['u_glow_width'].value = uv_glow_width
        self._glow_prog['u_glow_intensity'].value = intensity
        self._glow_vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.depth_mask = True

    def _render_shadow(self, mgl_fbo, vp_mat):
        if not getattr(self, '_shadow_enabled', True):
            return
        if getattr(self, '_shadow_prog', None) is None or getattr(self, '_shadow_vao', None) is None:
            return
        opacity = float(getattr(self, '_shadow_opacity', 0.0))
        if opacity <= 0.0:
            return
        shadow_w = self.screen_width * 1.4
        shadow_h = self.screen_height * 0.35
        y_off = -(self.screen_height / 2.0 + shadow_h / 2.0)
        self.ctx.depth_mask = False
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        model = self._screen_effect_model(shadow_w, shadow_h, z_offset=-0.004, y_offset=y_off)
        mvp = vp_mat @ model
        self._shadow_prog['u_mvp'].write(mvp.T.astype('f4').tobytes())
        self._shadow_prog['u_opacity'].value = opacity
        self._shadow_vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.depth_mask = True

    def _render_ground_light(self, mgl_fbo, vp_mat):
        if not getattr(self, '_ground_light_enabled', False):
            return
        if getattr(self, '_ground_prog', None) is None or getattr(self, '_ground_vao', None) is None:
            return
        intensity = float(getattr(self, '_ground_light_intensity', 0.0))
        if intensity <= 0.0:
            return
        ground_w = self.screen_width * 1.5
        ground_h = self.screen_height * 0.45
        y_off = -(self.screen_height / 2.0 + ground_h / 2.0)
        self.ctx.depth_mask = False
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        model = self._screen_effect_model(ground_w, ground_h, z_offset=-0.006, y_offset=y_off)
        mvp = vp_mat @ model
        self._ground_prog['u_mvp'].write(mvp.T.astype('f4').tobytes())
        self._ground_prog['u_color'].value = tuple(getattr(self, '_ground_light_color', (0.25, 0.45, 1.0)))
        self._ground_prog['u_intensity'].value = intensity
        self._ground_vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.depth_mask = True

    def _render_metallic_border(self, mgl_fbo, vp_mat):
        if not getattr(self, '_metallic_border_enabled', True):
            return
        prog = getattr(self, '_metallic_border_prog', None)
        vao = getattr(self, '_metallic_border_vao', None)
        if prog is None or vao is None:
            return
        alpha = max(float(getattr(self, '_border_alpha', 1.0)), 0.35)
        self.ctx.depth_mask = False
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        border_w = self.screen_width * 1.012
        border_h = self.screen_height * 1.018
        model = self._screen_effect_model(border_w, border_h, z_offset=-0.001)
        mvp = vp_mat @ model
        prog['u_mvp'].write(mvp.T.astype('f4').tobytes())
        prog['u_color'].value = (0.65, 0.68, 0.72)
        prog['u_alpha'].value = alpha
        prog['u_border_uv'].value = (0.015, 0.022)
        vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.depth_mask = True

class OpenXRViewer(ScreenEffectsMixin, OpenXRViewerCore):
    """No-room viewer.

    Keeps normal viewing policy and screen effects separate from room/environment
    code. Environment hooks are explicitly disabled here.
    """

    ENVIRONMENT_MODE = False
    DEFAULT_ENVIRONMENT_MODEL = 'None'

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('environment_model', self.DEFAULT_ENVIRONMENT_MODEL)
        super().__init__(*args, **kwargs)

    def _discover_environment_models(self):
        return []

    def _reset_environment_profile_defaults(self):
        return None

    def _configure_environment_profile(self):
        self._environment_enabled = False
        self._environment_model = 'None'
        self._env_profile = {}
        self._env_model_path = None
        self._env_model_visible = False
        self._env_current_name = 'None'

    def _configure_profile_view_layout(self):
        return None

    def _screen_profile_value(self, key, default=None):
        return default

    def _environment_screen_locked(self):
        return False

    def _apply_profile_view_pose_to_xr_space(self, views=None):
        return None

    def _recenter_profile_view_pose(self, views=None):
        return None

    def _auto_view_position_from_screen(self):
        return None

    def _apply_profile_screen_layout(self, *args, **kwargs):
        return None

    def _init_env_model(self):
        self._env_model_visible = False
        self._env_model_prims = []
        self._env_model_textures = []
        self._env_model_lights = []
        self._env_available_models = []
        self._env_current_name = 'None'

    def _switch_environment_model(self, direction=1):
        return None

    def _render_env_model(self, mgl_fbo, vp_mat, view_mat):
        return None


def _smoke_test(viewer_cls):
    if not OPENXR_AVAILABLE:
        print("[TEST] pyopenxr not available - cannot run standalone test")
        sys.exit(1)

    import queue as _q
    W, H = 1280, 720
    white_rgb = np.full((H, W, 3), 255, dtype=np.uint8)
    zero_depth = np.zeros((H, W), dtype=np.float32)

    depth_q = _q.Queue(maxsize=2)
    depth_q.put((white_rgb, zero_depth, time.perf_counter()))

    viewer = viewer_cls(frame_size=(W, H), fps=60, depth_q=depth_q, show_fps=True)
    try:
        viewer.run(first_rgb=white_rgb, first_depth=zero_depth)
    except KeyboardInterrupt:
        print("[TEST] Interrupted")
    finally:
        viewer.cleanup()


if __name__ == "__main__":
    _smoke_test(OpenXRViewer)
