# xrviewer_env.py
# Desktop2Stereo OpenXR viewer: room/environment profile.
# Shared runtime/rendering code is in xrviewer_core.py; room-specific code lives here.

from xrviewer_core import *


class OpenXRViewer(OpenXRViewerCore):
    """Room/environment viewer.

    This class keeps the environment-specific behavior separate from the normal
    no-room viewer: room discovery, profile.json layout, GLB room loading,
    environment switching, and environment rendering.
    """

    ENVIRONMENT_MODE = True
    DEFAULT_ENVIRONMENT_MODEL = 'Default'

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('environment_model', self.DEFAULT_ENVIRONMENT_MODEL)
        super().__init__(*args, **kwargs)

    def _discover_environment_models(self):
        """Return room folders that can be switched at runtime."""
        models = []
        root = getattr(self, '_environment_root', None)
        if not root or not os.path.isdir(root):
            return models
        try:
            for name in sorted(os.listdir(root), key=lambda v: v.lower()):
                room_dir = os.path.join(root, name)
                if not os.path.isdir(room_dir):
                    continue
                if os.path.isfile(os.path.join(room_dir, 'profile.json')) or os.path.isfile(os.path.join(room_dir, 'environment.glb')):
                    models.append(name)
        except Exception:
            pass
        selected = (getattr(self, '_environment_model', '') or '').strip()
        if selected and selected.lower() != 'default' and selected not in models:
            models.insert(0, selected)
        return models


    def _reset_environment_profile_defaults(self):
        """Reset runtime room settings before applying another profile."""
        base = getattr(self, '_env_base_settings', None)
        if not isinstance(base, dict):
            return
        self._env_model_pos = list(base['model_pos'])
        self._env_model_rot = list(base['model_rot'])
        self._env_model_scale = list(base['model_scale'])
        self._env_head_light_color = tuple(base['head_light_color'])
        self._env_ambient_color = tuple(base['ambient_color'])
        self._env_fallback_dir = np.array(base['fallback_dir'], dtype=np.float32)
        self._env_fallback_dir = self._env_fallback_dir / (np.linalg.norm(self._env_fallback_dir) + 1e-8)
        self._env_fallback_dir_color = tuple(base['fallback_dir_color'])
        self._env_fill_lights = list(base['fill_lights'])
        self._env_exposure = float(base['exposure'])
        self._env_gamma = float(base['gamma'])
        self._env_emissive_strength = float(base['emissive_strength'])
        self._env_khr_light_scale = float(base['khr_light_scale'])
        self._env_render_quality = str(base['render_quality'])
        self._env_shading_mode = str(base['shading_mode'])
        self._env_texture_anisotropy = float(base['texture_anisotropy'])
        self._env_perf_log = bool(base.get('perf_log', False))
        self._xr_render_scale = float(base['xr_render_scale'])


    def _configure_environment_profile(self):
        """Resolve the selected room folder and apply optional profile settings."""
        self._reset_environment_profile_defaults()
        selected = (self._environment_model or 'Default').strip() or 'Default'
        if selected.lower() == 'none':
            self._environment_enabled = False
            self._environment_model = 'None'
            self._env_profile = {}
            self._env_model_path = None
            self._env_model_visible = False
            return
        self._environment_enabled = True
        root = self._environment_root
        room_dir = root if selected.lower() == 'default' else os.path.join(root, selected)
        profile_path = os.path.join(room_dir, 'profile.json')
        profile = {}

        if os.path.exists(profile_path):
            try:
                with open(profile_path, 'r', encoding='utf-8-sig') as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    profile = loaded
            except Exception as exc:
                print(f"[OpenXRViewer] Failed to read environment profile {profile_path}: {exc}")

        glb_name = str(profile.get('glb', 'environment.glb') or 'environment.glb')
        glb_path = glb_name if os.path.isabs(glb_name) else os.path.join(room_dir, glb_name)
        if not os.path.exists(glb_path) and selected.lower() != 'default':
            fallback = os.path.join(root, 'environment.glb')
            print(f"[OpenXRViewer] Environment '{selected}' missing GLB, fallback to Default")
            selected = 'Default'
            room_dir = root
            glb_path = fallback
            profile = {}

        self._environment_model = selected
        self._env_profile = profile
        self._env_model_path = glb_path

        self._env_model_pos = self._profile_vec3(profile, ('model_position', 'position'), self._env_model_pos)
        self._env_model_scale = self._profile_vec3(profile, ('model_scale', 'scale'), self._env_model_scale)
        rot_deg = profile.get('model_rotation_deg', profile.get('rotation_deg'))
        if isinstance(rot_deg, (list, tuple)) and len(rot_deg) >= 3:
            try:
                self._env_model_rot = [math.radians(float(rot_deg[0])),
                                       math.radians(float(rot_deg[1])),
                                       math.radians(float(rot_deg[2]))]
            except (TypeError, ValueError):
                pass
        else:
            self._env_model_rot = self._profile_vec3(profile, ('model_rotation', 'rotation'), self._env_model_rot)

        for key, attr in (
            ('env_exposure', '_env_exposure'),
            ('env_gamma', '_env_gamma'),
            ('env_emissive_strength', '_env_emissive_strength'),
            ('env_khr_light_scale', '_env_khr_light_scale'),
            ('khr_light_scale', '_env_khr_light_scale'),
        ):
            if key in profile:
                try:
                    setattr(self, attr, float(profile[key]))
                except (TypeError, ValueError):
                    pass

        quality = profile.get('env_render_quality', profile.get('render_quality'))
        if isinstance(quality, str):
            quality_l = quality.strip().lower()
            if quality_l in ('fast', 'balanced', 'quality'):
                self._env_render_quality = quality_l
        shading_mode = profile.get('env_shading_mode', profile.get('shading_mode'))
        if isinstance(shading_mode, str):
            shading_mode_l = shading_mode.strip().lower()
            if shading_mode_l in ('pbr', 'preview'):
                self._env_shading_mode = shading_mode_l
        if 'env_perf_log' in profile:
            self._env_perf_log = bool(profile.get('env_perf_log'))
        if 'env_texture_anisotropy' in profile:
            try:
                self._env_texture_anisotropy = max(1.0, float(profile['env_texture_anisotropy']))
            except (TypeError, ValueError):
                pass
        if 'xr_render_scale' in profile:
            try:
                self._xr_render_scale = max(0.5, min(1.0, float(profile['xr_render_scale'])))
            except (TypeError, ValueError):
                pass

        self._env_head_light_color = tuple(self._profile_vec3(
            profile, ('env_head_light_color', 'head_light_color'), self._env_head_light_color))
        self._env_ambient_color = tuple(self._profile_vec3(
            profile, ('env_ambient_color', 'ambient_color'), self._env_ambient_color))
        self._env_fallback_dir = np.array(self._profile_vec3(
            profile, ('env_directional_dir', 'directional_dir'), self._env_fallback_dir), dtype=np.float32)
        self._env_fallback_dir = self._env_fallback_dir / (np.linalg.norm(self._env_fallback_dir) + 1e-8)
        self._env_fallback_dir_color = tuple(self._profile_vec3(
            profile, ('env_directional_color', 'directional_color'), self._env_fallback_dir_color))

        fill_lights = profile.get('env_fill_lights', profile.get('fallback_lights'))
        if isinstance(fill_lights, list):
            self._env_fill_lights = fill_lights

        baked_lightmap = profile.get('baked_lightmap', profile.get('baked', None))
        baked_label = f" baked_lightmap={bool(baked_lightmap)}" if baked_lightmap is not None else ""
        print(
            f"[OpenXRViewer] Environment: {self._environment_model} ({self._env_model_path}) "
            f"quality={self._env_render_quality} shading={self._env_shading_mode} "
            f"xr_scale={self._xr_render_scale:.2f}{baked_label}"
        )


    def _configure_profile_view_layout(self):
        """Cache optional room-specific viewer and screen layout settings."""
        profile = self._env_profile if isinstance(self._env_profile, dict) else {}
        view_pose = profile.get('view_pose', profile.get('camera', {}))
        screen = profile.get('screen', {})
        self._view_pose_profile = view_pose if isinstance(view_pose, dict) else {}
        self._screen_profile = screen if isinstance(screen, dict) else {}
        if self._screen_profile:
            print(f"[OpenXRViewer] Profile screen layout enabled: {self._environment_model}")


    def _screen_profile_value(self, key, default=None):
        screen = getattr(self, '_screen_profile', {}) or {}
        return screen.get(key, default)


    def _environment_screen_locked(self):
        screen = getattr(self, '_screen_profile', {}) or {}
        return isinstance(screen, dict) and bool(screen)


    def _head_model_mat4_from_views(self, views):
        if not views or len(views) < 2 or views[0] is None or views[1] is None:
            return None
        head_mat = xr_pose_to_model_mat4(views[0].pose)
        try:
            p0 = views[0].pose.position
            p1 = views[1].pose.position
            head_mat[:3, 3] = np.array([
                (p0.x + p1.x) / 2.0,
                (p0.y + p1.y) / 2.0,
                (p0.z + p1.z) / 2.0,
            ], dtype=np.float32)
        except Exception:
            return None
        return head_mat


    def _level_head_model_mat4(self, head_mat):
        """Keep head position and yaw, dropping pitch/roll so the room stays level."""
        if head_mat is None:
            return None
        pos = head_mat[:3, 3].copy()
        forward = -head_mat[:3, 2].astype(np.float32)
        forward[1] = 0.0
        norm = float(np.linalg.norm(forward))
        if norm < 1e-6:
            yaw = 0.0
        else:
            forward = forward / norm
            yaw = math.atan2(-float(forward[0]), -float(forward[2]))
        leveled = euler_to_mat4(yaw, 0.0, 0.0).astype(np.float32)
        leveled[:3, 3] = pos
        return leveled


    def _apply_profile_view_pose_to_xr_space(self, views):
        if self._xr_profile_space_applied:
            return False

        self._xr_profile_space_applied = True
        view = getattr(self, '_view_pose_profile', {}) or {}
        if not isinstance(view, dict) or not view:
            return False

        pos_keys = ('position', 'camera_position', 'viewer_position')
        rot_deg_keys = ('rotation_deg', 'camera_rotation_deg', 'viewer_rotation_deg')
        rot_rad_keys = ('rotation', 'camera_rotation', 'viewer_rotation')
        has_pos = any(key in view for key in pos_keys)
        has_rot = any(key in view for key in rot_deg_keys + rot_rad_keys)
        auto_center = bool(view.get('auto_center_on_screen', False))
        if not (has_pos or has_rot or auto_center):
            return False
        if self._xr_session is None or self._xr_space is None or self._xr_ref_space_type is None:
            return False

        raw_head = self._head_model_mat4_from_views(views)
        if raw_head is None:
            self._xr_profile_space_applied = False
            return False

        desired_head = raw_head.copy()
        if auto_center:
            auto_pos = self._auto_view_position_from_screen(view, has_rot, rot_deg_keys, rot_rad_keys)
            if auto_pos is None and has_pos:
                auto_pos = self._profile_vec3(view, pos_keys, raw_head[:3, 3].tolist())
            if auto_pos is not None:
                desired_head[:3, 3] = np.array(auto_pos, dtype=np.float32)
        elif has_pos:
            desired_head[:3, 3] = np.array(
                self._profile_vec3(view, pos_keys, raw_head[:3, 3].tolist()),
                dtype=np.float32,
            )
        if has_rot:
            rot = self._profile_rotation_rad(view, rot_deg_keys, rot_rad_keys, [0.0, 0.0, 0.0])
            desired_head[:3, :3] = euler_to_mat4(*rot)[:3, :3]

        try:
            current_space_in_ref = getattr(self, '_xr_space_pose_in_ref', np.eye(4, dtype=np.float32))
            reference_head = current_space_in_ref @ raw_head
            if auto_center:
                reference_head = self._level_head_model_mat4(reference_head)
            space_in_ref = reference_head @ np.linalg.inv(desired_head)
            new_space = xr.create_reference_space(
                self._xr_session,
                xr.ReferenceSpaceCreateInfo(
                    reference_space_type=self._xr_ref_space_type,
                    pose_in_reference_space=mat4_to_xr_posef(space_in_ref.astype(np.float32)),
                ),
            )
        except Exception as exc:
            print(f"[OpenXRViewer] Failed to apply profile view_pose to XrSpace: {exc}")
            return False

        old_space = self._xr_space
        self._xr_space = new_space
        self._xr_space_pose_in_ref = space_in_ref.astype(np.float32)
        if old_space is not None:
            try:
                xr.destroy_space(old_space)
            except Exception:
                pass

        pos = [round(float(v), 4) for v in desired_head[:3, 3]]
        print(f"[OpenXRViewer] Applied profile view_pose to XrSpace: position={pos}")
        return True


    def _recenter_profile_view_pose(self):
        """Re-apply profile view_pose, used by Home/Menu long press."""
        view = getattr(self, '_view_pose_profile', {}) or {}
        if not isinstance(view, dict) or not view.get('auto_center_on_screen', False):
            return False
        views = getattr(self, '_last_located_views', None)
        if not views or views[0] is None or views[1] is None:
            return False
        self._xr_profile_space_applied = False
        applied = self._apply_profile_view_pose_to_xr_space(views)
        if applied:
            print("[OpenXRViewer] Home recentered view_pose to screen center")
        return applied


    def _auto_view_position_from_screen(self, view, has_view_rot, rot_deg_keys, rot_rad_keys):
        """Compute a viewer position centered on the configured screen."""
        screen = getattr(self, '_screen_profile', {}) or {}
        if not isinstance(screen, dict) or not screen:
            return None

        position = screen.get('position', screen.get('screen_position'))
        if not isinstance(position, (list, tuple)) or len(position) < 3:
            return None
        try:
            screen_pos = np.array(
                [float(position[0]), float(position[1]), float(position[2])],
                dtype=np.float32,
            )
        except (TypeError, ValueError):
            return None

        width = self._profile_float(screen, ('width', 'screen_width'), self.screen_width)
        ratio = self._profile_float(view, ('distance_width_ratio', 'view_distance_width_ratio'), 0.6)
        distance = self._profile_float(view, ('distance', 'view_distance'), width * ratio)
        distance = max(0.05, float(distance))

        if has_view_rot:
            yaw, pitch, _roll = self._profile_rotation_rad(
                view, rot_deg_keys, rot_rad_keys, [0.0, 0.0, 0.0]
            )
            cp = math.cos(pitch)
            forward = np.array(
                [-math.sin(yaw) * cp, math.sin(pitch), -math.cos(yaw) * cp],
                dtype=np.float32,
            )
        else:
            screen_rot = self._profile_rotation_rad(
                screen,
                ('rotation_deg', 'screen_rotation_deg'),
                ('rotation', 'screen_rotation'),
                [self.screen_yaw, self.screen_pitch, self.screen_roll],
            )
            screen_normal = euler_to_mat4(*screen_rot)[:3, 2].astype(np.float32)
            forward = -screen_normal

        norm = float(np.linalg.norm(forward))
        if norm < 1e-6:
            return None
        forward = forward / norm

        pos = screen_pos - forward * distance
        right = np.cross(forward, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        right_norm = float(np.linalg.norm(right))
        if right_norm > 1e-6:
            right = right / right_norm
            pos += right * self._profile_float(view, ('horizontal_offset', 'x_offset'), 0.0)
        pos[1] += self._profile_float(view, ('vertical_offset', 'y_offset'), 0.0)
        return [float(pos[0]), float(pos[1]), float(pos[2])]


    def _apply_profile_screen_layout(self, show_border=False):
        """Apply fixed room-specific screen layout from profile.json.

        This does not lock the physical headset pose.  OpenXR head tracking stays
        active; the profile only defines where the virtual screen is placed in
        the room.
        """
        screen = getattr(self, '_screen_profile', {}) or {}
        if not isinstance(screen, dict) or not screen:
            return False

        self._reset_orientation_offsets()
        width = self._profile_float(screen, ('width', 'screen_width'), self.screen_width)
        self.screen_width = max(0.05, width)
        self._screen_ref_size = self.screen_width
        self.screen_height = None


        rotation = self._profile_rotation_rad(
            screen,
            ('rotation_deg', 'screen_rotation_deg'),
            ('rotation', 'screen_rotation'),
            [self.screen_yaw, self.screen_pitch, self.screen_roll],
        )

        position = screen.get('position', screen.get('screen_position'))
        if isinstance(position, (list, tuple)) and len(position) >= 3:
            try:
                x, y, z = float(position[0]), float(position[1]), float(position[2])
                self.screen_pan_x = x
                self.screen_pan_y = y
                self.screen_distance = max(0.05, -z)
                self.screen_yaw, self.screen_pitch, self.screen_roll = rotation
            except (TypeError, ValueError):
                return False
        else:
            view = getattr(self, '_view_pose_profile', {}) or {}
            view_pos = self._profile_vec3(
                view,
                ('position', 'camera_position', 'viewer_position'),
                [0.0, float(self._initial_head_y), 0.0],
            )
            view_rot = self._profile_rotation_rad(
                view,
                ('rotation_deg', 'camera_rotation_deg', 'viewer_rotation_deg'),
                ('rotation', 'camera_rotation', 'viewer_rotation'),
                [0.0, 0.0, 0.0],
            )
            yaw, pitch, _roll = view_rot
            distance = self._profile_float(screen, ('distance', 'screen_distance'), self.screen_distance)
            cp = math.cos(pitch)
            fx = -math.sin(yaw) * cp
            fy = math.sin(pitch)
            fz = -math.cos(yaw) * cp
            self.screen_pan_x = float(view_pos[0] + fx * distance)
            self.screen_pan_y = float(view_pos[1] + fy * distance)
            self.screen_distance = max(0.05, -(float(view_pos[2]) + fz * distance))
            if 'rotation_deg' in screen or 'screen_rotation_deg' in screen or 'rotation' in screen or 'screen_rotation' in screen:
                self.screen_yaw, self.screen_pitch, self.screen_roll = rotation
            else:
                self.screen_yaw = math.atan2(-fx, -fz)
                self.screen_pitch = 0.0
                self.screen_roll = 0.0

        self._last_overlay_update = 0.0
        self._border_alpha = 0.0
        if self._keyboard_visible:
            self._anchor_keyboard_below_screen()
        return True


    def _build_env_model_mat4(self):
        sx, sy, sz = [float(v) for v in self._env_model_scale]
        yaw, pitch, roll = [float(v) for v in self._env_model_rot]
        cy, sy_ = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)

        scale = np.eye(4, dtype=np.float32)
        scale[0, 0], scale[1, 1], scale[2, 2] = sx, sy, sz
        ry = np.array([[cy, 0.0, sy_, 0.0],
                       [0.0, 1.0, 0.0, 0.0],
                       [-sy_, 0.0, cy, 0.0],
                       [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        rx = np.array([[1.0, 0.0, 0.0, 0.0],
                       [0.0, cp, -sp, 0.0],
                       [0.0, sp, cp, 0.0],
                       [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        rz = np.array([[cr, -sr, 0.0, 0.0],
                       [sr, cr, 0.0, 0.0],
                       [0.0, 0.0, 1.0, 0.0],
                       [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        trans = np.eye(4, dtype=np.float32)
        trans[:3, 3] = np.array(self._env_model_pos, dtype=np.float32)
        return trans @ ry @ rx @ rz @ scale


    def _transform_env_point(self, point, model_mat):
        p = np.array([float(point[0]), float(point[1]), float(point[2]), 1.0], dtype=np.float32)
        return (model_mat @ p)[:3]


    def _transform_env_direction(self, direction, model_mat):
        d = model_mat[:3, :3] @ np.array(direction, dtype=np.float32)
        return d / (np.linalg.norm(d) + 1e-8)


    def _env_light_range_scale(self):
        try:
            return max(abs(float(v)) for v in self._env_model_scale) or 1.0
        except Exception:
            return 1.0


    def _load_env_model(self, path):
        """Load a glTF environment model from *path*.

        Populates ``self._env_model_prims`` and ``self._env_model_tex_cache``.
        Textures use LINEAR_MIPMAP_LINEAR + mipmaps + 16x anisotropy.
        If the file is corrupt or resources cannot be allocated, this method
        fails silently (prints a warning) and leaves the primitive list empty.
        """
        prims_data = []
        textures = []
        try:
            prims_data, textures, env_lights = load_glb_model(path)
            if env_lights:
                self._scene_lights = env_lights
        except Exception as exc:
            print(f"[OpenXRViewer] Failed to load environment model {path}: {exc}")
            return

        _prefix = "env"
        try:
            # Upload textures. glTF sampler state belongs to textures[], not images[],
            # so cache by image id + sampler tuple.
            sampler_requests = set()
            for pd in prims_data:
                for tex_id_key, sampler_key in (
                    ('tex_id', 'base_sampler'),
                    ('normal_tex_id', 'normal_sampler'),
                    ('occlusion_tex_id', 'occlusion_sampler'),
                    ('mr_tex_id', 'mr_sampler'),
                    ('emissive_tex_id', 'emissive_sampler'),
                ):
                    tid = int(pd.get(tex_id_key, -1))
                    if tid >= 0:
                        sampler_requests.add((tid, normalize_gltf_sampler(pd.get(sampler_key))))
            for tid, sampler in sampler_requests:
                if tid < len(textures) and textures[tid] is not None:
                    cache_key = gltf_texture_cache_key(_prefix, tid, sampler)
                    h, w = textures[tid].shape[:2]
                    mtex = self.ctx.texture((w, h), 4, textures[tid].tobytes())
                    apply_gltf_sampler_to_texture(mtex, sampler)
                    mtex.build_mipmaps()
                    mtex.anisotropy = self._env_texture_anisotropy
                    self._env_model_tex_cache[cache_key] = mtex

            # Create VAOs (bound to _env_prog, no gl_FrontFacing discard)
            baked_lightmap = False
            if isinstance(getattr(self, '_env_profile', None), dict):
                baked_lightmap = bool(self._env_profile.get('baked_lightmap', self._env_profile.get('baked', False)))
            baked_uv1_forced = 0
            for pd in prims_data:
                if (
                    baked_lightmap
                    and pd.get('has_uv1', False)
                    and int(pd.get('occlusion_tex_id', -1)) >= 0
                    and int(pd.get('occlusion_texcoord', 0)) != 1
                ):
                    pd['occlusion_texcoord'] = 1
                    baked_uv1_forced += 1
                vbo = self.ctx.buffer(pd['vertices'].tobytes())
                tan_vbo = self.ctx.buffer(pd['tangent'].tobytes())
                ibo = self.ctx.buffer(pd['indices'].tobytes())
                vao = self.ctx.vertex_array(
                    self._env_prog,
                    [(vbo, '3f 3f 2f 2f', 'in_position', 'in_normal', 'in_uv', 'in_uv1'),
                     (tan_vbo, '4f', 'in_tangent')],
                    ibo,
                )
                base_color = pd.get('base_color', np.array([1.0, 1.0, 1.0], dtype=np.float32))
                emissive_factor = pd.get('emissive_factor', np.array([0.0, 0.0, 0.0], dtype=np.float32))
                base_alpha = float(pd.get('base_alpha', 1.0))
                alpha_mode = pd.get('alpha_mode', 'OPAQUE')
                vertices = pd.get('vertices')
                if isinstance(vertices, np.ndarray) and len(vertices) > 0:
                    sort_center_local = vertices[:, :3].mean(axis=0).astype(np.float32)
                else:
                    sort_center_local = np.zeros(3, dtype=np.float32)
                tex_key = (
                    gltf_texture_cache_key(_prefix, pd['tex_id'], pd.get('base_sampler'))
                    if pd['tex_id'] >= 0 else None
                )
                normal_tex_id = pd.get('normal_tex_id', -1)
                occlusion_tex_id = pd.get('occlusion_tex_id', -1)
                mr_tex_id = pd.get('mr_tex_id', -1)
                emissive_tex_id = pd.get('emissive_tex_id', -1)
                material_key = (
                    alpha_mode == 'BLEND',
                    tex_key or '',
                    normal_tex_id,
                    occlusion_tex_id,
                    mr_tex_id,
                    emissive_tex_id,
                    tuple(float(x) for x in base_color[:3]),
                    base_alpha,
                    float(pd.get('roughness_factor', 1.0)),
                    float(pd.get('metallic_factor', 0.0)),
                    tuple(float(x) for x in emissive_factor[:3]),
                    bool(pd.get('unlit', False)),
                    alpha_mode,
                    float(pd.get('alpha_cutoff', 0.5)),
                    tuple(float(x) for x in pd.get('tex_offset', np.array([0.0, 0.0], dtype=np.float32))[:2]),
                    tuple(float(x) for x in pd.get('tex_scale', np.array([1.0, 1.0], dtype=np.float32))[:2]),
                    float(pd.get('tex_rotation', 0.0)),
                )
                self._env_model_prims.append({
                    'vao': vao, 'vbo': vbo, 'tan_vbo': tan_vbo, 'ibo': ibo,
                    'tex_key': tex_key,
                    'render_mode': gltf_primitive_mode_to_moderngl(pd.get('primitive_mode', 4)),
                    'tri_count': len(pd['indices']) // 3,
                    'base_color': base_color,
                    'base_alpha': base_alpha,
                    'roughness_factor': pd.get('roughness_factor', 1.0),
                    'metallic_factor': pd.get('metallic_factor', 0.0),
                    'emissive_factor': emissive_factor,
                    'normal_tex_id': normal_tex_id,
                    'normal_sampler': pd.get('normal_sampler'),
                    'normal_texcoord': pd.get('normal_texcoord', 0),
                    'normal_scale': pd.get('normal_scale', 1.0),
                    'occlusion_tex_id': occlusion_tex_id,
                    'occlusion_sampler': pd.get('occlusion_sampler'),
                    'occlusion_texcoord': pd.get('occlusion_texcoord', 0),
                    'occlusion_strength': pd.get('occlusion_strength', 1.0),
                    'unlit': pd.get('unlit', False),
                    'alpha_mode': alpha_mode,
                    'alpha_cutoff': pd.get('alpha_cutoff', 0.5),
                    'mr_tex_id': mr_tex_id,
                    'mr_sampler': pd.get('mr_sampler'),
                    'mr_texcoord': pd.get('mr_texcoord', 0),
                    'emissive_tex_id': emissive_tex_id,
                    'emissive_sampler': pd.get('emissive_sampler'),
                    'emissive_texcoord': pd.get('emissive_texcoord', 0),
                    'double_sided': pd.get('double_sided', False),
                    'foliage_mode': pd.get('foliage_mode', False),
                    'sort_center_local': sort_center_local,
                    'base_texcoord': pd.get('base_texcoord', 0),
                    'tex_offset': pd.get('tex_offset', np.array([0.0, 0.0], dtype=np.float32)),
                    'tex_scale': pd.get('tex_scale', np.array([1.0, 1.0], dtype=np.float32)),
                    'tex_rotation': pd.get('tex_rotation', 0.0),
                    'material_key': material_key,
                })
            if self._env_shading_mode != 'preview':
                self._env_model_prims.sort(key=lambda prim: prim.get('material_key', ()))
            if baked_lightmap:
                occ_count = sum(1 for prim in self._env_model_prims if int(prim.get('occlusion_tex_id', -1)) >= 0)
                occ_uv1 = sum(1 for prim in self._env_model_prims if int(prim.get('occlusion_tex_id', -1)) >= 0 and int(prim.get('occlusion_texcoord', 0)) == 1)
                print(f"[OpenXRViewer] Baked lightmap primitives: occlusion={occ_count} uv1={occ_uv1}")
            if baked_uv1_forced:
                print(f"[OpenXRViewer] Baked lightmap forced occlusion texCoord=1 on {baked_uv1_forced} primitives")
        except Exception as exc:
            print(f"[OpenXRViewer] Failed to create environment model resources: {exc}")
            self._release_env_model_resources()


    def _release_env_model_resources(self):
        """Release current room GL resources before reloading or shutdown."""
        for prim in self._env_model_prims:
            for key in ('vao', 'vbo', 'tan_vbo', 'ibo'):
                obj = prim.get(key)
                if obj is not None:
                    try:
                        obj.release()
                    except Exception:
                        pass
        self._env_model_prims = []
        for tex in self._env_model_tex_cache.values():
            try:
                tex.release()
            except Exception:
                pass
        self._env_model_tex_cache = {}
        self._scene_lights = []
        self._env_model_visible = False


    def _generate_default_room(self):
        """Generate a simple room (floor, 4 walls, ceiling) procedurally."""
        W, H, D = 4.0, 3.0, 4.0
        import numpy as np
        faces = []
        faces.append((np.array([[-W,0,-D, 0,1,0, 0,0], [W,0,-D, 0,1,0, 1,0], [W,0,D, 0,1,0, 1,1], [-W,0,D, 0,1,0, 0,1]], dtype='f4'),
                      np.array([0,1,2, 0,2,3], dtype='u4'), (0.20, 0.20, 0.22)))
        faces.append((np.array([[-W,0,-D, 0,0,1, 0,0], [W,0,-D, 0,0,1, 1,0], [W,H,-D, 0,0,1, 1,1], [-W,H,-D, 0,0,1, 0,1]], dtype='f4'),
                      np.array([0,1,2, 0,2,3], dtype='u4'), (0.30, 0.30, 0.35)))
        faces.append((np.array([[-W,0,-D, 1,0,0, 0,0], [-W,0,D, 1,0,0, 1,0], [-W,H,D, 1,0,0, 1,1], [-W,H,-D, 1,0,0, 0,1]], dtype='f4'),
                      np.array([0,1,2, 0,2,3], dtype='u4'), (0.25, 0.25, 0.30)))
        faces.append((np.array([[W,0,-D, -1,0,0, 0,0], [W,H,-D, -1,0,0, 1,0], [W,H,D, -1,0,0, 1,1], [W,0,D, -1,0,0, 0,1]], dtype='f4'),
                      np.array([0,1,2, 0,2,3], dtype='u4'), (0.28, 0.28, 0.33)))
        faces.append((np.array([[-W,H,-D, 0,-1,0, 0,0], [-W,H,D, 0,-1,0, 1,0], [W,H,D, 0,-1,0, 1,1], [W,H,-D, 0,-1,0, 0,1]], dtype='f4'),
                      np.array([0,1,2, 0,2,3], dtype='u4'), (0.35, 0.35, 0.40)))
        for verts, idx, color in faces:
            verts = np.hstack([verts, verts[:, 6:8]]).astype('f4')
            vbo = self.ctx.buffer(verts.tobytes())
            # Dummy tangent: (1,0,0,1) -room faces have no normal map anyway
            dummy_tan = np.tile(np.array([1.0, 0.0, 0.0, 1.0], dtype='f4'), (verts.shape[0], 1))
            tan_vbo = self.ctx.buffer(dummy_tan.tobytes())
            ibo = self.ctx.buffer(idx.tobytes())
            vao = self.ctx.vertex_array(
                self._env_prog,
                [(vbo, '3f 3f 2f 2f', 'in_position', 'in_normal', 'in_uv', 'in_uv1'),
                 (tan_vbo, '4f', 'in_tangent')],
                ibo,
            )
            self._env_model_prims.append({
                'vao': vao, 'vbo': vbo, 'tan_vbo': tan_vbo, 'ibo': ibo,
                'tex_key': None, 'tri_count': 2, 'color': color,
                'base_color': np.array(color, dtype=np.float32),
                'base_alpha': 1.0,
                'roughness_factor': 1.0,
                'metallic_factor': 0.0,
                'emissive_factor': np.array([0.0, 0.0, 0.0], dtype=np.float32),
                'normal_tex_id': -1,
                'normal_texcoord': 0,
                'normal_scale': 1.0,
                'occlusion_tex_id': -1,
                'occlusion_texcoord': 0,
                'occlusion_strength': 1.0,
                'unlit': False,
                'alpha_mode': 'OPAQUE',
                'alpha_cutoff': 0.5,
                'mr_tex_id': -1,
                'mr_texcoord': 0,
                'emissive_tex_id': -1,
                'emissive_texcoord': 0,
                'double_sided': False,
                'base_texcoord': 0,
                'render_mode': moderngl.TRIANGLES,
                'tex_offset': np.array([0.0, 0.0], dtype=np.float32),
                'tex_scale': np.array([1.0, 1.0], dtype=np.float32),
                'tex_rotation': 0.0,
            })
        self._env_model_visible = True
        print(f'[OpenXRViewer] Default room generated ({len(faces)} faces)')


    def _init_env_model(self):
        """Try loading environment.glb, fall back to built-in room."""
        if not getattr(self, '_environment_enabled', True):
            self._env_model_visible = False
            return
        path = self._env_model_path or os.path.join(self._environment_root, 'environment.glb')
        if os.path.exists(path):
            self._load_env_model(path)
            if self._env_model_prims:
                self._env_model_visible = True
                print(f"[OpenXRViewer] Environment model loaded ({len(self._env_model_prims)} primitives): {self._environment_model}")
                return
        self._generate_default_room()


    def _switch_environment_model(self, model_name=None):
        """Switch to another room environment during runtime."""
        if not getattr(self, '_environment_enabled', True):
            return False
        models = self._available_environment_models or self._discover_environment_models()
        self._available_environment_models = models
        if not models:
            return False

        current = (self._environment_model or '').strip()
        if model_name is None:
            try:
                idx = models.index(current)
            except ValueError:
                idx = -1
            model_name = models[(idx + 1) % len(models)]
        if model_name == current and self._env_model_prims:
            return False

        print(f"[OpenXRViewer] Switching environment to: {model_name}")
        self._release_env_model_resources()
        self._environment_model = model_name
        self._configure_environment_profile()
        self._configure_profile_view_layout()
        self._init_env_model()
        self._apply_profile_screen_layout(show_border=True)
        self._xr_profile_space_applied = False
        if not self._recenter_profile_view_pose():
            self._reset_screen_to_default(show_border=True)
        return True


    def _render_screen_background_effects(self, mgl_fbo, vp_mat):
        return None

    def _render_screen_foreground_effects(self, mgl_fbo, vp_mat):
        return None

    def _render_env_model(self, mgl_fbo, vp_mat, view_mat):
        """Render the glTF environment model in world space."""
        if not self._env_model_visible or not self._env_model_prims:
            return
        perf_t0 = time.perf_counter() if self._env_perf_log else 0.0

        model_mat = self._build_env_model_mat4().astype('f4')
        view_inv = np.linalg.inv(view_mat)
        cam_pos = view_inv[:3, 3].astype('f4')

        self._env_prog['u_mvp'].write(vp_mat.astype('f4').T.tobytes())
        self._env_prog['u_model'].write(model_mat.T.tobytes())
        self._env_prog['u_camera_pos'].write(cam_pos.tobytes())
        self._env_prog['u_light_color'].value = self._env_head_light_color
        self._env_prog['u_ambient_color'].value = self._env_ambient_color
        self._env_prog['u_env_exposure'].value = self._env_exposure
        self._env_prog['u_env_gamma'].value = self._env_gamma
        self._env_prog['u_emissive_strength'].value = self._env_emissive_strength
        self._env_prog['u_shading_mode'].value = 1 if self._env_shading_mode == 'preview' else 0
        profile = getattr(self, '_env_profile', {}) or {}
        baked_lightmap = bool(profile.get('baked_lightmap', profile.get('baked', False))) if isinstance(profile, dict) else False
        self._env_prog['u_baked_lightmap'].value = 1 if baked_lightmap else 0

        directional = next((light for light in self._scene_lights if light.get('type') == 'directional'), None)
        if directional:
            light_dir = self._transform_env_direction(directional['direction'], model_mat)
            self._env_prog['u_light_dir'].value = (
                float(light_dir[0]), float(light_dir[1]), float(light_dir[2])
            )
            color = directional['color'] * directional['intensity'] * self._env_khr_light_scale
            self._env_prog['u_light_intensity'].value = (
                float(color[0]), float(color[1]), float(color[2])
            )
        else:
            light_dir = self._transform_env_direction(self._env_fallback_dir, model_mat)
            self._env_prog['u_light_dir'].value = (
                float(light_dir[0]), float(light_dir[1]), float(light_dir[2])
            )
            self._env_prog['u_light_intensity'].value = self._env_fallback_dir_color

        fill_specs = []
        range_scale = self._env_light_range_scale()
        for light in self._scene_lights:
            if light.get('type') not in ('point', 'spot') or 'position' not in light:
                continue
            color = light['color'] * light['intensity'] * self._env_khr_light_scale
            light_range = float(light.get('range', 0.0) or 0.0)
            fill_specs.append((
                self._transform_env_point(light['position'], model_mat),
                color,
                (light_range if light_range > 0.0 else 4.0) * range_scale,
            ))
            if len(fill_specs) >= 2:
                break
        for light in self._env_fill_lights:
            if len(fill_specs) >= 2:
                break
            pos = np.array(light.get('position', (0.0, 0.0, 0.0)), dtype=np.float32)
            color = np.array(light.get('color', (0.0, 0.0, 0.0)), dtype=np.float32)
            fill_specs.append((
                self._transform_env_point(pos, model_mat),
                color,
                float(light.get('range', 1.0)) * range_scale,
            ))

        for slot in range(2):
            if slot < len(fill_specs):
                pos, color, light_range = fill_specs[slot]
                self._env_prog[f'u_fill_light_pos{slot}'].value = (
                    float(pos[0]), float(pos[1]), float(pos[2])
                )
                self._env_prog[f'u_fill_light_color{slot}'].value = (
                    float(color[0]), float(color[1]), float(color[2])
                )
                self._env_prog[f'u_fill_light_range{slot}'].value = max(float(light_range), 0.001)
            else:
                self._env_prog[f'u_fill_light_color{slot}'].value = (0.0, 0.0, 0.0)
                self._env_prog[f'u_fill_light_range{slot}'].value = 1.0

        glFrontFace(GL_CCW)

        fast_env = self._env_render_quality == 'fast'
        if fast_env:
            self._env_prog['u_use_normal_tex'].value = 0
            self._env_prog['u_use_occlusion_tex'].value = 0
            self._env_prog['u_use_mr_tex'].value = 0
            self._env_prog['u_use_emissive_tex'].value = 0
            self._env_prog['u_normal_scale'].value = 1.0
            self._env_prog['u_occlusion_strength'].value = 1.0
            self._env_prog['u_baked_lightmap'].value = 0

        opaque_prims = []
        blend_prims = []
        for prim in self._env_model_prims:
            base_alpha = float(prim.get('base_alpha', 1.0))
            alpha_mode = prim.get('alpha_mode', 'OPAQUE')
            if alpha_mode == 'BLEND':
                blend_prims.append(prim)
            else:
                opaque_prims.append(prim)

        if len(blend_prims) > 1:
            def _blend_sort_key(prim):
                local_center = prim.get('sort_center_local')
                if local_center is None:
                    local_center = np.zeros(3, dtype=np.float32)
                world_center = self._transform_env_point(local_center, model_mat)
                delta = world_center - cam_pos
                return float(np.dot(delta, delta))

            blend_prims.sort(key=_blend_sort_key, reverse=True)

        for prim in opaque_prims + blend_prims:
            base_color = prim.get('base_color', np.array([1.0, 1.0, 1.0], dtype=np.float32))
            base_alpha = float(prim.get('base_alpha', 1.0))
            alpha_mode = prim.get('alpha_mode', 'OPAQUE')
            if prim.get('double_sided', False):
                self.ctx.disable(moderngl.CULL_FACE)
            else:
                self.ctx.enable(moderngl.CULL_FACE)

            self._env_prog['u_base_color_factor'].value = (
                float(base_color[0]), float(base_color[1]), float(base_color[2])
            )
            self._env_prog['u_base_alpha'].value = base_alpha
            self._env_prog['u_roughness'].value = float(prim.get('roughness_factor', 1.0))
            self._env_prog['u_metallic'].value = float(prim.get('metallic_factor', 0.0))
            emissive = prim.get('emissive_factor', np.array([0.0, 0.0, 0.0], dtype=np.float32))
            self._env_prog['u_emissive_factor'].value = (
                float(emissive[0]), float(emissive[1]), float(emissive[2])
            )
            self._env_prog['u_unlit'].value = 1 if prim.get('unlit', False) else 0
            self._env_prog['u_foliage_mode'].value = 1 if prim.get('foliage_mode', False) else 0
            self._env_prog['u_alpha_mode'].value = 0 if alpha_mode == 'OPAQUE' else (1 if alpha_mode == 'MASK' else 2)
            self._env_prog['u_alpha_cutoff'].value = float(prim.get('alpha_cutoff', 0.5))

            if alpha_mode == 'BLEND':
                self.ctx.enable(moderngl.BLEND)
                self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
                self.ctx.depth_mask = False
            else:
                self.ctx.disable(moderngl.BLEND)
                self.ctx.depth_mask = True

            tex_offset = prim.get('tex_offset', np.array([0.0, 0.0], dtype=np.float32))
            tex_scale = prim.get('tex_scale', np.array([1.0, 1.0], dtype=np.float32))
            self._env_prog['u_tex_offset'].value = (float(tex_offset[0]), float(tex_offset[1]))
            self._env_prog['u_tex_scale'].value = (float(tex_scale[0]), float(tex_scale[1]))
            self._env_prog['u_tex_rotation'].value = float(prim.get('tex_rotation', 0.0))
            self._env_prog['u_base_texcoord'].value = int(prim.get('base_texcoord', 0))
            tex_key = prim.get('tex_key')
            if tex_key and tex_key in self._env_model_tex_cache:
                self._env_model_tex_cache[tex_key].use(location=3)
                self._env_prog['u_use_texture'].value = 1
            else:
                self._env_prog['u_use_texture'].value = 0

            if not fast_env:
                for uniform, tex_id_key, location in (
                    ('normal', 'normal_tex_id', 4),
                    ('occlusion', 'occlusion_tex_id', 5),
                    ('mr', 'mr_tex_id', 6),
                    ('emissive', 'emissive_tex_id', 7),
                ):
                    tex_id = prim.get(tex_id_key, -1)
                    sampler = prim.get(f'{uniform}_sampler')
                    cache_key = gltf_texture_cache_key('env', tex_id, sampler) if tex_id >= 0 else None
                    use_name = f'u_use_{uniform}_tex'
                    if cache_key and cache_key in self._env_model_tex_cache:
                        self._env_model_tex_cache[cache_key].use(location=location)
                        self._env_prog[use_name].value = 1
                    else:
                        self._env_prog[use_name].value = 0

                self._env_prog['u_normal_scale'].value = float(prim.get('normal_scale', 1.0))
                self._env_prog['u_occlusion_strength'].value = float(prim.get('occlusion_strength', 1.0))
                self._env_prog['u_normal_texcoord'].value = int(prim.get('normal_texcoord', 0))
                self._env_prog['u_occlusion_texcoord'].value = int(prim.get('occlusion_texcoord', 0))
                self._env_prog['u_mr_texcoord'].value = int(prim.get('mr_texcoord', 0))
                self._env_prog['u_emissive_texcoord'].value = int(prim.get('emissive_texcoord', 0))
            prim['vao'].render(prim.get('render_mode', moderngl.TRIANGLES))

        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.depth_mask = True
        self._env_prog['u_use_texture'].value = 1
        self._env_prog['u_base_color_factor'].value = (1.0, 1.0, 1.0)
        self._env_prog['u_base_alpha'].value = 1.0

        if self._env_perf_log:
            now = time.perf_counter()
            self._env_perf_accum_ms += (now - perf_t0) * 1000.0
            self._env_perf_samples += 1
            if self._env_perf_last_log <= 0.0:
                self._env_perf_last_log = now
            elif now - self._env_perf_last_log >= 5.0:
                avg_ms = self._env_perf_accum_ms / max(1, self._env_perf_samples)
                print(
                    "[OpenXRViewer] Env perf: "
                    f"fps={self.actual_fps:.1f} "
                    f"prims={len(self._env_model_prims)} "
                    f"avg_env_render={avg_ms:.2f}ms/eye "
                    f"quality={self._env_render_quality} "
                    f"shading={self._env_shading_mode}"
                )
                self._env_perf_last_log = now
                self._env_perf_accum_ms = 0.0
                self._env_perf_samples = 0


# Standalone smoke test helper shared by the viewer entry modules.
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

    viewer = viewer_cls(
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
    _smoke_test(OpenXRViewer)

