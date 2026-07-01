"""Environment model loading, profiles, and rendering mixin for OpenXR viewer."""

import json
import math
import os

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_MODULE_DIR)
import time

import numpy as np

from OpenGL.GL import (
    glGenFramebuffers, glBindFramebuffer, glFramebufferTexture2D,
    glDeleteFramebuffers, glCheckFramebufferStatus,
    GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
    GL_FRAMEBUFFER_COMPLETE,
    glGenRenderbuffers, glBindRenderbuffer,
    glRenderbufferStorage, glFramebufferRenderbuffer, glDeleteRenderbuffers,
    GL_RENDERBUFFER, GL_DEPTH_ATTACHMENT, GL_DEPTH_COMPONENT24,
    glEnable, glDisable, GL_CULL_FACE, GL_FRAMEBUFFER_SRGB,
    glFrontFace, GL_CW, GL_CCW,
)

import moderngl

_PANORAMA_IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff')
_PANORAMA_IMAGE_NAMES = (
    'background',
    'panorama',
    'equirectangular',
    '360',
    'sky',
    'skybox',
)

try:
    import xr
except ImportError:
    xr = None

from . import render as _render
from .glsl import _ENV_VERT, _ENV_FRAG
from .render import (
    _apply_transform, _euler_to_mat4, _mat4_to_xr_posef,
    _pose_to_view_mat4, _view_mat_inv, _xr_pose_to_model_mat4,
    _xr_quat_to_mat4, load_glb_model,
)
from .constants import (
    _BG_COLORS, _DEFAULT_BC, _DEFAULT_EF, _DEFAULT_TO, _DEFAULT_TS,
    _SCREEN_ENV_DEPTH_BIAS_M, _CURVED_HALF_ANGLE_RAD,
)


class EnvironmentMixin:
    """Environment model loading, profile management, and rendering.

    Expects the owning class to provide:
      self._env_model_prims, self._scene_lights, self._env_model_tex_cache,
      self._env_model_visible, self._env_model_pos, self._env_model_rot,
      self._env_model_scale, self._env_model_init_done, self._env_model_path,
      self._env_profile, self._screen_profile, self._view_pose_profile,
      self._view_pose_profiles, self._view_pose_index, self._environment_enabled,
      self._env_allow_curve, self._lighting_presets, self._lighting_preset_index,
      self._environment_root, self._environment_model, self._available_environment_models,
      self._active_environment, self._env_switch_osd_t,
      self._dark_room_prims, self._env_head_light_color, self._env_ambient_color,
      self._env_fill_lights, self._env_exposure, self._env_gamma,
      self._env_emissive_strength, self._env_khr_light_scale,
      self._env_base_settings, self._screen_light_intensity,
      self._glow_color, self._glow_target_color, self._glow_width_m,
      self._glow_intensity, self._glow_intensity_multiplier, self._glow_ref_screen,
      self._screen_width, self._screen_height, self._screen_distance,
      self._screen_curved, self._screen_yaw, self._screen_pitch, self._screen_roll,
      self._screen_pan_x, self._screen_pan_y, self._preset_index, self._screen_presets,
      self._profile_loaded, self._xr_session, self._xr_instance, self._xr_space,
      self._xr_ref_space_type, self._xr_space_pose_in_ref, self._xr_profile_space_applied,
      self._pending_recenter, self._env_prog, self._env_model_init_done,
      self._brand_osd_tex, self._brand_osd_vao, self._brand_osd_tex_size,
      self._brand_osd_alpha, self._brand_osd_show_t, self._brand_osd_last_name,
      self._all_models, self._available_brands, self._current_brand,
      self._brand_switch_osd_t, self._brand_sw_start, self._brand_sw_fired,
      self._controllers_root, self._ctrl_prims_l, self._ctrl_prims_r,
      self._ctrl_tex_cache, self._controller_prog,
      self.ctx, self._env_prog, self.prog, self.font,
      self._settings_sync_dirty, self._settings_sync_save_t,
      self._last_persisted_depth_ratio, self.depth_ratio,
      self._bg_color_idx, self._prev_bg_color_idx, self._prev_active_env,
      self._seat_adjust_active, self.settings_file,
      self._screen_ref_size, self.frame_size, self._capture_mode,
      self._input_monitor_index, self._controller_model,
    """


    def _load_env_model(self, path, target=None):
        """Load a glTF environment model from *path* into ``target`` dict.

        ``target`` schema: ``{'prims': list, 'tex_cache': dict, 'scene_lights': list}``.
        If ``target`` is None, this falls back to populating the legacy active
        slots (``self._env_model_prims`` / ``self._env_model_tex_cache``) for
        callers that haven't been migrated to the cache API.

        Textures use LINEAR_MIPMAP_LINEAR + mipmaps + 16x anisotropy.  Each
        cache uses its own texture-key prefix so loading multiple environments
        never collides.  If the file is corrupt or resources cannot be
        allocated, this method fails silently (prints a warning) and leaves
        the primitive list empty.
        """
        # Backwards-compat: write into the legacy active slots when no target dict given.
        legacy = target is None
        if legacy:
            target = {
                'prims': self._env_model_prims,
                'tex_cache': self._env_model_tex_cache,
                'scene_lights': self._scene_lights,
            }
        prims_data = []
        textures = []
        env_lights = []
        try:
            prims_data, textures, env_lights = load_glb_model(path)
            try:
                with open(path, 'rb') as _f:
                    _gltf, _bin = _read_glb_chunks(_f.read())
                _nodes = _gltf.get('nodes', []) if isinstance(_gltf, dict) else []
                _screen_idx = next(
                    (i for i, n in enumerate(_nodes)
                     if str(n.get('name', '')).lower() == 'wall_screen_524'),
                    None,
                )
                _screen_indices = set()
                if _screen_idx is not None:
                    _stack = [_screen_idx]
                    while _stack:
                        _ni = _stack.pop()
                        if _ni in _screen_indices or not (0 <= _ni < len(_nodes)):
                            continue
                        _screen_indices.add(_ni)
                        _stack.extend(int(c) for c in _nodes[_ni].get('children', []))
                target['screen_node_index'] = _screen_idx
                target['screen_node_indices'] = sorted(_screen_indices)
            except Exception:
                target['screen_node_index'] = None
                target['screen_node_indices'] = []
            if env_lights:
                target['scene_lights'] = list(env_lights)
                if legacy:
                    self._scene_lights = target['scene_lights']
        except Exception as exc:
            print(f"[OpenXRViewer] Failed to load environment model {path}: {exc}")
            return

        # Per-environment texture-key prefix derived from the parent folder
        # name (e.g. "env:Monitor:0").  This keeps caches disjoint across
        # environments so swapping in/out is just a dict reference swap with
        # zero risk of one env reading the other's textures.
        env_name = os.path.basename(os.path.dirname(os.path.abspath(path))) or 'env'
        _prefix = f"env:{env_name}"
        try:
            # Upload textures
            for tid, tex_arr in enumerate(textures):
                if tex_arr is not None:
                    cache_key = f"{_prefix}:{tid}"
                    h, w = tex_arr.shape[:2]
                    mtex = self.ctx.texture((w, h), 4, tex_arr.tobytes())
                    mtex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
                    mtex.build_mipmaps()
                    mtex.anisotropy = 16.0
                    target['tex_cache'][cache_key] = mtex

            # Create VAOs (bound to _env_prog, no gl_FrontFacing discard)
            for pd in prims_data:
                vbo = self.ctx.buffer(pd['vertices'].tobytes())
                tan_vbo = self.ctx.buffer(pd['tangent'].tobytes())
                ibo = self.ctx.buffer(pd['indices'].tobytes())
                vao = self.ctx.vertex_array(
                    self._env_prog,
                    [(vbo, '3f 3f 2f', 'in_position', 'in_normal', 'in_uv'),
                     (tan_vbo, '4f', 'in_tangent')],
                    ibo,
                )
                target['prims'].append({
                    'vao': vao, 'vbo': vbo, 'ibo': ibo,
                    'tex_key': f"{_prefix}:{pd['tex_id']}" if pd['tex_id'] >= 0 else None,
                    'tri_count': len(pd['indices']) // 3,
                    'positions': pd['vertices'][:, :3].copy(),
                    'node_index': pd.get('node_index', -1),
                    'base_color': pd.get('base_color', np.array([1.0, 1.0, 1.0], dtype=np.float32)),
                    'base_alpha': float(pd.get('base_alpha', 1.0)),
                    'roughness_factor': pd.get('roughness_factor', 1.0),
                    'metallic_factor': pd.get('metallic_factor', 1.0),
                    'emissive_factor': pd.get('emissive_factor', np.array([0.0, 0.0, 0.0], dtype=np.float32)),
                    'normal_tex_id': pd.get('normal_tex_id', -1),
                    'normal_scale': pd.get('normal_scale', 1.0),
                    'occlusion_tex_id': pd.get('occlusion_tex_id', -1),
                    'occlusion_strength': pd.get('occlusion_strength', 1.0),
                    'unlit': pd.get('unlit', False),
                    'alpha_mode': pd.get('alpha_mode', 'OPAQUE'),
                    'alpha_cutoff': pd.get('alpha_cutoff', 0.5),
                    'mr_tex_id': pd.get('mr_tex_id', -1),
                    'emissive_tex_id': pd.get('emissive_tex_id', -1),
                    'double_sided': pd.get('double_sided', False),
                    'tex_offset': pd.get('tex_offset', np.array([0.0, 0.0], dtype=np.float32)),
                    'tex_scale': pd.get('tex_scale', np.array([1.0, 1.0], dtype=np.float32)),
                    '_tex_prefix': _prefix,   # used by the render loop to resolve normal/occlusion keys
                })
                self._prebake_prim_render_state(target['prims'][-1])
            target['prims'].sort(key=lambda p: p.get('_rs', {}).get('blend', False))
        except Exception as exc:
            print(f"[OpenXRViewer] Failed to create environment model resources: {exc}")
            # Clean up partial resources
            for v in target['tex_cache'].values():
                try:
                    v.release()
                except Exception:
                    pass
            target['tex_cache'].clear()
            target['prims'].clear()


    def _generate_default_room(self, target_list=None):
        """Generate a simple room (floor, 4 walls, ceiling) procedurally.

        ``target_list`` is the list that receives the new prim dicts.  When
        omitted, defaults to the legacy ``self._env_model_prims`` slot so
        old call sites keep working unchanged.  Pass ``self._dark_room_prims``
        to build the always-on dark room (see ``_init_dark_room``).
        """
        if target_list is None:
            target_list = self._env_model_prims
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
            vbo = self.ctx.buffer(verts.tobytes())
            # Dummy tangent: (1,0,0,1) room faces have no normal map anyway
            dummy_tan = np.tile(np.array([1.0, 0.0, 0.0, 1.0], dtype='f4'), (verts.shape[0], 1))
            tan_vbo = self.ctx.buffer(dummy_tan.tobytes())
            ibo = self.ctx.buffer(idx.tobytes())
            vao = self.ctx.vertex_array(
                self._env_prog,
                [(vbo, '3f 3f 2f', 'in_position', 'in_normal', 'in_uv'),
                 (tan_vbo, '4f', 'in_tangent')],
                ibo,
            )
            target_list.append({
                'vao': vao, 'vbo': vbo, 'ibo': ibo,
                'tex_key': None, 'tri_count': 2, 'color': color,
                'base_color': np.array(color, dtype=np.float32),
                'roughness_factor': 1.0,
                'metallic_factor': 0.0,
                'emissive_factor': np.array([0.0, 0.0, 0.0], dtype=np.float32),
                'normal_tex_id': -1,
                'normal_scale': 1.0,
                'occlusion_tex_id': -1,
                'occlusion_strength': 1.0,
                'unlit': False,
                'alpha_mode': 'OPAQUE',
                'alpha_cutoff': 0.5,
                'mr_tex_id': -1,
                'emissive_tex_id': -1,
                'double_sided': False,
                'tex_offset': np.array([0.0, 0.0], dtype=np.float32),
                'tex_scale': np.array([1.0, 1.0], dtype=np.float32),
            })
            self._prebake_prim_render_state(target_list[-1])
        # When generating into the legacy active slots, flag the env model
        # visible so the fallback path (no env folders found) shows the room.
        # When generating into a dedicated list (e.g. dark room), leave the
        # visibility flag alone the caller decides when to render.
        if target_list is self._env_model_prims:
            self._env_model_visible = True
            print(f'[OpenXRViewer] Default room generated ({len(faces)} faces)')
        else:
            print(f'[OpenXRViewer] Dark-room geometry built ({len(faces)} faces)')

    @staticmethod
    def _read_env_profile(profile_path):
        if not os.path.isfile(profile_path):
            return {}
        try:
            with open(profile_path, 'r', encoding='utf-8-sig') as f:
                loaded = json.load(f)
            return loaded if isinstance(loaded, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _find_panorama_image_file(room_dir):
        if not room_dir or not os.path.isdir(room_dir):
            return None
        for stem in _PANORAMA_IMAGE_NAMES:
            for ext in _PANORAMA_IMAGE_EXTS:
                path = os.path.join(room_dir, stem + ext)
                if os.path.isfile(path):
                    return path
        try:
            for name in sorted(os.listdir(room_dir), key=lambda v: v.lower()):
                path = os.path.join(room_dir, name)
                if os.path.isfile(path) and os.path.splitext(name)[1].lower() in _PANORAMA_IMAGE_EXTS:
                    return path
        except OSError:
            pass
        return None

    @staticmethod
    def _prebake_prim_render_state(prim):
        rs = {}
        bc = prim.get('base_color')
        rs['bc'] = (float(bc[0]), float(bc[1]), float(bc[2])) if bc is not None else _DEFAULT_BC
        rs['ba'] = float(prim.get('base_alpha', 1.0))
        rs['rf'] = float(prim.get('roughness_factor', 1.0))
        rs['mf'] = float(prim.get('metallic_factor', 0.0))
        ef = prim.get('emissive_factor')
        rs['ef'] = (float(ef[0]), float(ef[1]), float(ef[2])) if ef is not None else _DEFAULT_EF
        rs['unlit'] = 1 if prim.get('unlit', False) else 0
        am = prim.get('alpha_mode', 'OPAQUE')
        rs['am'] = 0 if am == 'OPAQUE' else (1 if am == 'MASK' else 2)
        rs['ac'] = float(prim.get('alpha_cutoff', 0.5))
        rs['blend'] = (am == 'BLEND')
        to = prim.get('tex_offset')
        rs['to'] = (float(to[0]), float(to[1])) if to is not None else _DEFAULT_TO
        ts = prim.get('tex_scale')
        rs['ts'] = (float(ts[0]), float(ts[1])) if ts is not None else _DEFAULT_TS
        _tp = prim.get('_tex_prefix', 'env')
        rs['tk'] = prim.get('tex_key')
        nid = prim.get('normal_tex_id', -1)
        rs['n_key'] = f"{_tp}:{nid}" if nid >= 0 else None
        rs['ns'] = float(prim.get('normal_scale', 1.0))
        oid = prim.get('occlusion_tex_id', -1)
        rs['o_key'] = f"{_tp}:{oid}" if oid >= 0 else None
        rs['os'] = float(prim.get('occlusion_strength', 1.0))
        mid = prim.get('mr_tex_id', -1)
        rs['m_key'] = f"{_tp}:{mid}" if mid >= 0 else None
        eid = prim.get('emissive_tex_id', -1)
        rs['e_key'] = f"{_tp}:{eid}" if eid >= 0 else None
        prim['_rs'] = rs

    def _init_dark_room(self):
        """Build the always-available procedural dark room.

        Triggered once at GL init time.  The geometry sits in a dedicated
        list (``self._dark_room_prims``) so it never collides with .glb
        environments swapped via the cycle.  It is rendered automatically
        by ``_render_dark_room`` whenever no .glb env is loaded AND the
        backdrop is "Black" (idx 0) giving the cinema bias light real
        surfaces to bounce off and turning a flat backdrop into a small
        theater per the user's "rectangular dark room with walls and
        ceilings" request.
        """
        self._dark_room_prims = []
        try:
            self._generate_default_room(self._dark_room_prims)
        except Exception as exc:
            print(f"[OpenXRViewer] _init_dark_room failed: {exc}")
            self._dark_room_prims = []

    def _init_env_model(self):
        """Load the selected environment GLB, or fall back to the procedural room."""
        if not getattr(self, '_environment_enabled', True):
            self._env_model_visible = False
            return
        if getattr(self, '_panorama_background_path', None):
            self._env_model_visible = False
            self._active_environment = None
            return
        selected = (self._environment_model or '').strip().lower()
        if selected == 'default':
            self._env_model_visible = False
            self._active_environment = None
            return
        path = self._env_model_path
        if path and os.path.exists(path):
            self._load_env_model(path)
            if self._env_model_prims:
                self._env_model_visible = True
                self._active_environment = self._environment_model
                print(f"[OpenXRViewer] Environment model loaded "
                      f"({len(self._env_model_prims)} primitives): {self._environment_model}")
                return
        self._generate_default_room()
        self._active_environment = self._environment_model

    def _release_env_model_resources(self):
        """Release current room GL resources before reloading or shutdown."""
        for prim in self._env_model_prims:
            for key in ('vao', 'vbo', 'ibo'):
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
        self._cached_env_light = None

    def _discover_environment_models(self):
        """Return room folders that can be switched at runtime."""
        panorama_models = []
        glb_models = []
        root = getattr(self, '_environment_root', None)
        if not root or not os.path.isdir(root):
            return ['Default']
        try:
            for name in sorted(os.listdir(root), key=lambda v: v.lower()):
                room_dir = os.path.join(root, name)
                if not os.path.isdir(room_dir):
                    continue
                profile_path = os.path.join(room_dir, 'profile.json')
                profile = self._read_env_profile(profile_path)
                glb_path = os.path.join(room_dir, 'environment.glb')
                explicit_pano, pano_path, _pano_cfg = self._panorama_profile_config(profile, room_dir)
                if explicit_pano and pano_path and os.path.isfile(pano_path):
                    panorama_models.append(name)
                    continue
                if not os.path.isfile(glb_path):
                    glb_name = str(profile.get('glb', 'environment.glb') or 'environment.glb')
                    glb_path = glb_name if os.path.isabs(glb_name) else os.path.join(room_dir, glb_name)
                if os.path.isfile(glb_path):
                    glb_models.append(name)
                    continue
                auto_pano = self._find_panorama_image_file(room_dir)
                if auto_pano:
                    panorama_models.append(name)
                    continue
        except Exception:
            pass
        models = ['Default'] + panorama_models + glb_models
        selected = (getattr(self, '_environment_model', '') or '').strip()
        if selected and selected.lower() not in ('default', 'default glow', 'default with glow') and selected not in models:
            models.insert(0, selected)
        return models

    def _environment_screen_locked(self):
        """Return True when the active profile has a non-empty screen section."""
        screen = getattr(self, '_screen_profile', {}) or {}
        return isinstance(screen, dict) and bool(screen)

    def _screen_profile_value(self, key, default=None):
        screen = getattr(self, '_screen_profile', {}) or {}
        return screen.get(key, default)

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
        self._env_fill_lights = list(base['fill_lights'])
        self._env_exposure = float(base['exposure'])
        self._env_gamma = float(base['gamma'])
        self._env_emissive_strength = float(base['emissive_strength'])
        self._env_khr_light_scale = float(base['khr_light_scale'])
        self._screen_light_intensity = float(base['screen_light_intensity'])
        if 'glow_intensity' in base:
            self._glow_intensity = float(base['glow_intensity'])
        if 'glow_width' in base:
            self._glow_width_m = float(base['glow_width'])
        if 'glow_intensity_multiplier' in base:
            self._glow_intensity_multiplier = float(base['glow_intensity_multiplier'])
        if 'glow_mode' in base:
            self._glow_mode = str(base['glow_mode'])
        if hasattr(self, '_refresh_active_glow_mode_cache'):
            self._refresh_active_glow_mode_cache()
        self._panorama_background_path = None
        self._panorama_background_settings = {}

    def _panorama_profile_config(self, profile, room_dir):
        if not isinstance(profile, dict):
            return False, None, {}
        raw_bg = profile.get('background')
        raw_panorama = profile.get('panorama')
        cfg = {}
        if isinstance(raw_bg, str):
            cfg['image'] = raw_bg
        elif isinstance(raw_bg, dict):
            cfg.update(raw_bg)
        if isinstance(raw_panorama, str):
            cfg.setdefault('image', raw_panorama)
        elif isinstance(raw_panorama, dict):
            cfg.update(raw_panorama)

        env_type = str(profile.get('environment_type', profile.get('type', '')) or '').strip().lower()
        bg_type = str(cfg.get('type', cfg.get('kind', '')) or '').strip().lower()
        projection = str(cfg.get('projection', cfg.get('format', '')) or '').strip().lower()
        is_panorama = (
            env_type in ('panorama', '360', '360_photo', '360-photo', 'photo_sphere', 'photosphere')
            or bg_type in ('panorama', '360', '360_photo', '360-photo', 'equirectangular', 'photo_sphere', 'photosphere')
            or projection in ('equirectangular', '360', '360_photo', '360-photo')
            or raw_panorama is True
        )
        if not is_panorama:
            return False, None, {}

        image = (
            cfg.get('image')
            or cfg.get('path')
            or cfg.get('file')
            or profile.get('background_image')
            or None
        )
        if image:
            image = str(image)
            path = image if os.path.isabs(image) else os.path.join(room_dir, image)
            cfg['image'] = image
        else:
            path = self._find_panorama_image_file(room_dir)
            if path:
                cfg['image'] = os.path.basename(path)
        return True, path, cfg

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
        default_key = selected.lower()
        is_default = (default_key == 'default')
        if is_default:
            self._env_model_path = None
            self._env_model_visible = False
            self._active_environment = None
            self._environment_model = 'Default'
            builtin_path = os.path.join(root, '.builtin_default.json')
            builtin_profile = {}
            if os.path.isfile(builtin_path):
                try:
                    with open(builtin_path, 'r', encoding='utf-8-sig') as f:
                        loaded = json.load(f)
                    if isinstance(loaded, dict):
                        builtin_profile = loaded
                except Exception:
                    pass
            self._env_profile = builtin_profile
            for key, attr in (
                ('glow_intensity', '_glow_intensity'),
                ('glow_width', '_glow_width_m'),
                ('glow_intensity_multiplier', '_glow_intensity_multiplier'),
                ('screen_light_intensity', '_screen_light_intensity'),
            ):
                if key in builtin_profile:
                    try:
                        setattr(self, attr, float(builtin_profile[key]))
                    except (TypeError, ValueError):
                        pass
            if 'glow_mode' in builtin_profile:
                self._glow_mode = str(builtin_profile.get('glow_mode') or 'off').strip().lower()
            else:
                self._glow_mode = 'glow' if float(getattr(self, '_glow_intensity_multiplier', 0.0)) > 0.0 else 'off'
            if hasattr(self, '_refresh_active_glow_mode_cache'):
                self._refresh_active_glow_mode_cache()
            lp = builtin_profile.get('lighting_presets')
            self._lighting_presets = lp if isinstance(lp, list) and lp else []
            self._lighting_preset_index = 0
            print(f"[OpenXRViewer] Environment: Default (blank)")
            return
        room_dir = root if is_default else os.path.join(root, selected)
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
        is_panorama, panorama_path, panorama_cfg = self._panorama_profile_config(profile, room_dir)
        glb_name = str(profile.get('glb', 'environment.glb') or 'environment.glb')
        glb_path = glb_name if os.path.isabs(glb_name) else os.path.join(room_dir, glb_name)
        if not is_panorama and not os.path.isfile(glb_path):
            auto_panorama_path = self._find_panorama_image_file(room_dir)
            if auto_panorama_path:
                is_panorama = True
                panorama_path = auto_panorama_path
                panorama_cfg = {
                    'type': 'equirectangular',
                    'image': os.path.basename(auto_panorama_path),
                    'exposure': 1.0,
                    'yaw_offset_deg': 0.0,
                }
        if is_panorama:
            if panorama_path and os.path.isfile(panorama_path):
                glb_path = None
            else:
                missing = panorama_path or os.path.join(room_dir, 'background.png')
                print(f"[OpenXRViewer] Panorama environment '{selected}' missing image: {missing}")
                is_panorama = False
                panorama_path = None
                panorama_cfg = {}
        if not is_panorama and not os.path.exists(glb_path) and not is_default:
            fallback = os.path.join(root, 'environment.glb')
            print(f"[OpenXRViewer] Environment '{selected}' missing GLB, fallback to Default")
            selected = 'Default'
            room_dir = root
            glb_path = fallback
            profile = {}
            self._available_environment_models = []
        self._environment_model = selected
        self._env_profile = profile
        self._env_model_path = glb_path
        self._panorama_background_path = panorama_path if is_panorama else None
        self._panorama_background_settings = panorama_cfg if is_panorama else {}
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
        self._env_head_light_color = tuple(self._profile_vec3(
            profile, ('env_head_light_color', 'head_light_color'), self._env_head_light_color))
        self._env_ambient_color = tuple(self._profile_vec3(
            profile, ('env_ambient_color', 'ambient_color'), self._env_ambient_color))
        fill_lights = profile.get('env_fill_lights', profile.get('fallback_lights'))
        if isinstance(fill_lights, list):
            self._env_fill_lights = fill_lights
        if 'screen_light_intensity' in profile:
            try:
                self._screen_light_intensity = float(profile['screen_light_intensity'])
            except (TypeError, ValueError):
                pass
        for key, attr in (
            ('glow_intensity', '_glow_intensity'),
            ('glow_width', '_glow_width_m'),
            ('glow_intensity_multiplier', '_glow_intensity_multiplier'),
        ):
            if key in profile:
                try:
                    setattr(self, attr, float(profile[key]))
                except (TypeError, ValueError):
                    pass
        if 'glow_mode' in profile:
            self._glow_mode = str(profile.get('glow_mode') or 'off').strip().lower()
        else:
            self._glow_mode = 'glow' if float(getattr(self, '_glow_intensity_multiplier', 0.0)) > 0.0 else 'off'
        if hasattr(self, '_refresh_active_glow_mode_cache'):
            self._refresh_active_glow_mode_cache()
        lp = profile.get('lighting_presets')
        self._lighting_presets = lp if isinstance(lp, list) and lp else []
        self._lighting_preset_index = 0
        if is_panorama:
            print(f"[OpenXRViewer] Environment: {self._environment_model} (panorama {os.path.basename(panorama_path)})")
        else:
            print(f"[OpenXRViewer] Environment: {self._environment_model} ({self._env_model_path})")

    def _configure_profile_view_layout(self):
        """Cache optional room-specific viewer and screen layout settings."""
        profile = self._env_profile if isinstance(self._env_profile, dict) else {}
        view_poses = profile.get('view_poses')
        if isinstance(view_poses, list):
            self._view_pose_profiles = [p for p in view_poses if isinstance(p, dict)]
        else:
            self._view_pose_profiles = []
        try:
            self._view_pose_index = int(profile.get('view_pose_index', 0))
        except (TypeError, ValueError):
            self._view_pose_index = 0
        if self._view_pose_profiles:
            self._view_pose_index %= len(self._view_pose_profiles)
            view_pose = self._view_pose_profiles[self._view_pose_index]
        else:
            view_pose = profile.get('view_pose', profile.get('camera', {}))
        screen = profile.get('screen', {})
        self._view_pose_profile = view_pose if isinstance(view_pose, dict) else {}
        self._screen_profile = screen if isinstance(screen, dict) else {}
        if self._screen_profile:
            print(f"[OpenXRViewer] Profile screen layout enabled: {self._environment_model}")

    def _head_model_mat4_from_views(self, views):
        """Extract head centre from left+right eye poses as a 4x4 model matrix."""
        if not views or len(views) < 2 or views[0] is None or views[1] is None:
            return None
        head_mat = _xr_pose_to_model_mat4(views[0].pose)
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
        leveled = _euler_to_mat4(yaw, 0.0, 0.0).astype(np.float32)
        leveled[:3, 3] = pos
        return leveled

    def _auto_view_position_from_screen(self, view, has_view_rot, rot_deg_keys, rot_rad_keys):
        """Compute a viewer position centred on the configured screen."""
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
            screen_normal = _euler_to_mat4(*screen_rot)[:3, 2].astype(np.float32)
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

    def _apply_profile_view_pose_to_xr_space(self, views):
        """Offset the XR reference space so the user sits at the profile view_pose."""
        if self._xr_profile_space_applied:
            return False
        self._xr_profile_space_applied = True
        view = getattr(self, '_view_pose_profile', {}) or {}
        if not isinstance(view, dict) or not view:
            return False
        if self._xr_session is None or self._xr_space is None or self._xr_ref_space_type is None:
            return False
        if 'y' in view:
            x = float(view.get('x', 0.0))
            y = float(view.get('y', 0.6))
            z = float(view.get('z', 0.0))
            angle = float(view.get('angle', 0.0))
            self._last_located_views = views
            self._apply_seat_adjust_xr_space(x, y, z, angle)
            pos = [round(x, 4), round(y, 4), round(z, 4)]
            print(f"[OpenXRViewer] Applied view_pose (screen-relative): x={pos[0]} y={pos[1]} z={pos[2]} angle={angle}")
            return True
        pos_keys = ('position', 'camera_position', 'viewer_position')
        rot_deg_keys = ('rotation_deg', 'camera_rotation_deg', 'viewer_rotation_deg')
        rot_rad_keys = ('rotation', 'camera_rotation', 'viewer_rotation')
        has_pos = any(key in view for key in pos_keys)
        has_rot = any(key in view for key in rot_deg_keys + rot_rad_keys)
        auto_center = bool(view.get('auto_center_on_screen', False))
        if not (has_pos or has_rot or auto_center):
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
            desired_head[:3, :3] = _euler_to_mat4(*rot)[:3, :3]
        try:
            current_space_in_ref = getattr(self, '_xr_space_pose_in_ref', np.eye(4, dtype=np.float32))
            reference_head = current_space_in_ref @ raw_head
            reference_head[:3, :3] = desired_head[:3, :3]
            space_in_ref = reference_head @ np.linalg.inv(desired_head)
            new_space = xr.create_reference_space(
                self._xr_session,
                xr.ReferenceSpaceCreateInfo(
                    reference_space_type=self._xr_ref_space_type,
                    pose_in_reference_space=_mat4_to_xr_posef(space_in_ref.astype(np.float32)),
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

    def _reset_xr_space_to_identity(self):
        """Reset XR reference space offset to identity (no teleport)."""
        if self._xr_session is None or self._xr_ref_space_type is None:
            return
        current = getattr(self, '_xr_space_pose_in_ref', np.eye(4, dtype=np.float32))
        if np.allclose(current, np.eye(4)):
            return
        try:
            new_space = xr.create_reference_space(
                self._xr_session,
                xr.ReferenceSpaceCreateInfo(
                    reference_space_type=self._xr_ref_space_type,
                    pose_in_reference_space=xr.Posef(),
                ),
            )
        except Exception as exc:
            print(f"[OpenXRViewer] Failed to reset XR space: {exc}")
            return
        old_space = self._xr_space
        self._xr_space = new_space
        self._xr_space_pose_in_ref = np.eye(4, dtype=np.float32)
        if old_space is not None:
            try:
                xr.destroy_space(old_space)
            except Exception:
                pass
        print("[OpenXRViewer] XR space reset to identity")

    def _recenter_profile_view_pose(self):
        """Re-apply profile view_pose (Y-button / Home recenter when locked)."""
        view = getattr(self, '_view_pose_profile', {}) or {}
        if not isinstance(view, dict) or not view:
            return False
        views = getattr(self, '_last_located_views', None)
        if not views or views[0] is None or views[1] is None:
            return False
        self._xr_profile_space_applied = False
        applied = self._apply_profile_view_pose_to_xr_space(views)
        if applied:
            print("[OpenXRViewer] Recentered view_pose to profile position")
        return applied

    def _apply_profile_screen_layout(self, show_border=False):
        """Apply fixed room-specific screen layout from profile.json."""
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
            if ('rotation_deg' in screen or 'screen_rotation_deg' in screen
                    or 'rotation' in screen or 'screen_rotation' in screen):
                self.screen_yaw, self.screen_pitch, self.screen_roll = rotation
            else:
                self.screen_yaw = math.atan2(-fx, -fz)
                self.screen_pitch = 0.0
                self.screen_roll = 0.0
        curve_mode = self._curve_mode_from_json(screen)
        if curve_mode is not None:
            self._set_screen_curve_mode(curve_mode)
        if 'allow_curve' in screen:
            self._env_allow_curve = bool(screen['allow_curve'])
        self._last_overlay_update = 0.0
        self._border_alpha = 0.0
        if self._keyboard_visible:
            self._anchor_keyboard_below_screen()
        self._profile_loaded = True
        self._clear_screen_grab_anchors()
        return True

    def _curve_mode_from_json(self, data):
        """Read current curve_axis values, with old curved bool compatibility."""
        if not isinstance(data, dict):
            return None
        if 'curve_axis' in data:
            return str(data.get('curve_axis') or 'none').strip().lower()
        if 'curved' not in data:
            return None
        curved = data.get('curved')
        if isinstance(curved, str):
            val = curved.strip().lower()
            if val in ('horizontal', 'vertical', 'none', 'flat', 'off'):
                return val
            if val in ('true', 'yes', 'on', 'curved'):
                return 'horizontal'
            return 'none'
        return 'horizontal' if bool(curved) else 'none'

    def _switch_environment_model(self, model_name=None):
        """Switch to another room environment during runtime."""
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
        if self._seat_adjust_active:
            self._exit_seat_adjust_mode(save=False)
        if not self._environment_screen_locked():
            self._persist_screen_state()
        self._release_env_model_resources()
        self._environment_model = model_name
        self._kb_cached_position = None   # reset keyboard cache on env switch
        self._configure_environment_profile()
        self._configure_profile_view_layout()
        self._init_env_model()
        self._apply_profile_screen_layout(show_border=True)
        self._xr_profile_space_applied = False
        views = getattr(self, '_last_located_views', None)
        if views:
            self._apply_profile_view_pose_to_xr_space(views)
        if not self._environment_screen_locked():
            self._reset_xr_space_to_identity()
            if not self._restore_screen_state():
                self._reset_screen_to_default(show_border=True)
        self._persist_runtime_settings()
        return True

    # ------------------------------------------------------------------
    # Profile helper methods (safe parsing with fallbacks)
    # ------------------------------------------------------------------
    def _profile_vec3(self, profile, keys, default):
        """Try multiple key names for a vec3, return [x, y, z] or *default*."""
        for key in keys:
            value = profile.get(key)
            if isinstance(value, (list, tuple)) and len(value) >= 3:
                try:
                    return [float(value[0]), float(value[1]), float(value[2])]
                except (TypeError, ValueError):
                    pass
        return list(default)

    def _profile_float(self, profile, keys, default):
        """Try multiple key names for a float, return float or *default*."""
        for key in keys:
            if key in profile:
                try:
                    return float(profile[key])
                except (TypeError, ValueError):
                    pass
        return float(default)

    def _profile_bool(self, profile, keys, default):
        """Try multiple key names for a bool, return bool or *default*."""
        for key in keys:
            if key in profile:
                value = profile[key]
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    return value.strip().lower() in ('1', 'true', 'yes', 'on')
                return bool(value)
        return bool(default)

    def _profile_rotation_rad(self, profile, deg_keys, rad_keys, default):
        """Try degree keys first (convert to radians), then radian keys, then default."""
        for key in deg_keys:
            value = profile.get(key)
            if isinstance(value, (list, tuple)) and len(value) >= 3:
                try:
                    return [math.radians(float(value[0])),
                            math.radians(float(value[1])),
                            math.radians(float(value[2]))]
                except (TypeError, ValueError):
                    pass
        return self._profile_vec3(profile, rad_keys, default)

    # ------------------------------------------------------------------
    # settings.yaml persistence (live updates from OpenXR side)
    # ------------------------------------------------------------------
    def _persist_setting(self, key, value):
        try:
            from utils import write_yaml as _wy
            path = os.path.join(_PROJECT_ROOT, 'settings.yaml')
            _wy(path, {key: value})
        except Exception as exc:
            print(f"[OpenXRViewer] _persist_setting({key!r}) failed: {exc}")

    def _persist_active_environment(self):
        current = (self._environment_model or '').strip()
        if current.lower() in ('default', 'default glow', 'default with glow'):
            val = current
        elif self._active_environment:
            val = self._active_environment
        else:
            val = 'Default'
        self._persist_setting('Environment Model', val)

    def _settings_snapshot(self):
        current = (self._environment_model or '').strip()
        if current.lower() in ('default', 'default glow', 'default with glow'):
            env_val = current
        elif self._active_environment:
            env_val = self._active_environment
        elif current:
            env_val = current
        else:
            env_val = 'Default'
        ctrl_val = getattr(self, '_current_brand', None) or self._controller_model
        return {
            'Controller Model': ctrl_val,
            'Environment Model': env_val,
            'Depth Strength': round(float(self.depth_ratio), 4),
        }

    def _persist_runtime_settings(self):
        """Save GUI-facing runtime settings without touching render-only state."""
        try:
            import yaml
            from utils import read_yaml as _ry
            path = os.path.join(_PROJECT_ROOT, 'settings.yaml')
            cfg = {}
            if os.path.exists(path):
                try:
                    cfg = _ry(path)
                except Exception:
                    cfg = {}
            if not isinstance(cfg, dict):
                cfg = {}
            for stale_key in ('Glow Mode', 'Screen Curve', 'Screen Opacity'):
                cfg.pop(stale_key, None)
            cfg.update(self._settings_snapshot())
            with open(path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
            self._last_persisted_depth_ratio = float(self.depth_ratio)
            self._settings_sync_dirty = False
        except Exception as exc:
            print(f"[OpenXRViewer] _persist_runtime_settings failed: {exc}")

    def _mark_runtime_settings_dirty(self):
        self._settings_sync_dirty = True
        self._settings_sync_save_t = time.perf_counter()

    def _flush_runtime_settings_if_idle(self, delay=0.5):
        if not self._settings_sync_dirty:
            return
        if time.perf_counter() - self._settings_sync_save_t >= delay:
            self._persist_runtime_settings()

    def _builtin_profile_path(self):
        return os.path.join(self._environment_root, '.builtin_default.json')

    def _active_profile_path(self):
        env_name = (self._active_environment or self._environment_model or 'Default').strip()
        if env_name.lower() in ('default', 'default glow', 'default with glow'):
            return self._builtin_profile_path()
        return os.path.join(self._environment_root, env_name, 'profile.json')

    def _persist_screen_state(self):
        """Save Default-environment screen layout to .builtin_default.json."""
        if self._environment_screen_locked():
            return
        if self._active_environment is not None:
            return
        state = {
            'width': round(float(self.screen_width), 4),
            'distance': round(float(self.screen_distance), 4),
            'pan_x': round(float(self.screen_pan_x), 4),
            'pan_y': round(float(self.screen_pan_y), 4),
            'yaw': round(float(self.screen_yaw), 6),
            'pitch': round(float(self.screen_pitch), 6),
            'curve_axis': self._screen_curve_mode(),
            'preset_index': int(self._preset_index),
        }
        builtin_path = self._builtin_profile_path()
        try:
            profile = {}
            if os.path.isfile(builtin_path):
                with open(builtin_path, 'r', encoding='utf-8-sig') as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    profile = loaded
            profile['screen_state'] = state
            with open(builtin_path, 'w', encoding='utf-8') as f:
                json.dump(profile, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            print(f"[OpenXRViewer] _persist_screen_state failed: {exc}")

    def _restore_screen_state(self):
        """Load Default-environment screen layout from .builtin_default.json."""
        if self._environment_screen_locked():
            return False
        if self._active_environment is not None:
            return False
        state = None
        builtin_path = self._builtin_profile_path()
        try:
            if os.path.isfile(builtin_path):
                with open(builtin_path, 'r', encoding='utf-8-sig') as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    state = loaded.get('screen_state')
        except Exception:
            pass
        if not isinstance(state, dict):
            # Migration: fall back to settings.yaml for first run after update.
            try:
                from utils import read_yaml as _ry
                cfg = _ry(os.path.join(_PROJECT_ROOT, 'settings.yaml'))
                state = cfg.get('Screen State')
            except Exception:
                pass
        if not isinstance(state, dict):
            return False
        self.screen_width = float(state.get('width', self.screen_width))
        self._screen_ref_size = self.screen_width
        self.screen_height = None
        self.screen_distance = float(state.get('distance', self.screen_distance))
        self.screen_pan_x = float(state.get('pan_x', self.screen_pan_x))
        self.screen_pan_y = float(state.get('pan_y', self.screen_pan_y))
        self.screen_yaw = float(state.get('yaw', self.screen_yaw))
        self.screen_pitch = float(state.get('pitch', self.screen_pitch))
        curve_mode = self._curve_mode_from_json(state)
        if curve_mode is not None:
            self._set_screen_curve_mode(curve_mode)
        self._preset_index = int(state.get('preset_index', self._preset_index))
        print(f"[OpenXRViewer] Restored screen state: {state.get('width')}m, "
              f"dist={state.get('distance')}, curve={self._screen_curve_mode()}")
        return True

    def _cycle_environment(self):
        """Advance the environment one slot (left-stick short press)."""
        self._switch_environment_model()

    def _cycle_lighting_preset(self):
        """Cycle through lighting_presets in the current profile (Y long-press)."""
        presets = self._lighting_presets
        if not presets:
            return
        self._lighting_preset_index = (self._lighting_preset_index + 1) % len(presets)
        p = presets[self._lighting_preset_index]
        if 'env_exposure' in p:
            try:
                self._env_exposure = float(p['env_exposure'])
            except (TypeError, ValueError):
                pass
        if 'env_ambient_color' in p:
            v = p['env_ambient_color']
            if isinstance(v, (list, tuple)) and len(v) >= 3:
                self._env_ambient_color = (float(v[0]), float(v[1]), float(v[2]))
        if 'env_head_light_color' in p:
            v = p['env_head_light_color']
            if isinstance(v, (list, tuple)) and len(v) >= 3:
                self._env_head_light_color = (float(v[0]), float(v[1]), float(v[2]))
        if 'env_gamma' in p:
            try:
                self._env_gamma = float(p['env_gamma'])
            except (TypeError, ValueError):
                pass
        if 'env_emissive_strength' in p:
            try:
                self._env_emissive_strength = float(p['env_emissive_strength'])
            except (TypeError, ValueError):
                pass
        if 'screen_light_intensity' in p:
            try:
                self._screen_light_intensity = float(p['screen_light_intensity'])
            except (TypeError, ValueError):
                pass
        for key, attr in (
            ('glow_intensity', '_glow_intensity'),
            ('glow_width', '_glow_width_m'),
            ('glow_intensity_multiplier', '_glow_intensity_multiplier'),
        ):
            if key in p:
                try:
                    setattr(self, attr, float(p[key]))
                except (TypeError, ValueError):
                    pass
        if 'glow_mode' in p:
            self._glow_mode = str(p.get('glow_mode') or 'off').strip().lower()
        elif 'glow_intensity_multiplier' in p:
            self._glow_mode = 'glow' if float(getattr(self, '_glow_intensity_multiplier', 0.0)) > 0.0 else 'off'
        if hasattr(self, '_refresh_active_glow_mode_cache'):
            self._refresh_active_glow_mode_cache()
        name = p.get('name', f'Preset {self._lighting_preset_index}')
        print(f"[OpenXRViewer] Lighting preset: {name}")

    def _cycle_view_pose(self):
        """Cycle through multi-seat view_poses in the active environment profile."""
        poses = getattr(self, '_view_pose_profiles', []) or []
        if len(poses) < 2:
            return False
        self._view_pose_index = (int(getattr(self, '_view_pose_index', 0)) + 1) % len(poses)
        self._view_pose_profile = poses[self._view_pose_index]
        if isinstance(getattr(self, '_env_profile', None), dict):
            self._env_profile['view_pose_index'] = self._view_pose_index
        env_name = self._active_environment or self._environment_model
        if env_name and env_name.lower() not in ('default', 'default glow', 'default with glow'):
            profile_path = os.path.join(self._environment_root, env_name, 'profile.json')
            try:
                with open(profile_path, 'r', encoding='utf-8-sig') as f:
                    profile = json.load(f)
                if isinstance(profile.get('view_poses'), list):
                    profile['view_pose_index'] = self._view_pose_index
                    with open(profile_path, 'w', encoding='utf-8') as f:
                        json.dump(profile, f, indent=2, ensure_ascii=False)
            except Exception as exc:
                print(f"[OpenXRViewer] Failed to save view_pose_index: {exc}")
        self._xr_profile_space_applied = False
        views = getattr(self, '_last_located_views', None)
        if views:
            self._apply_profile_view_pose_to_xr_space(views)
        self._seat_adjust_osd_dirty = True
        self._seat_adjust_osd_show_t = time.perf_counter()
        name = self._view_pose_profile.get('name', f'View {self._view_pose_index + 1}')
        print(f"[OpenXRViewer] View pose: {name} ({self._view_pose_index + 1}/{len(poses)})")
        return True

    def _env_uses_view_pose_cycle(self):
        """Only selected room models use Y long-press for seating/view cycling."""
        env = (self._active_environment or self._environment_model or '').strip().lower()
        return env in ('cinema', 'living room', 'theater')

    def _toggle_passthrough_backdrop(self):
        """Toggle green passthrough without unloading the active environment."""
        if self._bg_color_idx == 1 and self._prev_bg_color_idx is not None:
            self._bg_color_idx = self._prev_bg_color_idx
            self._prev_bg_color_idx = None
            print("[OpenXRViewer] Passthrough backdrop: off")
        else:
            if self._prev_bg_color_idx is None:
                self._prev_bg_color_idx = self._bg_color_idx
            self._bg_color_idx = 1
            print("[OpenXRViewer] Passthrough backdrop: on")

    def _cycle_light_from_x(self):
        """Cycle screen glow mode from the left X long-press release."""
        modes = ('glow', 'veil', 'frosted', 'off')
        if hasattr(self, '_active_glow_mode'):
            current = self._active_glow_mode()
        else:
            current = str(getattr(self, '_glow_mode', 'off') or 'off').strip().lower()
        if current not in modes:
            current = 'glow' if float(getattr(self, '_glow_intensity_multiplier', 0.0)) > 0.0 else 'off'
        next_mode = modes[(modes.index(current) + 1) % len(modes)]
        self._glow_mode = next_mode
        if next_mode == 'off':
            self._glow_intensity_multiplier = 0.0
        elif float(getattr(self, '_glow_intensity_multiplier', 0.0)) <= 0.0:
            self._glow_intensity_multiplier = 1.5
        if hasattr(self, '_refresh_active_glow_mode_cache'):
            self._refresh_active_glow_mode_cache()
        print(f"[OpenXRViewer] Glow mode: {next_mode}")
        self._save_glow_to_active_profile()
        return True

    def _save_glow_to_builtin_profile(self):
        """Write glow settings into .builtin_default.json for the Default env."""
        self._save_glow_to_profile_path(self._builtin_profile_path(), update_active=False)

    def _save_glow_to_active_profile(self):
        """Write glow settings into the active XR JSON profile."""
        self._save_glow_to_profile_path(self._active_profile_path(), update_active=True)

    def _save_glow_to_profile_path(self, profile_path, update_active=False):
        try:
            profile = {}
            if os.path.isfile(profile_path):
                with open(profile_path, 'r', encoding='utf-8-sig') as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    profile = loaded
            profile['glow_intensity'] = self._glow_intensity
            profile['glow_width'] = self._glow_width_m
            profile['glow_intensity_multiplier'] = self._glow_intensity_multiplier
            profile['glow_mode'] = str(getattr(self, '_glow_mode', 'off') or 'off')
            with open(profile_path, 'w', encoding='utf-8') as f:
                json.dump(profile, f, indent=2, ensure_ascii=False)
            if update_active and profile_path != self._builtin_profile_path():
                self._env_profile = profile
        except Exception as exc:
            print(f"[OpenXRViewer] _save_glow_to_profile_path failed: {exc}")

    def _save_curve_to_active_profile(self):
        """Write the active curve mode into the relevant XR JSON profile."""
        if self._active_environment is None:
            self._persist_screen_state()
            return
        profile_path = self._active_profile_path()
        try:
            profile = {}
            if os.path.isfile(profile_path):
                with open(profile_path, 'r', encoding='utf-8-sig') as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    profile = loaded
            screen = profile.get('screen')
            if not isinstance(screen, dict):
                screen = {}
            screen['curve_axis'] = self._screen_curve_mode()
            screen.pop('curved', None)
            profile['screen'] = screen
            with open(profile_path, 'w', encoding='utf-8') as f:
                json.dump(profile, f, indent=2, ensure_ascii=False)
            self._env_profile = profile
            self._screen_profile = screen
        except Exception as exc:
            print(f"[OpenXRViewer] _save_curve_to_active_profile failed: {exc}")

    # ------------------------------------------------------------------
    # Dead stub: kept for backward compatibility with callers that still
    # reference _switch_environment.  Delegates to the new pipeline.
    # ------------------------------------------------------------------
    def _switch_environment(self, name, *, save_outgoing=True, apply_profile=True):
        if name is None:
            self._release_env_model_resources()
            self._active_environment = None
            self._glow_intensity_multiplier = 0.0
            self._persist_runtime_settings()
            return
        self._switch_environment_model(model_name=name)


    def _load_brand_models(self, brand_name):
        """Load models and configuration for a specific brand, returning {prims_l, prims_r, tex_cache, offset, rot_deg}."""
        base_dir = os.path.join(self._controllers_root, brand_name)
        result = {
            'prims_l': [], 'prims_r': [], 'tex_cache': {},
            'offset': [0.0, 0.0, 0.0], 'rot_deg': 0.0,
        }
        # Read profile.json
        profile_path = os.path.join(base_dir, 'profile.json')
        if os.path.isfile(profile_path):
            try:
                import json as _json
                with open(profile_path, 'r') as f:
                    prof = _json.load(f)
                overrides = prof.get('overrides', {})
                if overrides.get('model_offset'):
                    result['offset'] = [float(v) for v in overrides['model_offset']]
                if 'model_rotation_deg' in overrides:
                    result['rot_deg'] = float(overrides['model_rotation_deg'])
            except Exception as e:
                print(f"[OpenXRViewer] Failed to read {profile_path}: {e}")

        _dir_key = brand_name

        def _create_prims(glb_path, target_list):
            prims_data, textures, _lights = load_glb_model(glb_path)
            _file_stem = os.path.splitext(os.path.basename(glb_path))[0]
            _prefix = f"{_dir_key}/{_file_stem}"
            for tid, tex_arr in enumerate(textures):
                if tex_arr is not None:
                    cache_key = f"{_prefix}:{tid}"
                    if cache_key not in result['tex_cache']:
                        h, w = tex_arr.shape[:2]
                        mtex = self.ctx.texture((w, h), 4, tex_arr.tobytes())
                        mtex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
                        mtex.build_mipmaps()
                        result['tex_cache'][cache_key] = mtex
            for pd in prims_data:
                vbo = self.ctx.buffer(pd['vertices'].tobytes())
                ibo = self.ctx.buffer(pd['indices'].tobytes())
                vao = self.ctx.vertex_array(
                    self._controller_prog,
                    [(vbo, '3f 3f 2f', 'in_position', 'in_normal', 'in_uv')],
                    ibo,
                )
                target_list.append({
                    'vao': vao, 'vbo': vbo, 'ibo': ibo,
                    'tex_key': f"{_prefix}:{pd['tex_id']}" if pd['tex_id'] >= 0 else None,
                    'tri_count': len(pd['indices']) // 3,
                })

        try:
            _create_prims(os.path.join(base_dir, 'right.glb'), result['prims_r'])
            _create_prims(os.path.join(base_dir, 'left.glb'),  result['prims_l'])
            result['prims_r'].sort(key=lambda p: p['tri_count'], reverse=True)
            result['prims_l'].sort(key=lambda p: p['tri_count'], reverse=True)
        except Exception as e:
            print(f"[OpenXRViewer] {brand_name} model load failed: {e}")
            result['prims_l'], result['prims_r'] = [], []
        return result

    def _init_all_controller_models(self):
        """Preload all controller brand models under controllers/."""
        if not os.path.isdir(self._controllers_root):
            return
        brands = sorted(d for d in os.listdir(self._controllers_root)
                    if os.path.isdir(os.path.join(self._controllers_root, d)))
        for bn in brands:
            model = self._load_brand_models(bn)
            self._all_models[bn] = model
            self._available_brands.append(bn)
        # Set default brand
        default = self._controller_model if self._controller_model in self._all_models else (
            self._available_brands[0] if self._available_brands else None)
        if default is None:
            print("[OpenXRViewer] No controller brands available!")
            return
        self._switch_brand(default)
        print(f"[OpenXRViewer] Controller: {self._current_brand}")

    def _switch_brand(self, brand_name):
        """Switch controller brand with zero latency."""
        if brand_name not in self._all_models:
            return
        m = self._all_models[brand_name]
        self._ctrl_prims_l      = m['prims_l']
        self._ctrl_prims_r      = m['prims_r']
        self._ctrl_tex_cache    = m['tex_cache']
        self._ctrl_model_offset = m['offset']
        self._ctrl_model_rot_deg = m['rot_deg']
        self._current_brand      = brand_name
        self._brand_switch_osd_t = time.perf_counter()
        # Persist the new brand to settings.yaml so the next launch (GUI or
        # standalone) picks the same one without further user input.
        self._persist_setting('Controller Model', brand_name)
        print(f"[OpenXRViewer] Switched to: {brand_name} "
            f"offset={self._ctrl_model_offset} rot={self._ctrl_model_rot_deg}")

    def _env_model_mat4(self):
        """Return model->world transform for the Environment Model (cached by pose/scale)."""
        cache_key = (
            tuple(self._env_model_pos),
            tuple(self._env_model_rot),
            tuple(self._env_model_scale),
        )
        if getattr(self, '_env_model_mat4_key', None) == cache_key:
            return self._cached_env_model_mat4_val
        yaw, pitch, roll = self._env_model_rot
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)
        sx, sy_s, sz = self._env_model_scale
        tx, ty, tz = self._env_model_pos
        m = np.array([
            [(cy*cr + sy*sp*sr)*sx, (-cy*sr + sy*sp*cr)*sy_s, sy*cp*sz,              tx],
            [cp*sr*sx,               cp*cr*sy_s,              -sp*sz,                 ty],
            [(-sy*cr + cy*sp*sr)*sx, (sy*sr + cy*sp*cr)*sy_s,  cy*cp*sz,              tz],
            [0,                      0,                        0,                      1],
        ], dtype='f4')
        m.flags.writeable = False
        self._cached_env_model_mat4_val = m
        self._env_model_mat4_key = cache_key
        self._env_normal_mat_key = None  # invalidate normal matrix cache
        return m

    def _render_env_model(self, mgl_fbo, vp_mat, view_mat, view_inv=None):
        """Render the glTF environment model in world space.

        Must be called from ``_render_eye`` **before** any other geometry so the
        environment acts as the background layer.  Uses a dedicated env shader
        (no ``gl_FrontFacing discard``, double-sided lighting) so all faces of
        glTF models are rendered correctly without winding tricks.
        """
        if not self._env_model_visible or not self._env_model_prims:
            return

        model_mat = self._env_model_mat4()

        if view_inv is None:
            view_inv = _view_mat_inv(view_mat)
        cam_pos = view_inv[:3, 3]

        self._env_prog['u_mvp'].write(vp_mat.T.tobytes())
        self._env_prog['u_model'].write(model_mat.T.tobytes())
        _env_nm_key = id(model_mat)
        if getattr(self, '_env_normal_mat_key', None) != _env_nm_key:
            normal_mat3 = np.linalg.inv(model_mat[:3, :3].astype(np.float32).T)
            self._env_normal_mat_bytes = normal_mat3.T.tobytes()
            self._env_normal_mat_key = _env_nm_key
        self._env_prog['u_normal_mat'].write(self._env_normal_mat_bytes)
        self._env_prog['u_camera_pos'].write((cam_pos).tobytes())
        # Light colors: Dark Room / Cinema use near-zero head-lamp (they rely
        # on cinema bias light / screen spill).  All other environments read
        # from profile-backed attributes so each room can tune its own look.
        _hlc = getattr(self, '_env_head_light_color', (0.45, 0.45, 0.48))
        _amb = getattr(self, '_env_ambient_color', (0.08, 0.08, 0.09))
        if self._active_environment == 'Dark Room':
            self._env_prog['u_light_color'].value = (0.003, 0.003, 0.003)
            self._env_prog['u_ambient_color'].value = (0.0024, 0.0024, 0.0027)
        else:
            self._env_prog['u_light_color'].value = (
                float(_hlc[0]), float(_hlc[1]), float(_hlc[2]))
            self._env_prog['u_ambient_color'].value = (
                float(_amb[0]), float(_amb[1]), float(_amb[2]))
        # Directional light (KHR_lights_punctual) — cached since model rotation is static
        if self._scene_lights:
            cached = getattr(self, '_cached_env_light', None)
            if cached is None or cached[0] is not model_mat:
                dl = self._scene_lights[0]
                ldir = model_mat[:3, :3].astype('f8') @ dl['direction'].astype('f8')
                ldir /= np.linalg.norm(ldir) + 1e-12
                ci = dl['color'] * dl['intensity']
                self._cached_env_light = (
                    model_mat,
                    (float(ldir[0]), float(ldir[1]), float(ldir[2])),
                    (float(ci[0]), float(ci[1]), float(ci[2])),
                )
            self._env_prog['u_light_dir'].value = self._cached_env_light[1]
            self._env_prog['u_light_intensity'].value = self._cached_env_light[2]
        else:
            self._env_prog['u_light_intensity'].value = (0.0, 0.0, 0.0)

        # Fill lights from profile env_fill_lights (up to 2)
        _fl = getattr(self, '_env_fill_lights', [])
        for i in range(2):
            if i < len(_fl):
                fl = _fl[i]
                pos = fl.get('position', (0, 0, 0))
                col = fl.get('color', (0, 0, 0))
                rng = float(fl.get('range', 1.0))
                self._env_prog[f'u_fill_light_enabled{i}'].value = 1
                self._env_prog[f'u_fill_light_pos{i}'].value = (
                    float(pos[0]), float(pos[1]), float(pos[2]))
                self._env_prog[f'u_fill_light_color{i}'].value = (
                    float(col[0]), float(col[1]), float(col[2]))
                self._env_prog[f'u_fill_light_range{i}'].value = rng
            else:
                self._env_prog[f'u_fill_light_enabled{i}'].value = 0

        # Exposure / gamma / emissive from profile (fall back to defaults)
        self._env_prog['u_env_exposure'].value = float(
            getattr(self, '_env_exposure', 1.0))
        self._env_prog['u_env_gamma'].value = float(
            getattr(self, '_env_gamma', 2.2))
        self._env_prog['u_emissive_strength'].value = float(
            getattr(self, '_env_emissive_strength', 1.0))

        # ----- Cinema bias light (screen as rectangular area light) ---------
        # Builds a world-space orthonormal basis for the screen from
        # screen_pan_xyz + screen_yaw/pitch/roll, then writes uniforms the env
        # shader uses to add a Lambertian, forward-hemisphere area-light
        # contribution.  Follows the Meta Horizon lighting design page:
        #   * Emissive screen + Area light category
        #   * Lambertian diffuse for soft, gradual response
        #   * Sampled frame-average colour ties light hue to actual content
        #   * Single light, no extra texture samples -> shader stays cheap
        self._apply_cinema_light_uniforms()

        # Disable back-face culling so both sides of walls/objects are visible.
        # The env shader flips the normal for back-faces (double-sided lighting).
        # Explicitly set GL_CCW winding so glTF models use standard convention.
        self.ctx.disable(moderngl.CULL_FACE)
        glFrontFace(GL_CCW)

        ep = self._env_prog
        last_blend = False
        tc = self._env_model_tex_cache
        for prim in self._env_model_prims:
            rs = prim.get('_rs')
            if rs is None:
                continue
            ep['u_base_color_factor'].value = rs['bc']
            ep['u_base_alpha'].value = rs['ba']
            ep['u_roughness'].value = rs['rf']
            ep['u_metallic'].value = rs['mf']
            ep['u_emissive_factor'].value = rs['ef']
            ep['u_unlit'].value = rs['unlit']
            ep['u_alpha_mode'].value = rs['am']
            ep['u_alpha_cutoff'].value = rs['ac']
            need_blend = rs['blend']
            if need_blend != last_blend:
                if need_blend:
                    self.ctx.enable(moderngl.BLEND)
                    self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
                    self.ctx.depth_mask = False
                else:
                    self.ctx.disable(moderngl.BLEND)
                    self.ctx.depth_mask = True
                last_blend = need_blend
            ep['u_tex_offset'].value = rs['to']
            ep['u_tex_scale'].value = rs['ts']
            tk = rs['tk']
            if tk and tk in tc:
                tc[tk].use(location=3)
                ep['u_use_texture'].value = 1
            else:
                ep['u_use_texture'].value = 0
            n_key = rs['n_key']
            if n_key and n_key in tc:
                tc[n_key].use(location=4)
                ep['u_use_normal_tex'].value = 1
                ep['u_normal_scale'].value = rs['ns']
            else:
                ep['u_use_normal_tex'].value = 0
            o_key = rs['o_key']
            if o_key and o_key in tc:
                tc[o_key].use(location=5)
                ep['u_use_occlusion_tex'].value = 1
                ep['u_occlusion_strength'].value = rs['os']
            else:
                ep['u_use_occlusion_tex'].value = 0
            m_key = rs['m_key']
            if m_key and m_key in tc:
                tc[m_key].use(location=6)
                ep['u_use_mr_tex'].value = 1
            else:
                ep['u_use_mr_tex'].value = 0
            e_key = rs['e_key']
            if e_key and e_key in tc:
                tc[e_key].use(location=7)
                ep['u_use_emissive_tex'].value = 1
            else:
                ep['u_use_emissive_tex'].value = 0
            prim['vao'].render(moderngl.TRIANGLES)

        # Reset state for subsequent rendering
        self.ctx.disable(moderngl.BLEND)
        self.ctx.depth_mask = True
        self._env_prog['u_use_texture'].value = 1
        self._env_prog['u_base_color_factor'].value = (1.0, 1.0, 1.0)
        self._env_prog['u_base_alpha'].value = 1.0

    def _apply_cinema_light_uniforms(self):
        """Push current cinema bias-light uniforms to ``self._env_prog``.

        Shared by ``_render_env_model`` and ``_render_dark_room`` so both
        paths get exactly the same physically-grounded screen→world light
        contribution (Meta Horizon "Emissive + Area light", Lambertian
        diffuse, forward 180-degree hemisphere only).  The env shader's
        rectangular area-light block reads these uniforms each fragment.
        """
        if self.screen_height is None or self._screen_light_intensity <= 0.0:
            self._env_prog['u_screen_light_enabled'].value = 0
            self._cl_light_state_key = None
            self._cl_uniform_frame = -5
            return
        fc = getattr(self, '_frame_count', 0)
        _pose_key = (self.screen_yaw, self.screen_pitch, self.screen_roll,
                     self.screen_pan_x, self.screen_pan_y, self.screen_distance,
                     self.screen_width, self.screen_height)
        if _pose_key != getattr(self, '_cl_pose_key', None):
            sx_pos = float(self.screen_pan_x)
            sy_pos = float(self.screen_pan_y)
            sz_pos = float(-self.screen_distance)
            cy = math.cos(self.screen_yaw);   sy_ = math.sin(self.screen_yaw)
            cp = math.cos(self.screen_pitch); sp = math.sin(self.screen_pitch)
            cr = math.cos(self.screen_roll);  sr = math.sin(self.screen_roll)
            nz0 = sy_ * cp
            nz1 = -sp
            nz2 = cy * cp
            self._cl_pos = (sx_pos, sy_pos, sz_pos)
            self._cl_normal = (nz0, nz1, nz2)
            self._cl_half = (float(self.screen_width) * 0.5, float(self.screen_height) * 0.5)
            self._cl_pose_key = _pose_key
        state_key = (_pose_key, getattr(self, '_active_environment', None),
                     float(self._screen_light_intensity))
        last_state_key = getattr(self, '_cl_light_state_key', None)
        last_frame = getattr(self, '_cl_uniform_frame', -999)
        if state_key == last_state_key and (fc - last_frame) < 5:
            return
        self._cl_light_state_key = state_key
        self._cl_uniform_frame = fc
        self._advance_glow_color(lerp=0.14)
        sc = self._glow_color
        intensity = float(self._screen_light_intensity)
        if self._active_environment == 'Dark Room':
            intensity *= 0.9
        self._env_prog['u_screen_light_enabled'].value   = 1
        self._env_prog['u_screen_light_pos'].value       = self._cl_pos
        self._env_prog['u_screen_light_normal'].value    = self._cl_normal
        self._env_prog['u_screen_light_half_size'].value = self._cl_half
        self._env_prog['u_screen_light_color'].value     = (float(sc[0]), float(sc[1]), float(sc[2]))
        self._env_prog['u_screen_light_intensity'].value = intensity

