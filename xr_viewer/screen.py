"""ScreenMixin: virtual screen placement, pose, animation, and seating."""

import math
import sys
import time
import json
import os

import numpy as np

try:
    import xr
except ImportError:
    xr = None
from OpenGL.GL import (
    glGenFramebuffers, glBindFramebuffer, glFramebufferTexture2D,
    glCheckFramebufferStatus, GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
    GL_TEXTURE_2D, GL_FRAMEBUFFER_COMPLETE,
    glGenRenderbuffers, glBindRenderbuffer, glRenderbufferStorage,
    glFramebufferRenderbuffer, glDeleteRenderbuffers,
    GL_RENDERBUFFER, GL_DEPTH_ATTACHMENT, GL_DEPTH_COMPONENT24,
)

from .render import _euler_to_mat4, _view_mat_inv, _xr_pose_to_model_mat4, _mat4_to_xr_posef, _xr_quat_to_mat4
from .constants import _SCREEN_ENV_DEPTH_BIAS_M, _CURVED_CURVATURE_SCALE, _CURVED_HALF_ANGLE_RAD


class ScreenMixin:
    def _build_model_mat4(self, normal_offset=0.0):
        """
        Build the model matrix for the screen in world space.
        Caller must transpose before writing to OpenGL.
        """
        if self.screen_height is None:
            self.screen_width, self.screen_height = self._crop_screen_dims(self._screen_ref_size)

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
            self.screen_width, self.screen_height = self._crop_screen_dims(self._screen_ref_size)

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

