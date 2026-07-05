"""LaserMixin: controller laser/ray, hit detection, smoothing, calibration, laser UV."""

import math
import time
import ctypes
import json

import os
import os as _os

import numpy as np
import moderngl
from OpenGL.GL import glFrontFace, GL_CW, GL_CCW

from .render import _view_mat_inv
from .constants import (
    DEAD, EDGE_STRENGTH, KB_CURSOR_PRIORITY_BIAS, KB_CURSOR_RELEASE_GRACE,
    _CURVED_HALF_ANGLE_RAD,
)
from .input import OneEuroFilter3D, EMAPositionFilter

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


class LaserMixin:
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
            smoothed_origin = ctrl_pos  # pre-override smoothed pos (for hit circles)

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

            # Ray-target intersections: compute ONCE per frame here (inputs are
            # identical for both eyes), stored in the beam so _render_lasers and
            # _render_laser_hit_circles reuse them instead of recomputing 4x.
            kb_dist = self._keyboard_laser_hit_dist(ctrl_pos, fwd_w)
            sc_dist = self._laser_screen_hit_dist(ctrl_pos, fwd_w)
            _bov_cp, _bov_fw = self._pre_snap_overlay_ray(is_left, aim_mat, grip_mat)
            ov_dist = self._overlay_panel_hit_dist(_bov_cp, _bov_fw)
            beams.append((now, ctrl_name, aim_mat, ctrl_pos, fwd_w, right2, fwd, up,
                          kb_dist, sc_dist, ov_dist, _bov_cp, _bov_fw, smoothed_origin))
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
        for (now, ctrl_name, aim_mat, ctrl_pos, fwd_w, right2, fwd, up,
             kb_dist, sc_dist, ov_dist, _bov_cp, _bov_fw, _sm_origin) in beams:
            # Auto-shorten beam when target is closer than the default length
            # mirrors the priority logic in _render_laser_hit_circles so the
            # beam tip lines up with the hit circle (no over/undershoot).
            # Hit distances precomputed in _laser_beam_setup (1x/frame).
            cursor_uv = self._cursor_uv_l if ctrl_name == 'left' else self._cursor_uv_r
            if (self._cursor_ctrl == ctrl_name and cursor_uv is not None):
                hit_dist = max(0.01, float(cursor_uv[2]))
            else:
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

        for (now, ctrl_name, aim_mat, ctrl_pos, fwd_w, right2, fwd, up,
             kb_dist, sc_dist, ov_dist, _ov_cp, _ov_fw, _sm_pos) in beams:
            # All hit distances + overlay pre-snap ray precomputed once per frame
            # in _laser_beam_setup; _sm_pos is the pre-override smoothed origin.

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

