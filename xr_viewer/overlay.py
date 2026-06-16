"""OSD overlay, help panel, keyboard, and calibration UI rendering mixin."""

import math
import os

from utils import ROWS

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_MODULE_DIR)
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from OpenGL.GL import (
    glGenFramebuffers, glBindFramebuffer, glFramebufferTexture2D,
    GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
    glBindTexture, glTexSubImage2D, glGenerateMipmap,
    GL_RGB, GL_RGBA, GL_UNSIGNED_BYTE, GL_FLOAT,
    glEnable, glDisable, GL_BLEND, GL_CULL_FACE,
)

import moderngl

from .constants import (
    _KB_UNITS_WIDE, _KB_ROWS, _KeyEntry, _KB_TEX_W, _KB_TEX_H,
)


class OverlayMixin:
    """Overlay, OSD, keyboard, and calibration panel rendering.

    Expects the owning class to provide:
      self._overlay_prog, self._overlay_vao, self._overlay_tex, self._overlay_tex_size,
      self._fps_overlay_visible, self.show_fps, self.actual_fps, self.sbs_fps,
      self.total_latency, self._cached_actual_fps, self._cached_sbs_fps,
      self._cached_latency, self._cached_screen_width, self._cached_screen_height,
      self._cached_screen_dist, self._cached_screen_curved, self._cached_depth_ratio,
      self._cached_vr_res, self._cached_sbs_res, self._fps_display_ema,
      self._frame_ts_ring, self._sbs_ts_ring, self._last_overlay_update,
      self.font, self.bold_font, self.label_font, self.base_font_size,
      self._depth_osd_tex, self._depth_osd_vao, self._depth_osd_tex_size,
      self._depth_osd_alpha, self._depth_osd_show_t, self._depth_osd_last_val,
      self.depth_ratio, self.depth_strength,
      self._preset_osd_tex, self._preset_osd_vao, self._preset_osd_tex_size,
      self._preset_osd_alpha, self._preset_osd_show_t, self._preset_osd_last_key,
      self._preset_index, self._screen_presets,
      self._screen_osd_tex, self._screen_osd_vao, self._screen_osd_tex_size,
      self._screen_osd_show_t, self._screen_osd_last_key,
      self.screen_width, self.screen_height, self.screen_distance,
      self._screen_curved, self._preset_name_overlay,
      self._brand_osd_tex, self._brand_osd_vao, self._brand_osd_tex_size,
      self._brand_osd_alpha, self._brand_osd_show_t, self._brand_osd_last_name,
      self._current_brand,
      self._seat_adjust_osd_tex, self._seat_adjust_osd_vao,
      self._seat_adjust_osd_tex_size, self._seat_adjust_osd_show_t,
      self._seat_adjust_osd_alpha, self._seat_adjust_osd_dirty,
      self._seat_adjust_active, self._seat_adjust_current_pos,
      self._calib_tex, self._calib_tex_size, self._calibration_mode,
      self._calibration_temp_offset, self._calibration_temp_rot,
      self._help_tex, self._help_vao, self._help_tex_size,
      self._help_panel_visible,
      self._keyboard_visible, self._keyboard_tex, self._keyboard_vao,
      self._keyboard_keys, self._keyboard_width, self._keyboard_height,
      self._kb_show_shifted, self._kb_last_build_width,
      self._keyboard_pan_x, self._keyboard_pan_y, self._keyboard_distance,
      self._keyboard_pitch, self._keyboard_yaw,
      self._mod_state, self._caps_lock,
      self._head_pos_w, self._head_fwd_w,
      self.ctx,
      self.ipd_uv, self.frame_size, self._screen_visible,
      self._screen_yaw, self._screen_pitch, self._screen_roll,
      self._screen_pan_x, self._screen_pan_y, self._corner_radius,
      self._bg_color_idx, self._BG_COLORS,
      self._screen_eye_init, self._head_pos_w, self._head_fwd_w, self._initial_head_y,
    """


    def _render_fps_overlay(self, eye_index, mgl_fbo, vp_mat):
        """Render the FPS/latency text quad on the top-left of the screen surface."""
        if self.screen_height is None:
            return

        now = self._frame_now

        # Update cached values once per second
        if now - self._last_overlay_update >= 1.0:
            self._cached_actual_fps    = self.actual_fps
            self._cached_sbs_fps       = self.sbs_fps
            self._cached_latency       = self.total_latency
            self._cached_screen_width  = self.screen_width
            self._cached_screen_height = self.screen_height if self.screen_height is not None else 0.0
            self._cached_screen_dist   = self.screen_distance
            self._cached_screen_curved = self._screen_curved
            self._cached_depth_ratio   = self.depth_ratio
            self._cached_help_visible  = self._help_panel_visible
            self._cached_vr_res        = self._swapchain_sizes.get(0, (0, 0))
            self._cached_sbs_res       = self.frame_size
            self._last_overlay_update  = now

            # Rebuild text texture left eye only
            if eye_index == 0 and self.font is not None:
                ow, oh = self._overlay_tex_size
                img  = Image.new('RGBA', (ow, oh), (0, 0, 0, 0))
                draw = ImageDraw.Draw(img)

                draw.rounded_rectangle(
                    [0, 0, ow - 1, oh - 1],
                    radius=14,
                    fill=(32, 32, 36, 210),
                )

                C_LABEL = (150, 158, 185, 255)
                C_GREEN = (  0, 230,  90, 255)
                C_CYAN  = (  0, 210, 230, 255)
                C_AMBER = (255, 190,  40, 255)
                label_font = self.bold_font or self.font
                value_font = self.font

                PAD    = 14
                ROW0   = 22
                ROW1   = 56
                ROW2   = 90
                ROW3   = 124
                ROW4   = 158

                # Compute per-font ascent so both columns sit on the same baseline.
                # getmetrics() returns (ascent, descent); fall back to 0 if unavailable.
                def _ascent(f):
                    try:
                        return f.getmetrics()[0]
                    except Exception:
                        return 0
                lbl_asc = _ascent(label_font)
                val_asc = _ascent(value_font)
                # Positive offset shifts the shorter font down to align cap-heights.
                lbl_dy = max(0, val_asc - lbl_asc)
                val_dy = max(0, lbl_asc - val_asc)

                labels = ["[Performance]", "[3D Display]", "[Resolution]", "[Show Shortcuts]", "[Models]"]
                try:
                    max_lw = max(int(draw.textlength(l, font=label_font)) for l in labels)
                except AttributeError:
                    max_lw = max(
                        (int(label_font.getsize(l)[0]) if hasattr(label_font, 'getsize') else 190)
                        for l in labels
                    )
                VAL_X = PAD + max_lw + 10

                def _draw_row(y, label, label_color, value, value_color):
                    draw.text((PAD,   y + lbl_dy), label, font=label_font, fill=label_color)
                    draw.text((VAL_X, y + val_dy), value, font=value_font, fill=value_color)

                lat_str = f"{self._cached_latency:.0f}ms" if self._cached_latency > 0 else "--"
                fps_str = (f"XR {self._cached_actual_fps:.0f} FPS"
                          f"   SBS {self._cached_sbs_fps:.0f} FPS"
                          f"   Latency {lat_str}")
                _draw_row(ROW0, "[Performance]", C_LABEL, fps_str, C_GREEN)
                if self._current_brand:
                    env_str = (self._active_environment or self._environment_model or 'Default').strip() or 'Default'
                    model_str = f"Environment: {env_str}   Controller: {self._current_brand}"
                    _draw_row(ROW3, "[Models]", C_LABEL, model_str, C_CYAN)
                scr_str = (f"{self._cached_screen_width:.2f}"
                          f" x {self._cached_screen_height:.2f} m"
                          f"  @  {self._cached_screen_dist:.2f} m"
                          f"   Depth {self._cached_depth_ratio:.2f}")
                _draw_row(ROW1, "[3D Display]", C_LABEL, scr_str, C_CYAN)

                vw, vh = self._cached_vr_res
                sw, sh = self._cached_sbs_res
                res_str = f"XR {vw}x{vh}/eye   Screen {sw}x{sh}"
                _draw_row(ROW2, "[Resolution]", C_LABEL, res_str, C_AMBER)

                draw.text((PAD, ROW4 + lbl_dy), "[Show Shortcuts]", font=label_font, fill=C_LABEL)
                # Toggle switch: pill track + circle knob
                SW_W, SW_H = 52, 26          # track width, height
                SW_X = VAL_X                 # left edge of switch
                SW_Y = ROW4 + (34 - SW_H) // 2  # vertically centred in row
                on   = self._cached_help_visible
                track_col = (0, 200, 80, 255) if on else (80, 84, 100, 255)
                draw.rounded_rectangle([SW_X, SW_Y, SW_X + SW_W, SW_Y + SW_H],
                                       radius=SW_H // 2, fill=track_col)
                KR   = SW_H // 2 - 2         # knob radius
                KX   = (SW_X + SW_W - KR - 3) if on else (SW_X + KR + 3)
                KY   = SW_Y + SW_H // 2
                draw.ellipse([KX - KR, KY - KR, KX + KR, KY + KR],
                             fill=(255, 255, 255, 255))

                data = np.flipud(np.array(img, dtype=np.uint8))
                self._overlay_tex.write(data.tobytes())

        # Position below the screen bottom edge, same plane, left-aligned
        sh = self.screen_height
        sx = self.screen_width / 2.0
        sy = sh / 2.0
        GAP = sh * 0.02  # proportional gap scales with screen size

        ow, oh = self._overlay_tex_size
        OVERLAY_H = sh / 8.0
        OVERLAY_W = OVERLAY_H * (ow / oh)

        # Below bottom edge: cy = -sy - GAP - OVERLAY_H/2
        # Left edge flush with screen left edge
        local_cx = -sx + OVERLAY_W / 2.0
        local_cy = -sy - GAP - OVERLAY_H / 2.0

        # Screen rotation + translation (reuse _build_model_mat4 pattern)
        cy_s = math.cos(self.screen_yaw);   sy_s = math.sin(self.screen_yaw)
        cp_s = math.cos(self.screen_pitch); sp_s = math.sin(self.screen_pitch)
        R = (np.array([[ cy_s,  0, sy_s, 0],
                       [    0,  1,     0, 0],
                       [-sy_s,  0, cy_s,  0],
                       [    0,  0,     0, 1]], dtype=np.float32) @
             np.array([[1,    0,     0, 0],
                       [0, cp_s, -sp_s, 0],
                       [0, sp_s,  cp_s, 0],
                       [0,    0,     0, 1]], dtype=np.float32))
        T = np.eye(4, dtype=np.float32)
        T[0, 3] = self.screen_pan_x
        T[1, 3] = self.screen_pan_y
        T[2, 3] = -self.screen_distance

        S_ov = np.diag([OVERLAY_W / 2.0, OVERLAY_H / 2.0, 1.0, 1.0]).astype(np.float32)
        T_local = np.eye(4, dtype=np.float32)
        T_local[0, 3] = local_cx
        T_local[1, 3] = local_cy
        model = T @ R @ T_local @ S_ov

        # Fade panel while trigger is held on it; snap back to full immediately on release.
        trigger_held = self._ov_ltrig_held or self._ov_rtrig_held
        if trigger_held:
            FADE_SPEED = 8.0  # alpha units/sec toward faded (smooth fade-in)
            dt = max(0.001, getattr(self, '_last_frame_dt', 0.016))
            self._status_panel_alpha = max(0.15, self._status_panel_alpha - FADE_SPEED * dt)
        else:
            self._status_panel_alpha = 1.0  # instant recovery on trigger release

        mvp = vp_mat @ model
        mgl_fbo.use()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._overlay_tex.use(location=2)
        self._overlay_prog['u_mvp'].write(mvp.T.tobytes())
        self._overlay_prog['u_alpha'].value = self._status_panel_alpha
        self._overlay_vao.render(moderngl.TRIANGLE_STRIP)
        self._overlay_prog['u_alpha'].value = 1.0   # restore for other overlays
        self.ctx.disable(moderngl.BLEND)

    def _render_depth_osd(self, eye_index, mgl_fbo, vp_mat):
        """Floating depth-ratio indicator panel.

        Appears when depth_ratio changes; fades out automatically.
        Floats in front of the screen; distance from screen grows with depth_ratio.
        Style matches the FPS status panel (dark rounded rectangle).
        """
        if self._depth_osd_tex is None or self.screen_height is None:
            return

        now = self._frame_now

        # Detect change and (re)trigger on left eye only to avoid double writes.
        # Composite key encodes both depth_ratio and the on/off state from
        # depth_strength so the right-grip + left-stick-click toggle also
        # fires the indicator.
        if eye_index == 0:
            depth_on = self.depth_strength > 0.0
            cur_val  = round(self.depth_ratio, 3)
            cur_key  = (cur_val, depth_on)
            if cur_key != self._depth_osd_last_val:
                self._depth_osd_last_val = cur_key
                self._depth_osd_show_t   = now
                self._depth_osd_alpha    = 1.0

                # Rebuild PIL texture
                if self.font is not None:
                    dw, dh = self._depth_osd_tex_size
                    img  = Image.new('RGBA', (dw, dh), (0, 0, 0, 0))
                    draw = ImageDraw.Draw(img)
                    draw.rounded_rectangle(
                        [0, 0, dw - 1, dh - 1],
                        radius=12,
                        fill=(32, 32, 36, 210),
                    )
                    label = "Depth"
                    value = f"{cur_val:.2f}" if depth_on else "0.00"
                    bfont = self.bold_font or self.font
                    C_LABEL = (150, 158, 185, 255)
                    C_VALUE = (  0, 210, 230, 255)
                    PAD = 12
                    cy  = (dh - 32) // 2
                    draw.text((PAD, cy), label, font=bfont, fill=C_LABEL)
                    try:
                        lw = int(draw.textlength(label, font=bfont))
                    except AttributeError:
                        lw = int(bfont.getsize(label)[0]) if hasattr(bfont, 'getsize') else 60
                    draw.text((PAD + lw + 8, cy), value, font=self.font, fill=C_VALUE)
                    data = np.flipud(np.array(img, dtype=np.uint8))
                    self._depth_osd_tex.write(data.tobytes())

        # Fade: hold for 1.5 s then decay over 0.8 s
        HOLD  = 1.5
        DECAY = 0.8
        elapsed = now - self._depth_osd_show_t
        if elapsed < HOLD:
            alpha = 1.0
        elif elapsed < HOLD + DECAY:
            alpha = 1.0 - (elapsed - HOLD) / DECAY
        else:
            alpha = 0.0
        self._depth_osd_alpha = alpha

        if alpha <= 0.0:
            return

        # Screen-relative positioning above the screen; text size scales with screen width
        OSD_H = self.screen_width * 0.03  # scales with screen size (0.072m @ 2.4m default)
        dw, dh = self._depth_osd_tex_size
        OSD_W = OSD_H * (dw / dh)

        BASE_Z_EXTRA = 0.05
        SCALE_Z      = 0.028
        z_extra = BASE_Z_EXTRA + self.depth_ratio * SCALE_Z
        dist = self.screen_distance - z_extra

        cy_  = math.cos(self.screen_yaw);   sy_ = math.sin(self.screen_yaw)
        cp   = math.cos(self.screen_pitch); sp  = math.sin(self.screen_pitch)
        rot_y = np.array([
            [ cy_,  0, sy_, 0],
            [  0,   1,   0, 0],
            [-sy_,  0, cy_, 0],
            [  0,   0,   0, 1],
        ], dtype=np.float32)
        rot_x = np.array([
            [1,  0,   0, 0],
            [0,  cp, -sp, 0],
            [0,  sp,  cp, 0],
            [0,  0,   0,  1],
        ], dtype=np.float32)

        y_pos = self.screen_pan_y + self.screen_height / 2.0 + self.screen_width * 0.016 + OSD_H / 2.0
        S = np.diag([OSD_W / 2.0, OSD_H / 2.0, 1.0, 1.0]).astype(np.float32)
        R = rot_y @ rot_x
        T = np.eye(4, dtype=np.float32)
        T[0, 3] = self.screen_pan_x; T[1, 3] = y_pos; T[2, 3] = -dist
        mvp = vp_mat @ T @ R @ S

        mgl_fbo.use()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._depth_osd_tex.use(location=2)
        self._overlay_prog['u_mvp'].write(mvp.T.tobytes())
        self._overlay_prog['u_alpha'].value = alpha
        self._depth_osd_vao.render(moderngl.TRIANGLE_STRIP)
        self._overlay_prog['u_alpha'].value = 1.0   # restore for other overlays
        self.ctx.disable(moderngl.BLEND)

    def _render_preset_osd(self, eye_index, mgl_fbo, vp_mat):
        """Floating preset-name indicator panel.

        Appears when the user cycles screen presets (Y button); fades out automatically.
        Same style as the depth-ratio OSD.
        """
        if self._preset_osd_tex is None or self.screen_height is None:
            return

        now = self._frame_now

        if eye_index == 0:
            cur_key = self._preset_index
            if cur_key != self._preset_osd_last_key:
                self._preset_osd_last_key = cur_key
                self._preset_osd_show_t   = now
                self._preset_osd_alpha    = 1.0

                if self.font is not None and self._preset_name_overlay:
                    dw, dh = self._preset_osd_tex_size
                    img  = Image.new('RGBA', (dw, dh), (0, 0, 0, 0))
                    draw = ImageDraw.Draw(img)
                    draw.rounded_rectangle(
                        [0, 0, dw - 1, dh - 1],
                        radius=12,
                        fill=(32, 32, 36, 210),
                    )
                    bfont = self.bold_font or self.font
                    C_LABEL = (150, 158, 185, 255)
                    C_VALUE = (  0, 210, 230, 255)
                    PAD = 12
                    cy  = (dh - 32) // 2
                    draw.text((PAD, cy), "Preset", font=bfont, fill=C_LABEL)
                    try:
                        lw = int(draw.textlength("Preset", font=bfont))
                    except AttributeError:
                        lw = int(bfont.getsize("Preset")[0]) if hasattr(bfont, 'getsize') else 60
                    draw.text((PAD + lw + 8, cy), self._preset_name_overlay,
                            font=self.font, fill=C_VALUE)
                    data = np.flipud(np.array(img, dtype=np.uint8))
                    self._preset_osd_tex.write(data.tobytes())

        # Fade: hold 1.5 s then decay over 0.8 s
        HOLD  = 1.5
        DECAY = 0.8
        elapsed = now - self._preset_osd_show_t
        if elapsed < HOLD:
            alpha = 1.0
        elif elapsed < HOLD + DECAY:
            alpha = 1.0 - (elapsed - HOLD) / DECAY
        else:
            alpha = 0.0
        self._preset_osd_alpha = alpha

        if alpha <= 0.0:
            return

        # Screen-relative positioning; text size scales with screen width
        OSD_H = self.screen_width * 0.03
        dw, dh = self._preset_osd_tex_size
        OSD_W  = OSD_H * (dw / dh)

        cy_  = math.cos(self.screen_yaw);   sy_ = math.sin(self.screen_yaw)
        cp   = math.cos(self.screen_pitch); sp  = math.sin(self.screen_pitch)
        rot_y = np.array([
            [ cy_,  0, sy_, 0],
            [  0,   1,   0, 0],
            [-sy_,  0, cy_, 0],
            [  0,   0,   0, 1],
        ], dtype=np.float32)
        rot_x = np.array([
            [1,  0,   0, 0],
            [0,  cp, -sp, 0],
            [0,  sp,  cp, 0],
            [0,  0,   0,  1],
        ], dtype=np.float32)

        y_pos = self.screen_pan_y + self.screen_height / 2.0 + self.screen_width * 0.016 + OSD_H / 2.0
        S = np.diag([OSD_W / 2.0, OSD_H / 2.0, 1.0, 1.0]).astype(np.float32)
        R = rot_y @ rot_x
        T = np.eye(4, dtype=np.float32)
        T[0, 3] = self.screen_pan_x; T[1, 3] = y_pos; T[2, 3] = -(self.screen_distance - 0.05)
        mvp = vp_mat @ T @ R @ S

        mgl_fbo.use()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._preset_osd_tex.use(location=2)
        self._overlay_prog['u_mvp'].write(mvp.T.tobytes())
        self._overlay_prog['u_alpha'].value = alpha
        self._preset_osd_vao.render(moderngl.TRIANGLE_STRIP)
        self._overlay_prog['u_alpha'].value = 1.0
        self.ctx.disable(moderngl.BLEND)

    def _render_screen_osd(self, eye_index, mgl_fbo, vp_mat):
        """Floating size+distance indicator shown while right grip + right stick adjusts."""
        if self._screen_osd_tex is None or self.screen_height is None:
            return

        now = self._frame_now

        # Rebuild texture on left eye whenever values change
        if eye_index == 0 and self.font is not None:
            cur_key = (round(self.screen_width, 2), round(self.screen_distance, 2))
            if cur_key != self._screen_osd_last_key:
                self._screen_osd_last_key = cur_key
                sw, sh = self._screen_osd_tex_size
                img  = Image.new('RGBA', (sw, sh), (0, 0, 0, 0))
                draw = ImageDraw.Draw(img)
                draw.rounded_rectangle(
                    [0, 0, sw - 1, sh - 1],
                    radius=12,
                    fill=(32, 32, 36, 210),
                )
                bfont   = self.bold_font or self.font
                C_LABEL = (150, 158, 185, 255)
                C_VALUE = (  0, 210, 230, 255)
                PAD = 12
                GAP = 8
                cy  = (sh - 32) // 2

                h = self.screen_height if self.screen_height is not None else 0.0

                def _tw(text, font):
                    try:
                        return int(draw.textlength(text, font=font))
                    except AttributeError:
                        return int(font.getsize(text)[0]) if hasattr(font, 'getsize') else 80

                # "Size" label + value
                size_lbl = "Size"
                size_val = f"{self.screen_width:.2f} × {h:.2f} m"
                draw.text((PAD, cy), size_lbl, font=bfont, fill=C_LABEL)
                x = PAD + _tw(size_lbl, bfont) + GAP
                draw.text((x, cy), size_val, font=self.font, fill=C_VALUE)
                x += _tw(size_val, self.font) + GAP * 3

                # "Dist" label (same grey as "Size") + value
                dist_lbl = "Dist"
                dist_val = f"{self.screen_distance:.2f} m"
                draw.text((x, cy), dist_lbl, font=bfont, fill=C_LABEL)
                x += _tw(dist_lbl, bfont) + GAP
                draw.text((x, cy), dist_val, font=self.font, fill=C_VALUE)

                data = np.flipud(np.array(img, dtype=np.uint8))
                self._screen_osd_tex.write(data.tobytes())

        # Fade: hold 1.5 s then decay over 0.8 s (same rhythm as depth OSD)
        HOLD  = 1.5
        DECAY = 0.8
        elapsed = now - self._screen_osd_show_t
        if elapsed < HOLD:
            alpha = 1.0
        elif elapsed < HOLD + DECAY:
            alpha = 1.0 - (elapsed - HOLD) / DECAY
        else:
            alpha = 0.0

        if alpha <= 0.0:
            return

        # Screen-relative positioning; text size scales with screen width
        OSD_H = self.screen_width * 0.03
        sw, sh = self._screen_osd_tex_size
        OSD_W = OSD_H * (sw / sh)

        cy_  = math.cos(self.screen_yaw);   sy_ = math.sin(self.screen_yaw)
        cp   = math.cos(self.screen_pitch); sp  = math.sin(self.screen_pitch)
        rot_y = np.array([
            [ cy_,  0, sy_, 0],
            [  0,   1,   0, 0],
            [-sy_,  0, cy_, 0],
            [  0,   0,   0, 1],
        ], dtype=np.float32)
        rot_x = np.array([
            [1,  0,   0, 0],
            [0,  cp, -sp, 0],
            [0,  sp,  cp, 0],
            [0,  0,   0,  1],
        ], dtype=np.float32)

        y_pos = self.screen_pan_y + self.screen_height / 2.0 + self.screen_width * 0.016 + OSD_H / 2.0
        S = np.diag([OSD_W / 2.0, OSD_H / 2.0, 1.0, 1.0]).astype(np.float32)
        R = rot_y @ rot_x
        T = np.eye(4, dtype=np.float32)
        T[0, 3] = self.screen_pan_x; T[1, 3] = y_pos; T[2, 3] = -self.screen_distance
        mvp = vp_mat @ T @ R @ S

        mgl_fbo.use()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._screen_osd_tex.use(location=2)
        self._overlay_prog['u_mvp'].write(mvp.T.tobytes())
        self._overlay_prog['u_alpha'].value = alpha
        self._screen_osd_vao.render(moderngl.TRIANGLE_STRIP)
        self._overlay_prog['u_alpha'].value = 1.0
        self.ctx.disable(moderngl.BLEND)

    def _render_brand_osd(self, eye_index, mgl_fbo, vp_mat):
        """Controller brand indicator attached to the right controller.

        Appears on the left side of the right controller when the user switches
        controller models via A+B combo.  Auto-fades after a short hold.
        """
        if self._brand_osd_tex is None or self._current_brand is None:
            return

        now = self._frame_now

        if eye_index == 0 and self.font is not None:
            env_name = (self._active_environment or self._environment_model or 'Default').strip() or 'Default'
            ctrl_name = self._current_brand
            cur_key = (env_name, ctrl_name)
            if cur_key != self._brand_osd_last_name:
                self._brand_osd_last_name = cur_key
                self._brand_osd_show_t = now
                self._brand_osd_alpha = 1.0

                bw, bh = self._brand_osd_tex_size
                img = Image.new('RGBA', (bw, bh), (0, 0, 0, 0))
                draw = ImageDraw.Draw(img)
                draw.rounded_rectangle(
                    [0, 0, bw - 1, bh - 1],
                    radius=12,
                    fill=(32, 32, 36, 210),
                )
                bfont = self.bold_font or self.font
                C_LABEL = (150, 158, 185, 255)
                C_VALUE = (  0, 210, 230, 255)
                PAD = 12
                row_y = [18, 58]
                rows = (("Environment:", env_name), ("Controller:", ctrl_name))
                label_w = 0
                for label, _value in rows:
                    try:
                        label_w = max(label_w, int(draw.textlength(label, font=bfont)))
                    except AttributeError:
                        label_w = max(label_w, int(bfont.getsize(label)[0]) if hasattr(bfont, 'getsize') else 110)
                for y, (label, value) in zip(row_y, rows):
                    draw.text((PAD, y), label, font=bfont, fill=C_LABEL)
                    draw.text((PAD + label_w + 10, y), str(value), font=self.font, fill=C_VALUE)
                data = np.flipud(np.array(img, dtype=np.uint8))
                self._brand_osd_tex.write(data.tobytes())

        # Fade: hold 1.5 s then decay over 0.8 s
        HOLD  = 1.5
        DECAY = 0.8
        elapsed = now - self._brand_osd_show_t
        if elapsed < HOLD:
            alpha = 1.0
        elif elapsed < HOLD + DECAY:
            alpha = 1.0 - (elapsed - HOLD) / DECAY
        else:
            alpha = 0.0
        self._brand_osd_alpha = alpha

        if alpha <= 0.0:
            return

        OSD_H = self.screen_width * 0.03
        bw, bh = self._brand_osd_tex_size
        OSD_W = OSD_H * (bw / bh)

        # Anchor to the left of the right controller grip
        if self._grip_mat_r is None:
            return

        grip_pos = self._grip_mat_r[:3, 3].astype('f8')
        # Midpoint between screen bottom-edge centre and the controller grip
        sh = self.screen_height
        if sh is None:
            fw, fh = self.frame_size
            sh = self.screen_width * (fh / fw if fw > 0 else 9.0 / 16.0)
        bottom_edge = np.array([self.screen_pan_x,
                                self.screen_pan_y - sh / 2.0,
                                -self.screen_distance], dtype='f8')
        panel_pos = (bottom_edge + grip_pos) * 0.5

        if self._head_pos_w is not None:
            toward = np.array(self._head_pos_w, dtype='f8') - panel_pos
            toward /= np.linalg.norm(toward) + 1e-10
            panel_fwd = toward
        else:
            panel_fwd = np.array([0.0, 0.0, -1.0], dtype='f8')
        panel_up = np.array([0.0, 1.0, 0.0], dtype='f8')

        S = np.diag([OSD_W / 2.0, OSD_H / 2.0, 1.0, 1.0]).astype(np.float32)
        panel_right = np.cross(panel_up, panel_fwd)
        panel_right /= np.linalg.norm(panel_right) + 1e-10
        panel_up2 = np.cross(panel_fwd, panel_right)
        panel_up2 /= np.linalg.norm(panel_up2) + 1e-10
        R = np.eye(4, dtype=np.float32)
        R[:3, 0] = panel_right.astype(np.float32)
        R[:3, 1] = panel_up2.astype(np.float32)
        R[:3, 2] = panel_fwd.astype(np.float32)
        T = np.eye(4, dtype=np.float32)
        T[0, 3] = panel_pos[0]; T[1, 3] = panel_pos[1]; T[2, 3] = panel_pos[2]
        mvp = vp_mat @ T @ R @ S

        mgl_fbo.use()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._brand_osd_tex.use(location=2)
        self._overlay_prog['u_mvp'].write(mvp.T.tobytes())
        self._overlay_prog['u_alpha'].value = alpha
        self._brand_osd_vao.render(moderngl.TRIANGLE_STRIP)
        self._overlay_prog['u_alpha'].value = 1.0
        self.ctx.disable(moderngl.BLEND)

    def _render_seat_adjust_osd(self, eye_index, mgl_fbo, vp_mat):
        """Floating OSD showing seat position while in adjust mode."""
        if self._seat_adjust_osd_tex is None or self.screen_height is None:
            return
        if not self._seat_adjust_active:
            now = getattr(self, '_frame_now', time.perf_counter())
            elapsed = now - self._seat_adjust_osd_show_t
            if elapsed > 2.3:
                return
        now = getattr(self, '_frame_now', time.perf_counter())
        if eye_index == 0 and self._seat_adjust_osd_dirty:
            self._seat_adjust_osd_dirty = False
            self._seat_adjust_osd_show_t = now
            if self.font is not None:
                x, y, z, angle = self._seat_adjust_current_pos()
                dw, dh = self._seat_adjust_osd_tex_size
                img = Image.new('RGBA', (dw, dh), (0, 0, 0, 0))
                draw = ImageDraw.Draw(img)
                if self._seat_adjust_active:
                    fill_color = (40, 60, 40, 220)
                else:
                    fill_color = (32, 32, 36, 210)
                draw.rounded_rectangle([0, 0, dw - 1, dh - 1], radius=12, fill=fill_color)
                bfont = self.bold_font or self.font
                C_LABEL = (150, 158, 185, 255)
                C_VALUE = (0, 210, 230, 255)
                C_MODE  = (100, 255, 100, 255)
                PAD = 12
                cy = (dh - 32) // 2
                if self._seat_adjust_active:
                    if getattr(self, '_seat_adjust_grip_move', False):
                        mp = self._env_model_pos
                        mode_text = "ROOM MOVE"
                        draw.text((PAD, cy), mode_text, font=bfont, fill=(255, 200, 80, 255))
                        try:
                            lw = int(draw.textlength(mode_text, font=bfont))
                        except AttributeError:
                            lw = int(bfont.getsize(mode_text)[0]) if hasattr(bfont, 'getsize') else 110
                        pos_text = f"  X:{mp[0]:+.2f}  Y:{mp[1]:+.2f}  Z:{mp[2]:+.2f}"
                    else:
                        mode_text = "SEAT ADJUST"
                        draw.text((PAD, cy), mode_text, font=bfont, fill=C_MODE)
                        try:
                            lw = int(draw.textlength(mode_text, font=bfont))
                        except AttributeError:
                            lw = int(bfont.getsize(mode_text)[0]) if hasattr(bfont, 'getsize') else 110
                        pos_text = f"  X:{x:+.2f}  Y:{y:.2f}  Z:{z:+.2f}  A:{angle:+.0f}°"
                else:
                    mode_text = "Saved"
                    draw.text((PAD, cy), mode_text, font=bfont, fill=C_LABEL)
                    try:
                        lw = int(draw.textlength(mode_text, font=bfont))
                    except AttributeError:
                        lw = int(bfont.getsize(mode_text)[0]) if hasattr(bfont, 'getsize') else 50
                    pos_text = f"  X:{x:+.2f}  Y:{y:.2f}  Z:{z:+.2f}  A:{angle:+.0f}°"
                draw.text((PAD + lw + 4, cy), pos_text, font=self.font, fill=C_VALUE)
                data = np.flipud(np.array(img, dtype=np.uint8))
                self._seat_adjust_osd_tex.write(data.tobytes())
        if self._seat_adjust_active:
            alpha = 1.0
        else:
            HOLD = 1.5
            DECAY = 0.8
            elapsed = now - self._seat_adjust_osd_show_t
            if elapsed < HOLD:
                alpha = 1.0
            elif elapsed < HOLD + DECAY:
                alpha = 1.0 - (elapsed - HOLD) / DECAY
            else:
                return
        self._seat_adjust_osd_alpha = alpha
        OSD_H = self.screen_width * 0.03
        dw, dh = self._seat_adjust_osd_tex_size
        OSD_W = OSD_H * (dw / dh)
        cy_ = math.cos(self.screen_yaw);  sy_ = math.sin(self.screen_yaw)
        cp  = math.cos(self.screen_pitch); sp  = math.sin(self.screen_pitch)
        rot_y = np.array([[ cy_, 0, sy_, 0], [0, 1, 0, 0], [-sy_, 0, cy_, 0], [0, 0, 0, 1]], dtype=np.float32)
        rot_x = np.array([[1, 0, 0, 0], [0, cp, -sp, 0], [0, sp, cp, 0], [0, 0, 0, 1]], dtype=np.float32)
        y_pos = self.screen_pan_y - self.screen_height / 2.0 - self.screen_width * 0.016 - OSD_H / 2.0
        S = np.diag([OSD_W / 2.0, OSD_H / 2.0, 1.0, 1.0]).astype(np.float32)
        R = rot_y @ rot_x
        T = np.eye(4, dtype=np.float32)
        T[0, 3] = self.screen_pan_x; T[1, 3] = y_pos; T[2, 3] = -(self.screen_distance - 0.05)
        mvp = vp_mat @ T @ R @ S
        mgl_fbo.use()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._seat_adjust_osd_tex.use(location=2)
        self._overlay_prog['u_mvp'].write(mvp.T.tobytes())
        self._overlay_prog['u_alpha'].value = alpha
        self._seat_adjust_osd_vao.render(moderngl.TRIANGLE_STRIP)
        self._overlay_prog['u_alpha'].value = 1.0
        self.ctx.disable(moderngl.BLEND)

    def _render_calibration_panel(self, mgl_fbo, vp_mat):
        """Draw semi-transparent calibration panel in VR (reuses FPS overlay shader)."""
        if self._overlay_prog is None:
            return
        now = self._frame_now
        # stale data check: only update texture every 0.5s to avoid excessive PIL overhead when using sticks to adjust values
        if not hasattr(self, '_calib_last_update'):
            self._calib_last_update = 0.0
        if now - self._calib_last_update < 0.5:
            if self._calib_tex is not None:
                self._render_overlay_quad(mgl_fbo, vp_mat, self._calib_tex,
                                        self._calib_tex_size, 0.6)
            return
        self._calib_last_update = now

        # Generate panel content with PIL: controller model + current temp offsets/rotation + instructions.
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            return
        off = self._calibration_temp_offset
        rot = self._calibration_temp_rot
        lines = [
            f"Calibration: {self._current_brand or self._controller_model}",
            f"Y offset: {off[1]:.3f} m",
            f"Z offset: {off[2]:.3f} m",
            f"Rotation: {rot:.1f} deg",
            "",
            "L-stick U/D: Y  |  R-stick U/D: Z",
            "R-stick L/R: Rotation",
            "B: Save  Menu+A+B: Quit",
        ]
        fw, fh = 420, 200
        img = Image.new('RGBA', (fw, fh), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle([0, 0, fw - 1, fh - 1], radius=12,
                            fill=(20, 20, 28, 200))
        # Cache the consola font once re-loading from disk every 0.5s
        # (the panel's rebuild cadence) is wasted I/O.
        if not hasattr(self, '_calib_font') or self._calib_font is None:
            try:
                self._calib_font = ImageFont.truetype("consola.ttf", 20)
            except Exception:
                self._calib_font = ImageFont.load_default()
        font = self._calib_font
        y = 16
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            draw.text((16, y), line, font=font, fill=(200, 220, 255, 255))
            y += bbox[3] - bbox[1] + 4

        data = np.flipud(np.array(img, dtype=np.uint8))
        if self._calib_tex is None:
            self._calib_tex = self.ctx.texture((fw, fh), 4, dtype='f1')
            self._calib_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._calib_tex.write(data.tobytes())
        self._calib_tex_size = (fw, fh)

        self._render_overlay_quad(mgl_fbo, vp_mat, self._calib_tex,
                                (fw, fh), 0.6)

    def _render_overlay_quad(self, mgl_fbo, vp_mat, tex, tex_size, alpha):
        """Generate MVP for a quad in front of the user's view, with size based on tex_size and distance based on typical arm's length, then render with the given texture and alpha."""
        tex_w, tex_h = tex_size
        OVERLAY_H = 0.10
        OVERLAY_W = OVERLAY_H * (tex_w / tex_h)

        # Head-Locked Overlay: position the quad in front of the user's head, slightly below eye level, and oriented to always face the user. This is more comfortable for reading text and ensures the overlay is always visible regardless of head orientation. The quad's forward direction is aligned with the headset's forward direction projected onto the horizontal plane, so it remains readable even when looking down or up.
        if self._head_pos_w is not None and self._head_fwd_w is not None:
            hx, hy, hz = self._head_pos_w
            fx, fy, fz = self._head_fwd_w
            wx = hx + fx * 1.2
            wy = hy + fy * 1.2 - 0.1
            wz = hz + fz * 1.2
            V3 = vp_mat[:3, :3]
            right = V3[0, :].copy(); right = right / (np.linalg.norm(right) + 1e-10)
            up    = V3[1, :].copy(); up    = up    / (np.linalg.norm(up)    + 1e-10)
            fwd   = -V3[2,:].copy(); fwd  = fwd  / (np.linalg.norm(fwd)   + 1e-10)
            S = np.diag([OVERLAY_W/2.0, OVERLAY_H/2.0, 1.0, 1.0]).astype(np.float32)
            R = np.eye(4, dtype=np.float32)
            R[:3, 0] = right; R[:3, 1] = up; R[:3, 2] = fwd
            T = np.eye(4, dtype=np.float32)
            T[0, 3] = wx; T[1, 3] = wy; T[2, 3] = wz
            mvp = vp_mat @ T @ R @ S
        else:
            return
        mgl_fbo.use()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        tex.use(location=2)
        self._overlay_prog['u_mvp'].write(mvp.T.tobytes())
        self._overlay_prog['u_alpha'].value = alpha
        self._overlay_vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.disable(moderngl.BLEND)

    # Shortcuts help panel (right controller attachment)
    def _build_help_texture(self):
        """Generate help panel texture (3-column layout, adaptive size, Chinese font priority)."""
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            return

        # font loading with fallback: first look for "font.ttf" in project root, then common system fonts (Windows/macOS/Linux), finally default PIL font.
        _font_paths = [
            # internal project font (supports Chinese, bundled with the app)
            os.path.join(_PROJECT_ROOT, "font.ttf"),
            # Windows system fonts
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\simhei.ttf",
            # macOS system fonts
            "/System/Library/Fonts/PingFang.ttc",
            # Linux common Chinese fonts
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        ]
        _font_size = 21
        _title_size = 21
        font = None
        for fp in _font_paths:
            if os.path.isfile(fp):
                try:
                    font  = ImageFont.truetype(fp, _font_size)
                    bfont = ImageFont.truetype(fp, _title_size)
                    break
                except Exception:
                    continue
        if font is None:
            try:
                font  = ImageFont.truetype("consola.ttf", _font_size)
                bfont = ImageFont.truetype("consolab.ttf", _title_size)
            except Exception:
                font  = ImageFont.load_default()
                bfont = font

        draw = ImageDraw.Draw(Image.new('RGBA', (1, 1)))
        # Measure maximum width for each column
        col_w = [0, 0, 0]
        for r in ROWS:
            for ci in range(3):
                b = draw.textbbox((0, 0), r[ci], font=bfont if r[3] else font)
                w = b[2] - b[0]
                if w > col_w[ci]:
                    col_w[ci] = w

        GAP = 20    # Column spacing
        PAD_X = 30  # Left and right padding
        PAD_Y = 20  # Top and bottom padding
        LINE_H = (_font_size + 6) if font else 20
        tw = col_w[0] + GAP + col_w[1] + GAP + col_w[2] + PAD_X * 2
        th = len(ROWS) * LINE_H + PAD_Y * 2

        # Rebuild texture with correct dimensions
        if hasattr(self, '_help_tex') and self._help_tex is not None:
            try:
                self._help_tex.release()
            except Exception:
                pass
        self._help_tex = self.ctx.texture((tw, th), 4, dtype='f1')
        self._help_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._help_tex_size = (tw, th)

        img = Image.new('RGBA', (tw, th), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle([0, 0, tw - 1, th - 1], radius=14,
                            fill=(18, 18, 28, 210))

        col_x = [PAD_X, PAD_X + col_w[0] + GAP,
                PAD_X + col_w[0] + GAP + col_w[1] + GAP]
        for ri, (c1, c2, c3, is_title) in enumerate(ROWS):
            y = PAD_Y + ri * LINE_H
            f = bfont if is_title else font
            color = (90, 190, 255, 255) if is_title else (200, 210, 235, 255)
            for ci, txt in enumerate([c1, c2, c3]):
                if txt:
                    draw.text((col_x[ci], y), txt, font=f, fill=color)

        data = np.flipud(np.array(img, dtype=np.uint8))
        self._help_tex.write(data.tobytes())

    def _render_help_panel(self, mgl_fbo, vp_mat):
        """Render the help/shortcut panel anchored to the left side of the screen.

        Hinged at its right edge and rotated to face the user. The right edge
        stays fixed to the screen's left edge; screen yaw/pitch/pan translate
        the entire assembly in world space.
        """
        if self._help_tex is None or self.screen_height is None:
            return

        tex_w, tex_h = self._help_tex_size
        sh = self.screen_height
        sx = self.screen_width / 2.0
        GAP = sh * 0.02

        PANEL_H = sh
        PANEL_W = PANEL_H * (tex_w / tex_h)

        # Screen rotation
        cy_s = math.cos(self.screen_yaw);   sy_s = math.sin(self.screen_yaw)
        cp_s = math.cos(self.screen_pitch); sp_s = math.sin(self.screen_pitch)
        R_yaw = np.array([[ cy_s,  0, sy_s, 0],
                          [    0,  1,     0, 0],
                          [-sy_s,  0, cy_s,  0],
                          [    0,  0,     0, 1]], dtype=np.float32)
        R_pitch = np.array([[1,    0,     0, 0],
                            [0, cp_s, -sp_s, 0],
                            [0, sp_s,  cp_s, 0],
                            [0,    0,     0, 1]], dtype=np.float32)
        R = R_yaw @ R_pitch
        T = np.eye(4, dtype=np.float32)
        T[0, 3] = self.screen_pan_x
        T[1, 3] = self.screen_pan_y
        T[2, 3] = -self.screen_distance

        # Head position in screen-local space to compute hinge angle
        head_w = np.array(self._head_pos_w, dtype=np.float32) if self._head_pos_w is not None else np.array([0.0, 0.0, 0.0], dtype=np.float32)
        screen_c_w = np.array([self.screen_pan_x, self.screen_pan_y, -self.screen_distance], dtype=np.float32)
        R3 = R[:3, :3].astype(np.float32)
        head_local = R3.T @ (head_w - screen_c_w)
        # Panel right edge fixed to screen's left edge
        right_edge_local = np.array([-sx - GAP, 0.0, 0.0], dtype=np.float32)
        to_user = head_local - right_edge_local
        to_user /= np.linalg.norm(to_user) + 1e-10

        # Hinge angle: panel normal [sinθ,0,cosθ] points toward user
        theta = math.atan2(float(to_user[0]), float(to_user[2]))
        ct = math.cos(theta); st = math.sin(theta)

        # Position right edge at (-sx - GAP, 0, 0) in screen-local (screen's left edge)
        T_right_edge = np.eye(4, dtype=np.float32)
        T_right_edge[0, 3] = -sx - GAP

        # Shift quad so right edge is at origin before hinge rotation (panel extends left)
        T_offset = np.eye(4, dtype=np.float32)
        T_offset[0, 3] = -PANEL_W / 2.0

        # Hinge rotation around Y (vertical axis through right edge)
        Ry = np.eye(4, dtype=np.float32)
        Ry[0, 0] = ct; Ry[0, 2] = st
        Ry[2, 0] = -st; Ry[2, 2] = ct

        S_panel = np.diag([PANEL_W / 2.0, PANEL_H / 2.0, 1.0, 1.0]).astype(np.float32)

        model = T @ R @ T_right_edge @ Ry @ T_offset @ S_panel

        mvp = vp_mat @ model
        mgl_fbo.use()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._help_tex.use(location=2)
        self._overlay_prog['u_mvp'].write(mvp.T.tobytes())
        self._overlay_prog['u_alpha'].value = 0.75
        self._help_vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.disable(moderngl.BLEND)

    def _build_keyboard_texture(self):
        """(Re)build the virtual keyboard texture with the current shift state.

        When Shift or Caps Lock is active the number/symbol keys show their
        shifted glyph (e.g. '!' instead of '1').  Modifier key backgrounds
        are highlighted as before.
        """
        TW, TH   = _KB_TEX_W, _KB_TEX_H
        ROW_H    = TH / len(_KB_ROWS)
        UNIT_W   = TW / float(_KB_UNITS_WIDE)
        UNIT_M   = self._keyboard_width / float(_KB_UNITS_WIDE)
        PAD      = 3
        show_s   = self._kb_show_shifted   # whether to render shifted labels

        img  = Image.new('RGBA', (TW, TH), (30, 30, 35, 230))
        draw = ImageDraw.Draw(img)

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
        row_h_m  = self._keyboard_height / len(_KB_ROWS)

        for row_i, row in enumerate(_KB_ROWS):
            py0 = int(row_i * ROW_H)
            py1 = int((row_i + 1) * ROW_H)
            ly1 = kh_half - row_i * row_h_m
            ly0 = ly1 - row_h_m
            px  = 0.0
            lx  = -kw_half
            for (label, vk_normal, shifted_label, vk_shifted, width_units) in row:
                px_end  = px + width_units * UNIT_W
                lx_end  = lx + width_units * UNIT_M

                if vk_normal == -1:
                    px = px_end
                    lx = lx_end
                    continue

                # Key background
                draw.rectangle([px + PAD, py0 + PAD, px_end - PAD, py1 - PAD],
                            fill=(60, 62, 70, 255), outline=(130, 132, 140, 255))

                # Pick label: shifted version if available and shift is active
                display_label = label
                if show_s and shifted_label is not None:
                    display_label = shifted_label

                if fnt:
                    tx = (px + px_end) / 2
                    ty = (py0 + py1) / 2
                    draw.text((tx, ty), display_label, font=fnt,
                            fill=(220, 220, 225, 255), anchor='mm')
                else:
                    draw.text((int(px + PAD + 2), int(py0 + PAD + 2)),
                            display_label, fill=(220, 220, 225, 255))

                uv_rect   = (px / TW, py0 / TH, px_end / TW, py1 / TH)
                loc_rect  = (lx, ly0, lx_end, ly1)

                self._keyboard_keys.append(_KeyEntry(
                    label=label,
                    shifted_label=shifted_label,
                    vk=vk_normal,
                    shifted_vk=vk_shifted if vk_shifted is not None else vk_normal,
                    rect_uv=uv_rect,
                    rect_local=loc_rect,
                ))

                px  = px_end
                lx  = lx_end

        # Upload
        tex_data = np.flipud(np.array(img, dtype=np.uint8))
        if self._keyboard_tex is not None:
            self._keyboard_tex.release()
        self._keyboard_tex = self.ctx.texture((TW, TH), 4, dtype='f1')
        self._keyboard_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._keyboard_tex.write(tex_data.tobytes())

        # VAO (first build only geometry never changes)
        if self._keyboard_vao is None:
            verts = np.array([-1,-1,0,0, 1,-1,1,0, -1,1,0,1, 1,1,1,1], dtype='f4')
            self._keyboard_vao = self.ctx.vertex_array(
                self._overlay_prog,
                [(self.ctx.buffer(verts.tobytes()), '2f 2f', 'in_position', 'in_uv')],
            )

    def _init_keyboard(self):
        """Initial keyboard build (called once when the user toggles it on)."""
        self._kb_show_shifted = False
        self._build_keyboard_texture()

    def _kb_world_mat(self):
        """Build the keyboard's world transform: rot_y(yaw) ∘ rot_x(pitch) then translate.

        The keyboard's local frame has Z = surface normal. Negative pitch tilts the
        face up so a user looking down at it sees the face dead-on (friendly angle,
        like a desk keyboard).
        """
        cp = math.cos(self._keyboard_pitch); sp = math.sin(self._keyboard_pitch)
        cy = math.cos(self._keyboard_yaw);   sy = math.sin(self._keyboard_yaw)
        rot_y = np.array([[ cy, 0, sy, 0],
                        [  0, 1,  0, 0],
                        [-sy, 0, cy, 0],
                        [  0, 0,  0, 1]], dtype=np.float32)
        rot_x = np.array([[1,  0,   0, 0],
                        [0, cp, -sp, 0],
                        [0, sp,  cp, 0],
                        [0,  0,   0, 1]], dtype=np.float32)
        trans = np.eye(4, dtype=np.float32)
        # Translate to (pan_x, pan_y, -distance) matches the world-anchor convention
        # used by the main screen.
        trans[0, 3] = self._keyboard_pan_x
        trans[1, 3] = self._keyboard_pan_y
        trans[2, 3] = -self._keyboard_distance
        return trans @ rot_y @ rot_x

    def _anchor_keyboard_below_screen(self):
        """Snap the keyboard below the screen's bottom edge, facing the same direction.

        Keyboard width is 0.75× screen width. Sits below the FPS overlay panel.
        Returns a dict of the computed position for caching.
        """
        if self.screen_height is None:
            fw, fh = self.frame_size
            sh = self.screen_width * (fh / fw if fw > 0 else 9.0 / 16.0)
        else:
            sh = self.screen_height
        self._keyboard_width    = self.screen_width * 0.75
        FPS_GAP = sh * 0.02    # matches _render_fps_overlay gap
        FPS_H   = sh / 8.0     # matches _render_fps_overlay height
        KB_GAP  = FPS_GAP      # same proportional gap below FPS overlay
        self._keyboard_pan_x    = self.screen_pan_x
        self._keyboard_pan_y    = (self.screen_pan_y - sh / 2.0
                                - FPS_GAP - FPS_H - KB_GAP
                                - self._keyboard_height / 2.0)
        self._keyboard_distance = self.screen_distance
        # Auto-orient keyboard toward the head (sphere center), same logic as screen.
        head = getattr(self, '_head_pos_w', None)
        if head is not None:
            hx, hy, hz = float(head[0]), float(head[1]), float(head[2])
            dx = self._keyboard_pan_x - hx
            dy = self._keyboard_pan_y - hy
            dz = -self._keyboard_distance - hz
            r = math.sqrt(dx*dx + dy*dy + dz*dz)
            if r > 1e-4:
                nx, ny, nz = dx / r, dy / r, dz / r
                base_yaw   = math.atan2(-nx, -nz)
                base_pitch = math.asin(max(-1.0, min(1.0, ny)))
                self._keyboard_yaw   = base_yaw   + self._kb_yaw_offset
                self._keyboard_pitch = base_pitch + self._kb_pitch_offset
            else:
                self._keyboard_yaw   = self.screen_yaw + self._kb_yaw_offset
                self._keyboard_pitch = self._kb_pitch_offset
        else:
            self._keyboard_yaw      = self.screen_yaw + self._kb_yaw_offset
            self._keyboard_pitch    = self._kb_pitch_offset
        return {
            'pan_x': self._keyboard_pan_x, 'pan_y': self._keyboard_pan_y,
            'distance': self._keyboard_distance, 'width': self._keyboard_width,
            'yaw': self._keyboard_yaw, 'pitch': self._keyboard_pitch,
        }

    def _render_keyboard(self, mgl_fbo, vp_mat):
        """Render the virtual keyboard quad and highlight hovered keys."""
        if self._keyboard_tex is None or self._keyboard_vao is None:
            return

        kw2 = self._keyboard_width  / 2.0
        kh2 = self._keyboard_height / 2.0
        kb_world = self._kb_world_mat()
        vp_kb = vp_mat @ kb_world   # shared for all key highlights

        # Keyboard quad: vertices are in [-1, +1] in X and Y, so scale by half-extents.
        scale_kb = np.array([[kw2, 0,   0, 0],
                            [0,   kh2, 0, 0],
                            [0,   0,   1, 0],
                            [0,   0,   0, 1]], dtype=np.float32)
        mvp = vp_kb @ scale_kb

        mgl_fbo.use()
        self.ctx.depth_mask = False
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        # Keyboard border: solid quad slightly larger than the keyboard
        if self._kb_border_alpha > 0.0 and self._border_prog is not None:
            BORDER = 0.008
            bx = kw2 + BORDER
            by = kh2 + BORDER
            border_scale = np.array([[bx, 0, 0, 0],
                                    [0, by, 0, 0],
                                    [0, 0,  1, -0.001],
                                    [0, 0,  0, 1]], dtype=np.float32)
            border_mvp = vp_kb @ border_scale
            self._border_prog['u_mvp'].write(border_mvp.T.tobytes())
            self._border_prog['u_color'].value = (0.3, 0.7, 1.0, self._kb_border_alpha)
            self._border_vao.render(moderngl.TRIANGLE_STRIP)

        self._keyboard_tex.use(location=2)
        self._overlay_prog['u_mvp'].write(mvp.T.tobytes())
        self._keyboard_vao.render(moderngl.TRIANGLE_STRIP)

        def _hl_quad(rect_local, color):
            # rect_local is already in metres, expressed in the keyboard's local frame
            # (X right, Y up, Z = surface). Place a unit quad scaled to the key rect at
            # +1 mm in front of the surface to avoid z-fighting.
            x0, y0, x1, y1 = rect_local
            cx = (x0 + x1) / 2.0; cy_ = (y0 + y1) / 2.0
            hw = (x1 - x0) / 2.0; hh  = (y1 - y0) / 2.0
            hl_local = np.array([[hw, 0,  0, cx ],
                                [0,  hh, 0, cy_],
                                [0,  0,  1, 0.001],
                                [0,  0,  0, 1  ]], dtype=np.float32)
            hl_mvp = vp_kb @ hl_local
            self._border_prog['u_mvp'].write(hl_mvp.T.tobytes())
            self._border_prog['u_color'].value = color
            self._border_vao.render(moderngl.TRIANGLE_STRIP)

        # Highlight every key whose VK matches an armed modifier.
        # Locked modifiers get a brighter amber than one-shot to make state legible.
        VK_SHIFT = 0x10; VK_CAPS = 0x14; VK_CTRL = 0x11; VK_ALT = 0x12; VK_WIN = 0x5B
        oneshot_vks = set(); locked_vks = set()
        for name, vk in (('shift', VK_SHIFT), ('ctrl', VK_CTRL),
                        ('alt', VK_ALT), ('win', VK_WIN)):
            active, locked, _ = self._mod_state[name]
            if locked:   locked_vks.add(vk)
            elif active: oneshot_vks.add(vk)
        if self._caps_lock: locked_vks.add(VK_CAPS)
        for key in self._keyboard_keys:
            if key.vk in locked_vks:
                _hl_quad(key.rect_local, (1.0, 0.55, 0.05, 0.65))
            elif key.vk in oneshot_vks:
                _hl_quad(key.rect_local, (1.0, 0.7, 0.15, 0.45))

        # Highlight keys held by triggers (pressed) bright, strong
        for held_idx in set(x for x in [self._kb_held_key_l, self._kb_held_key_r] if x is not None):
            _hl_quad(self._keyboard_keys[held_idx].rect_local, (0.5, 0.85, 1.0, 0.70))

        # Darker highlight on keys hovered by either laser (suppressed while gripping)
        if not (self._grip_l_now or self._grip_r_now):
            for hover_idx in set(x for x in [self._kb_hover_l, self._kb_hover_r] if x is not None):
                if hover_idx not in set(x for x in [self._kb_held_key_l, self._kb_held_key_r] if x is not None):
                    _hl_quad(self._keyboard_keys[hover_idx].rect_local, (0.15, 0.50, 0.80, 0.25))

        self.ctx.disable(moderngl.BLEND)
        self.ctx.depth_mask = True

