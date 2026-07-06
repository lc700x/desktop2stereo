"""CropMixin: movie letterbox/pillarbox crop detection and UV helpers."""

import math
import time

import numpy as np


class CropMixin:
    def _movie_crop_profile_enabled(self):
        if getattr(self, '_crop_mode', 'auto') != 'auto':
            return False
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
        self._refit_screen_geometry_for_crop()

    def _refit_screen_geometry_for_crop(self):
        """Rebuild screen size/geometry caches to match the current crop.

        Split out of ``_set_movie_crop_render_uv`` so manual crop-adjust can
        defer this until the drag ends instead of doing it every frame.
        """
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

    def _set_manual_crop_uv(self, w, h):
        w = max(0.0, min(1.0, float(w)))
        h = max(0.0, min(1.0, float(h)))
        x = (1.0 - w) / 2.0
        y = (1.0 - h) / 2.0
        crop = (x, y, w, h)
        self._manual_crop_uv = crop
        self._set_movie_crop_render_uv(crop)
        return crop

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
        crop_mode = getattr(self, '_crop_mode', 'auto')
        if crop_mode == 'manual':
            crop = getattr(self, '_manual_crop_uv', (0.0, 0.0, 1.0, 1.0))
            self._set_movie_crop_render_uv(crop)
            return getattr(self, '_movie_crop_frame_uv', crop)
        if crop_mode == 'off':
            crop = (0.0, 0.0, 1.0, 1.0)
            self._set_movie_crop_render_uv(crop)
            return getattr(self, '_movie_crop_frame_uv', crop)

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

    def _crop_screen_dims(self, ref_size):
        """Screen (width, height) that keeps a cropped axis the SAME size.

        Sizes the quad from the FULL frame aspect pinned to ``ref_size``, then
        scales each axis by its own crop fraction. Cropping the sides shrinks
        width only (height unchanged), so the screen never appears to move
        closer/farther when adjusting crop. Content still fills without stretch
        because quad aspect == cropped-source aspect.
        """
        fw, fh = self.frame_size
        crop = self._current_movie_crop_uv()
        full_ar = (float(fh) / float(fw)) if fw > 0 else 9.0 / 16.0
        if fh > fw:
            base_h = float(ref_size)
            base_w = base_h / full_ar
        else:
            base_w = float(ref_size)
            base_h = base_w * full_ar
        return base_w * float(crop[2]), base_h * float(crop[3])

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
        top_i_f, bottom_count_f, center_mean, center_bright = stats[:4]
        left_i_f = float(stats[4]) if len(stats) > 4 else 0.0
        right_count_f = float(stats[5]) if len(stats) > 5 else 0.0
        n_rows = int(y_rows.shape[0])
        top_i = int(round(float(top_i_f)))
        bottom_count = int(round(float(bottom_count_f)))
        left_i = int(round(left_i_f))
        right_count = int(round(right_count_f))

        has_tb = top_i > 0 and bottom_count > 0 and top_i + bottom_count < n_rows
        x_cols = getattr(self, '_crop_x_cols', None)
        n_cols = int(x_cols.shape[0]) if x_cols is not None else 0
        has_lr = left_i > 0 and right_count > 0 and left_i + right_count < n_cols

        if not has_tb and not has_lr:
            return (0.0, 0.0, 1.0, 1.0)

        crop_u0, crop_v0, crop_uw, crop_vh = 0.0, 0.0, 1.0, 1.0

        if has_tb:
            bottom_anchor_i = n_rows - bottom_count - 1
            if bottom_anchor_i >= top_i:
                top = int(y_rows[min(top_i, n_rows - 1)])
                bottom = int(h) - min(int(h), int(y_rows[bottom_anchor_i]) + 1)
                min_bar = max(8, int(h * 0.035))
                if top >= min_bar and bottom >= min_bar:
                    bigger = max(top, bottom)
                    smaller = min(top, bottom)
                    if bigger - smaller <= max(18, int(bigger * 0.25)):
                        edge_trim = max(2, min(8, int(round(h * 0.004))))
                        crop_top = max(0, min(top + edge_trim, h - 2))
                        crop_bottom = max(crop_top + 1, h - bottom - edge_trim)
                        crop_h = crop_bottom - crop_top
                        removed = h - crop_h
                        if removed >= max(16, int(h * 0.07)):
                            if float(center_mean) >= 14.0 or float(center_bright) >= 0.035:
                                crop_v0 = crop_top / float(h)
                                crop_vh = crop_h / float(h)

        if has_lr and x_cols is not None:
            right_anchor_i = n_cols - right_count - 1
            if right_anchor_i >= left_i:
                left = int(x_cols[min(left_i, n_cols - 1)])
                right = int(w) - min(int(w), int(x_cols[right_anchor_i]) + 1)
                min_bar = max(8, int(w * 0.035))
                if left >= min_bar and right >= min_bar:
                    bigger = max(left, right)
                    smaller = min(left, right)
                    if bigger - smaller <= max(18, int(bigger * 0.25)):
                        edge_trim = max(2, min(8, int(round(w * 0.004))))
                        crop_left = max(0, min(left + edge_trim, w - 2))
                        crop_right = max(crop_left + 1, w - right - edge_trim)
                        crop_w = crop_right - crop_left
                        removed = w - crop_w
                        if removed >= max(16, int(w * 0.07)):
                            crop_u0 = crop_left / float(w)
                            crop_uw = crop_w / float(w)

        if crop_u0 == 0.0 and crop_uw == 1.0 and crop_v0 == 0.0 and crop_vh == 1.0:
            return (0.0, 0.0, 1.0, 1.0)
        return (crop_u0, crop_v0, crop_uw, crop_vh)

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

        # Column sampling for left/right pillarbox detection
        y0_col = int(h * 0.10)
        y1_col = max(y0_col + 1, int(h * 0.90))
        col_stride = max(1, (int(w) + 359) // 360)
        x_cols = np.arange(0, int(w), col_stride, dtype=np.int64)
        if x_cols.size == 0 or int(x_cols[-1]) != int(w) - 1:
            x_cols = np.append(x_cols, int(w) - 1)
        step_y = max(1, (y1_col - y0_col) // 128)
        self._crop_x_cols = x_cols

        plan = {
            'key': key,
            'x0': x0,
            'x1': x1,
            'step_x': step_x,
            'y_rows': y_rows,
            'center_mask_np': center_mask_np,
            'center_has_rows': bool(np.any(center_mask_np)),
            'y0_col': y0_col,
            'y1_col': y1_col,
            'step_y': step_y,
            'x_cols': x_cols,
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

    @staticmethod
    def _uniform_rows(luma):
        """Detect uniform-color rows: low std dev (solid bar of any color)."""
        row_std = luma.std(axis=1) if hasattr(luma, 'std') else np.std(luma, axis=1)
        row_mean = luma.mean(axis=1) if hasattr(luma, 'mean') else np.mean(luma, axis=1)
        return row_std < 6.0, row_mean

    @staticmethod
    def _uniform_cols(luma):
        """Detect uniform-color columns."""
        col_std = luma.std(axis=0) if hasattr(luma, 'std') else np.std(luma, axis=0)
        return col_std < 6.0

    def _detect_movie_letterbox_crop(self, rgb, is_tensor, w, h, async_gpu=False):
        if w < 64 or h < 64:
            return (0.0, 0.0, 1.0, 1.0)

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
                # Row scan (top/bottom)
                sample = rgb.index_select(1, y_idx)[:, :, x0:x1:step_x].detach().float()
                luma = sample[0] * 0.2126 + sample[1] * 0.7152 + sample[2] * 0.0722
                row_mean = luma.mean(dim=1)
                row_std = luma.std(dim=1)
                uniform_row = (row_std < 6.0).to(torch.int32)
                top_i_t = torch.cumprod(uniform_row, dim=0).sum().float()
                bottom_count_t = torch.cumprod(torch.flip(uniform_row, dims=(0,)), dim=0).sum().float()
                center_mask = plan['center_mask']
                center_count = plan['center_count']
                bright_frac = (luma > 20.0).float().mean(dim=1)
                center_mean_t = (row_mean * center_mask).sum() / center_count
                center_bright_t = (bright_frac * center_mask).sum() / center_count
                # Column scan (left/right)
                y0_col = plan['y0_col']
                y1_col = plan['y1_col']
                step_y = plan['step_y']
                x_cols = plan['x_cols']
                x_idx = plan.get('x_idx')
                if x_idx is None:
                    x_idx = torch.as_tensor(x_cols, device=device, dtype=torch.long)
                    plan['x_idx'] = x_idx
                col_sample = rgb[:, y0_col:y1_col:step_y, :].index_select(2, x_idx).detach().float()
                col_luma = col_sample[0] * 0.2126 + col_sample[1] * 0.7152 + col_sample[2] * 0.0722
                col_std = col_luma.std(dim=0)
                uniform_col = (col_std < 6.0).to(torch.int32)
                left_i_t = torch.cumprod(uniform_col, dim=0).sum().float()
                right_count_t = torch.cumprod(torch.flip(uniform_col, dims=(0,)), dim=0).sum().float()
                stats_t = torch.stack((top_i_t, bottom_count_t, center_mean_t, center_bright_t,
                                       left_i_t, right_count_t))
                if async_gpu and getattr(device, 'type', '') == 'cuda':
                    host = plan.get('host')
                    if host is None or host.shape[0] < 6:
                        host = torch.empty((6,), dtype=torch.float32, pin_memory=True)
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
        # Row scan
        sample = arr[y_rows, x0:x1:step_x, :].astype(np.float32, copy=False)
        luma = sample[:, :, 0] * 0.2126 + sample[:, :, 1] * 0.7152 + sample[:, :, 2] * 0.0722
        uniform_rows, row_mean_np = self._uniform_rows(luma)
        bright_frac_np = (luma > 20.0).mean(axis=1)
        top_i = 0
        n_rows = int(uniform_rows.shape[0])
        while top_i < n_rows and bool(uniform_rows[top_i]):
            top_i += 1
        bottom_count = 0
        while bottom_count < n_rows - top_i and bool(uniform_rows[n_rows - 1 - bottom_count]):
            bottom_count += 1
        center_mask = plan['center_mask_np']
        if plan.get('center_has_rows', False):
            center_mean = float(np.mean(row_mean_np[center_mask]))
            center_bright = float(np.mean(bright_frac_np[center_mask]))
        else:
            center_mean = 0.0
            center_bright = 0.0
        # Column scan
        x_cols = plan['x_cols']
        y0_col = plan['y0_col']
        y1_col = plan['y1_col']
        step_y = plan['step_y']
        col_sample = arr[y0_col:y1_col:step_y, :, :].astype(np.float32, copy=False)
        col_sample = col_sample[:, x_cols, :]
        col_luma = col_sample[:, :, 0] * 0.2126 + col_sample[:, :, 1] * 0.7152 + col_sample[:, :, 2] * 0.0722
        uniform_cols = self._uniform_cols(col_luma)
        left_i = 0
        n_cols = int(uniform_cols.shape[0])
        while left_i < n_cols and bool(uniform_cols[left_i]):
            left_i += 1
        right_count = 0
        while right_count < n_cols - left_i and bool(uniform_cols[n_cols - 1 - right_count]):
            right_count += 1
        return self._movie_crop_from_stats(
            (float(top_i), float(bottom_count), center_mean, center_bright,
             float(left_i), float(right_count)),
            y_rows,
            w,
            h,
        )

    def _maybe_update_movie_crop(self, rgb, is_tensor, w, h):
        if getattr(self, '_crop_mode', 'auto') != 'auto':
            # Manual/off modes own the render UV; auto-detection must not touch
            # it. Resetting here fought the per-frame manual crop set in
            # _current_movie_crop_uv, toggling the crop full<->manual every
            # frame (flicker + apparent screen resize/distance change).
            self._movie_crop_pending_gpu = None
            return
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
