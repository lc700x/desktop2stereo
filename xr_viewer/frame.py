"""FrameMixin: texture init, PBO management, glow color sampling."""

import math
import os
import time
import ctypes
import json
import collections

import moderngl
import numpy as np
from OpenGL.GL import (
    glBindTexture, GL_TEXTURE_2D,
    glTexParameterf, GL_TEXTURE_LOD_BIAS,
    glGenBuffers, glDeleteBuffers, glBindBuffer, glBufferData,
    GL_PIXEL_UNPACK_BUFFER, GL_DYNAMIC_DRAW, GL_STREAM_DRAW,
)

from .render import (
    _view_mat_inv, _euler_to_mat4, _pose_to_view_mat4, _fov_to_proj_mat4_cached,
    _mat3_to_quat_xyzw, _mat4_to_xr_posef, _xr_pose_to_model_mat4, _xr_quat_to_mat4,
)

from viewer import BACKEND


class FrameMixin:
    def _init_textures(self, w, h):
        if self.color_tex:
            self.color_tex.release()
        if self.depth_tex:
            self.depth_tex.release()
        self.color_tex = self.ctx.texture((w, h), 3, dtype='f1')
        self.color_tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self._color_tex_mipmap_filter_active = True
        self._glow_mipmap_pending = False
        self._glow_mipmap_pending_frame = -999999
        self._glow_mipmap_ready = False
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
        old_resources = list(getattr(self, '_cuda_res_color_ring', []) or [])
        old_resources += list(getattr(self, '_cuda_res_depth_ring', []) or [])
        for res in (getattr(self, '_cuda_res_color', None), getattr(self, '_cuda_res_depth', None)):
            if res is not None and res not in old_resources:
                old_resources.append(res)
        for res in old_resources:
            try:
                self._cuda_gl.unregister_resource(res)
            except Exception:
                pass

        old_ids = list(getattr(self, '_pbo_color_ring', []) or [])
        old_ids += list(getattr(self, '_pbo_depth_ring', []) or [])
        for pbo_id in (getattr(self, '_pbo_color', None), getattr(self, '_pbo_depth', None)):
            if pbo_id is not None and pbo_id not in old_ids:
                old_ids.append(pbo_id)
        if old_ids:
            try:
                glDeleteBuffers(len(old_ids), old_ids)
            except Exception:
                pass

        ring_count = max(int(getattr(self, '_gpu_pbo_ring_size', 2) or 2), 1)
        ids = glGenBuffers(ring_count * 2)
        try:
            ids = [int(x) for x in ids]
        except TypeError:
            ids = [int(ids)]
        color_ids = ids[:ring_count]
        depth_ids = ids[ring_count:ring_count * 2]

        for pbo_id in color_ids:
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_id)
            glBufferData(GL_PIXEL_UNPACK_BUFFER, w * h * 3, None, GL_DYNAMIC_DRAW)
        for pbo_id in depth_ids:
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_id)
            glBufferData(GL_PIXEL_UNPACK_BUFFER, w * h * 4, None, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

        self._pbo_color_ring = color_ids
        self._pbo_depth_ring = depth_ids
        self._cuda_res_color_ring = [self._cuda_gl.register_buffer(pbo_id) for pbo_id in color_ids]
        self._cuda_res_depth_ring = [self._cuda_gl.register_buffer(pbo_id) for pbo_id in depth_ids]
        self._gpu_pbo_index = 0
        self._pbo_color = color_ids[0] if color_ids else None
        self._pbo_depth = depth_ids[0] if depth_ids else None
        self._cuda_res_color = self._cuda_res_color_ring[0] if self._cuda_res_color_ring else None
        self._cuda_res_depth = self._cuda_res_depth_ring[0] if self._cuda_res_depth_ring else None
        self._pbo_texture_size = (w, h)
        print(f"[OpenXRViewer] GPU interop PBO ring created ({BACKEND}) {w}x{h} x{ring_count}")

    def _init_cpu_pbos(self, w, h):
        """Create unpack PBOs for CPU-path texture upload (async DMA)."""
        try:
            old_ids = list(getattr(self, '_cpu_pbo_color_ring', []) or [])
            old_ids += list(getattr(self, '_cpu_pbo_depth_ring', []) or [])
            for pbo_id in (getattr(self, '_cpu_pbo_color', None), getattr(self, '_cpu_pbo_depth', None)):
                if pbo_id is not None and pbo_id not in old_ids:
                    old_ids.append(pbo_id)
            if old_ids:
                try:
                    glDeleteBuffers(len(old_ids), old_ids)
                except Exception:
                    pass

            ring_count = max(int(getattr(self, '_cpu_pbo_ring_size', 3) or 3), 1)
            ids = glGenBuffers(ring_count * 2)
            try:
                ids = [int(x) for x in ids]
            except TypeError:
                ids = [int(ids)]
            color_ids = ids[:ring_count]
            depth_ids = ids[ring_count:ring_count * 2]
            color_bytes = w * h * 3
            depth_bytes = w * h * 4
            for pbo_id in color_ids:
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_id)
                glBufferData(GL_PIXEL_UNPACK_BUFFER, color_bytes, None, GL_STREAM_DRAW)
            for pbo_id in depth_ids:
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_id)
                glBufferData(GL_PIXEL_UNPACK_BUFFER, depth_bytes, None, GL_STREAM_DRAW)
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
            self._cpu_pbo_color_ring = color_ids
            self._cpu_pbo_depth_ring = depth_ids
            self._cpu_pbo_index = 0
            self._cpu_pbo_color = color_ids[0] if color_ids else None
            self._cpu_pbo_depth = depth_ids[0] if depth_ids else None
            self._cpu_pbo_size = (w, h)
            print(f"[OpenXRViewer] CPU-path PBO ring created {w}x{h} x{ring_count}")
        except Exception as exc:
            print(f"[OpenXRViewer] CPU PBO init failed, using direct upload: {exc}")
            self._cpu_pbo_color = None
            self._cpu_pbo_depth = None
            self._cpu_pbo_color_ring = []
            self._cpu_pbo_depth_ring = []
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

