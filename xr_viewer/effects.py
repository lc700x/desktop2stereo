"""EffectsMixin: glow, frost/veil, border, panorama background, per-eye render."""

import math
import os
import time

import numpy as np
from PIL import Image
import moderngl
from OpenGL.GL import (
    glGenerateMipmap, GL_TEXTURE_2D,
    glDisable, GL_FRAMEBUFFER_SRGB,
    glClear, GL_DEPTH_BUFFER_BIT,
)

from .render import _view_mat_inv
from .constants import _BG_COLORS, _SCREEN_ENV_DEPTH_BIAS_M
from .glsl import (
    _GLOW_FRAG, _FROST_GLOW_VERT, _FROST_CURVED_VERT,
    _FROST_GLOW_FRAG, _FROST_VEIL_FRAG,
    _PANORAMA_VERT, _PANORAMA_FRAG,
)


class EffectsMixin:
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

    # Each of the 4 veil walls is its own TRIANGLE_STRIP, subdivided as a GRID
    # in BOTH axes: N depth steps (kills the perspective crease) AND M steps
    # along the wall's long edge (so edge/beam shading is sampled densely and
    # never kinks along a skinny triangle's diagonal -> straight extension).
    # Walls are NOT stitched to each other (cross-wall joins caused the blinking
    # corner seam); depth rows WITHIN a wall are stitched with degenerate joins.
    _FLAT_FROST_N = 8    # depth steps
    _FLAT_FROST_M = 8    # steps along the wall long edge
    # verts per wall = N rows * 2*(M+1) + (N-1)*2 degenerate join verts
    _FLAT_FROST_STRIDE = 8 * 2 * (8 + 1) + (8 - 1) * 2

    def _build_flat_frost_verts(self, front_half_w, front_half_h, N=8, M=8):
        sx = max(float(self.screen_width) * 0.5, 1e-6)
        sy = max(float(self.screen_height) * 0.5, 1e-6)
        fx = front_half_w / sx
        fy = front_half_h / sy
        verts = []

        # A wall spans screen-edge points A->B (param s in [0,1]) and extends
        # from the screen plane (t=0) to the front rectangle (t=1). Emit N depth
        # rows, each a strip of M+1 columns, stitched within the wall.
        def wall(ax, ay, uva, bx, by, uvb):
            def vtx(s, t):
                px = ax + s * (bx - ax)
                py = ay + s * (by - ay)
                x = px + t * (px * fx - px)
                y = py + t * (py * fy - py)
                u = uva[0] + s * (uvb[0] - uva[0])
                v = uva[1] + s * (uvb[1] - uva[1])
                return [x, y, t, u, v]
            prev_last = None
            for i in range(N):
                t0, t1 = i / N, (i + 1) / N
                row = []
                for j in range(M + 1):
                    s = j / M
                    row.append(vtx(s, t0))
                    row.append(vtx(s, t1))
                if prev_last is not None:
                    verts.extend(prev_last)   # degenerate join between depth rows
                    verts.extend(row[0])
                for vv in row:
                    verts.extend(vv)
                prev_last = row[-1]

        # Top wall: TL(-1,1) -> TR(1,1)
        wall(-1, 1, (0, 0),  1, 1, (1, 0))
        # Bottom wall: BL(-1,-1) -> BR(1,-1)
        wall(-1, -1, (0, 1),  1, -1, (1, 1))
        # Left wall: TL(-1,1) -> BL(-1,-1)
        wall(-1, 1, (0, 0),  -1, -1, (0, 1))
        # Right wall: TR(1,1) -> BR(1,-1)
        wall(1, 1, (1, 0),  1, -1, (1, 1))
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
            # UV v=1 in curved-screen convention is top (+Y); flat veil expects v=0 at top.
            # Flip v for texture sampling to match the flat veil orientation.
            return [float(world[0]), float(world[1]), float(world[2]), u, 1.0 - v, lx, ly, local_z]

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
        mode = self._active_glow_mode()
        if mode != 'glow' or passthrough_active:
            return
        if env_model_active:
            return
        self._render_glow(mgl_fbo, vp_mat)

    def _render_screen_foreground_effects(self, mgl_fbo, vp_mat, passthrough_active=False):
        if passthrough_active:
            return
        mode = self._active_glow_mode()
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
        # NOTE: do NOT cache by id(mat). vp_mat is a fresh per-eye local; after
        # the left eye returns, the right eye's vp_mat can reuse the same id(),
        # which would feed the left eye's matrix to the right eye's veil/frost
        # pass -> per-eye mismatch -> flicker.
        transposed = np.asarray(mat).T
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
            round(float(front_depth), 1), round(float(front_half_w), 1), round(float(front_half_h), 1),
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
        # 4 separate wall strips (no cross-wall stitch -> no blinking corner seam).
        stride = self._FLAT_FROST_STRIDE
        for w in range(4):
            vao.render(moderngl.TRIANGLE_STRIP, vertices=stride, first=w * stride)
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
        veil_alpha = float(getattr(self, '_frost_veil_alpha', 1.0))
        if veil_alpha <= 0.002:
            return
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
        # view_rot is rotation-only (translation zeroed) inverse == transpose.
        inv_view_rot = view_rot.T
        # proj inverse changes only when the projection does cache by its bytes.
        proj_f = proj_mat.astype(np.float32)
        proj_key = proj_f.tobytes()
        if getattr(self, '_panorama_inv_proj_key', None) == proj_key:
            inv_proj = self._panorama_inv_proj_val
        else:
            try:
                inv_proj = np.linalg.inv(proj_f)
            except Exception:
                return
            self._panorama_inv_proj_key = proj_key
            self._panorama_inv_proj_val = inv_proj

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
