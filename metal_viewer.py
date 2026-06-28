import platform
import time
from ctypes import c_void_p

import glfw
import numpy as np


if platform.system() == "Darwin":
    import objc
    import Metal
    import Quartz
    from utils import get_font_type, send_ctrl_cmd_f

    CAMetalLayer = getattr(Quartz, "CAMetalLayer", None)
    if CAMetalLayer is None:
        quartz_core = {}
        objc.loadBundle(
            "QuartzCore",
            quartz_core,
            bundle_path=objc.pathForFramework("/System/Library/Frameworks/QuartzCore.framework"),
        )
        CAMetalLayer = quartz_core.get("CAMetalLayer")
    if CAMetalLayer is None:
        raise RuntimeError("PyObjC Quartz framework does not expose CAMetalLayer")
else:
    Metal = None
    Quartz = None
    CAMetalLayer = None
    get_font_type = None
    send_ctrl_cmd_f = None


MTLPixelFormatRGBA8Unorm = 70
MTLPixelFormatBGRA8Unorm = 80
MTLPixelFormatR32Float = 55
MTLLoadActionClear = 2
MTLStoreActionStore = 1
MTLPrimitiveTypeTriangleStrip = 4
MTLBlendOperationAdd = 0
MTLBlendFactorSourceAlpha = 4
MTLBlendFactorOneMinusSourceAlpha = 5
MTLBlendFactorOne = 1


METAL_SHADER = r"""
#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float2 uv;
};

struct Uniforms {
    float eyeOffset;
    float depthStrength;
    float convergence;
    float depthExponent;
    float4 viewport;
    float mode;
    float featherEnabled;
    float featherWidth;
    float _pad;
};

vertex VertexOut vertex_main(uint vid [[vertex_id]]) {
    float2 positions[4] = {
        float2(-1.0, -1.0),
        float2( 1.0, -1.0),
        float2(-1.0,  1.0),
        float2( 1.0,  1.0),
    };
    float2 uvs[4] = {
        float2(0.0, 1.0),
        float2(1.0, 1.0),
        float2(0.0, 0.0),
        float2(1.0, 0.0),
    };
    VertexOut out;
    out.position = float4(positions[vid], 0.0, 1.0);
    out.uv = uvs[vid];
    return out;
}

struct OverlayVertex {
    float2 position;
    float2 uv;
};

vertex VertexOut overlay_vertex_main(
    const device OverlayVertex* vertices [[buffer(0)]],
    uint vid [[vertex_id]]
) {
    VertexOut out;
    out.position = float4(vertices[vid].position, 0.0, 1.0);
    out.uv = vertices[vid].uv;
    return out;
}

fragment float4 overlay_fragment_main(
    VertexOut in [[stage_in]],
    texture2d<float> overlayTex [[texture(0)]]
) {
    constexpr sampler s(address::clamp_to_edge, filter::linear);
    return overlayTex.sample(s, in.uv);
}

static float2 displaced_uv(float2 uv, float eye, texture2d<float> depthTex, sampler s, constant Uniforms& u) {
    float d = depthTex.sample(s, uv).r;
    d = pow(clamp(d, 0.0, 1.0), u.depthExponent);
    float shift = (u.convergence - d) * u.depthStrength * eye;
    return float2(clamp(uv.x + shift, 0.0, 1.0), uv.y);
}

static float3 spectral_r_ultrafast(float t) {
    float3 color1 = float3(0.0, 0.298, 0.651);
    float3 color2 = float3(0.0, 0.5, 0.0);
    float3 color3 = float3(1.0, 0.851, 0.0);
    float3 color4 = float3(0.988, 0.0, 0.0);

    float w1 = max(0.0, 1.0 - abs(t - 0.125) * 4.0);
    float w2 = max(0.0, 1.0 - abs(t - 0.375) * 4.0);
    float w3 = max(0.0, 1.0 - abs(t - 0.625) * 4.0);
    float w4 = max(0.0, 1.0 - abs(t - 0.875) * 4.0);
    float total = w1 + w2 + w3 + w4;
    if (total > 0.0) {
        w1 /= total;
        w2 /= total;
        w3 /= total;
        w4 /= total;
    }
    return color1 * w1 + color2 * w2 + color3 * w3 + color4 * w4;
}

fragment float4 fragment_main(
    VertexOut in [[stage_in]],
    texture2d<float> colorTex [[texture(0)]],
    texture2d<float> depthTex [[texture(1)]],
    constant Uniforms& u [[buffer(0)]]
) {
    constexpr sampler s(address::clamp_to_edge, filter::linear);
    float2 frag = in.position.xy;
    if (frag.x < u.viewport.x || frag.y < u.viewport.y ||
        frag.x >= u.viewport.x + u.viewport.z || frag.y >= u.viewport.y + u.viewport.w) {
        discard_fragment();
    }
    float2 uv = clamp((frag - u.viewport.xy) / u.viewport.zw, 0.0, 1.0);
    int mode = int(u.mode + 0.5);

    if (mode == 0) {
        float4 color = colorTex.sample(s, uv);
        if (u.featherEnabled > 0.5) {
            float left = uv.x;
            float right = 1.0 - uv.x;
            float top = uv.y;
            float bottom = 1.0 - uv.y;
            float feather = max(u.featherWidth, 0.0001);
            float falloff = smoothstep(0.0, feather, left) *
                            smoothstep(0.0, feather, right) *
                            smoothstep(0.0, feather, top) *
                            smoothstep(0.0, feather, bottom);
            color.rgb *= pow(falloff, 0.7);
        }
        return color;
    }

    if (mode == 7) {
        float4 color = colorTex.sample(s, displaced_uv(uv, u.eyeOffset, depthTex, s, u));
        if (u.featherEnabled > 0.5) {
            float left = uv.x;
            float right = 1.0 - uv.x;
            float top = uv.y;
            float bottom = 1.0 - uv.y;
            float feather = max(u.featherWidth, 0.0001);
            float falloff = smoothstep(0.0, feather, left) *
                            smoothstep(0.0, feather, right) *
                            smoothstep(0.0, feather, top) *
                            smoothstep(0.0, feather, bottom);
            color.rgb *= pow(falloff, 0.7);
        }
        return color;
    }

    if (mode == 3) {
        float d = depthTex.sample(s, uv).r;
        return float4(spectral_r_ultrafast(d), 1.0);
    }

    float4 color;
    if (mode == 4) {
        float2 luv = displaced_uv(uv, -u.eyeOffset, depthTex, s, u);
        float2 ruv = displaced_uv(uv,  u.eyeOffset, depthTex, s, u);
        float4 lc = colorTex.sample(s, luv);
        float4 rc = colorTex.sample(s, ruv);
        color = float4(lc.r, rc.g, rc.b, 1.0);
    } else if (mode == 5) {
        float eye = (fmod(floor(in.position.y), 2.0) < 1.0) ? -u.eyeOffset : u.eyeOffset;
        color = colorTex.sample(s, displaced_uv(uv, eye, depthTex, s, u));
    } else if (mode == 6) {
        float eye = (fmod(floor(in.position.x), 2.0) < 1.0) ? -u.eyeOffset : u.eyeOffset;
        color = colorTex.sample(s, displaced_uv(uv, eye, depthTex, s, u));
    } else if (mode == 2) {
        float eye = uv.y < 0.5 ? u.eyeOffset : -u.eyeOffset;
        float2 src = float2(uv.x, uv.y < 0.5 ? uv.y * 2.0 : (uv.y - 0.5) * 2.0);
        color = colorTex.sample(s, displaced_uv(src, eye, depthTex, s, u));
    } else {
        float eye = uv.x < 0.5 ? -u.eyeOffset : u.eyeOffset;
        float2 src = float2(uv.x < 0.5 ? uv.x * 2.0 : (uv.x - 0.5) * 2.0, uv.y);
        color = colorTex.sample(s, displaced_uv(src, eye, depthTex, s, u));
    }

    if (u.featherEnabled > 0.5) {
        float left = uv.x;
        float right = 1.0 - uv.x;
        float top = uv.y;
        float bottom = 1.0 - uv.y;
        float feather = max(u.featherWidth, 0.0001);
        float falloff = smoothstep(0.0, feather, left) *
                        smoothstep(0.0, feather, right) *
                        smoothstep(0.0, feather, top) *
                        smoothstep(0.0, feather, bottom);
        color.rgb *= pow(falloff, 0.7);
    }
    return color;
}
"""


def _as_numpy_rgb(rgb):
    if hasattr(rgb, "detach"):
        import torch

        t = rgb.detach()
        if t.ndim == 3 and t.shape[0] in (3, 4):
            t = t[:3].permute(1, 2, 0)
        elif t.ndim == 3 and t.shape[-1] >= 3:
            t = t[..., :3]
        else:
            raise ValueError(f"Unsupported RGB tensor shape: {tuple(t.shape)}")
        return t.contiguous().clamp(0, 255).to(torch.uint8).cpu().numpy()
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise ValueError(f"Unsupported RGB array shape: {arr.shape}")
    return np.ascontiguousarray(arr[..., :3].astype(np.uint8, copy=False))


def _as_numpy_depth(depth):
    if hasattr(depth, "detach"):
        return depth.detach().contiguous().float().cpu().numpy()
    return np.asarray(depth, dtype=np.float32)


def _rgba_from_rgb(rgb):
    h, w = rgb.shape[:2]
    rgba = np.empty((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = rgb[..., :3]
    rgba[..., 3] = 255
    return rgba


class StereoWindow:
    """macOS Metal local viewer. Keeps GLFW for window/events, uses CAMetalLayer for rendering."""

    uses_metal = True

    def __init__(
        self,
        capture_mode="Monitor",
        monitor_index=0,
        ipd=0.064,
        depth_ratio=1.0,
        convergence=0.0,
        display_mode="Half-SBS",
        fill_16_9=True,
        show_fps=True,
        use_3d=False,
        fix_aspect=False,
        stream_mode=None,
        lossless_scaling=False,
        specify_display=False,
        stereo_display_index=0,
        feather_enabled=False,
        frame_size=(1280, 720),
        use_cuda=False,
        cuda_device_id=0,
        vsync=False,
        **kwargs,
    ):
        if platform.system() != "Darwin":
            raise RuntimeError("Metal viewer is macOS-only")
        if stream_mode == "MJPEG":
            raise RuntimeError("Metal viewer does not support MJPEG readback yet")
        if Metal is None:
            raise RuntimeError("PyObjC Metal framework not available")

        self.title = "Stereo Viewer Metal"
        self.use_3d = use_3d
        self.capture_mode = capture_mode
        self.input_monitor_index = monitor_index
        self.ipd_uv = ipd
        self.depth_ratio = depth_ratio
        self.depth_ratio_original = depth_ratio
        self.depth_strength = 0.1
        self.depth_exponent = 1.45
        self.convergence = convergence
        self.display_mode = display_mode
        self.fill_16_9 = fill_16_9
        self.show_fps = show_fps
        self.actual_fps = 0.0
        self.total_latency = 0.0
        self.frame_size = frame_size
        self.aspect = self.frame_size[0] / self.frame_size[1]
        self.fix_aspect = fix_aspect
        self.window_size = frame_size
        self.stream_mode = stream_mode
        self.lossless_scaling = lossless_scaling
        self.specify_display = specify_display
        self.stereo_display_index = stereo_display_index
        self.vsync = vsync
        self._texture_size = None
        self._color_tex = None
        self._depth_tex = None
        self._has_real_frame = False
        self._modes = ["Full-SBS", "Half-SBS", "Half-TAB", "Depth Map", "Full-TAB", "Anaglyph", "Interleaved", "Interleaved-V"]
        self._last_window_position = None
        self._last_window_size = None
        self._fullscreen = False
        self.feather_enabled = feather_enabled
        self.feather_width = 0.02
        self.show_original_in_depth_mode = False
        self.last_depth_change_time = 0.0
        self.show_depth_ratio = False
        self.depth_display_duration = 2.0
        self.last_mouse_toggle_time = 0.0
        self.show_mouse_state = False
        self.mouse_display_duration = 2.0
        self.mouse_pass_through = False
        self.font = None
        self.font_type = get_font_type() if get_font_type is not None else None
        self.base_font_size = 60
        self.current_font_size = self.base_font_size
        self.text_padding = 10
        self.text_spacing = 5
        self.overlay_update_interval = 0.5
        self._overlay_cache = {
            "image": None,
            "fps_text": None,
            "latency_text": None,
            "depth_text": None,
            "mouse_text": None,
            "last_update": 0.0,
            "values_update": 0.0,
            "disp_fps": None,
            "disp_latency": None,
            "pos": (self.text_padding, self.text_padding),
        }
        self._overlay_tex = None
        self._overlay_size = None
        self._update_font()

        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        self.monitor_index = self.get_glfw_mon_index_mss(self.input_monitor_index)
        if self.specify_display:
            self.monitor_index = self.get_glfw_mon_index_mss(self.stereo_display_index)

        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        glfw.window_hint(glfw.RESIZABLE, True)
        glfw.window_hint(glfw.DECORATED, glfw.TRUE)
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        if hasattr(glfw, "COCOA_RETINA_FRAMEBUFFER"):
            glfw.window_hint(glfw.COCOA_RETINA_FRAMEBUFFER, glfw.TRUE)
        if (self.use_3d or self.capture_mode == "Window" and self.specify_display) and self.input_monitor_index == self.stereo_display_index:
            glfw.window_hint(glfw.MOUSE_PASSTHROUGH, glfw.TRUE)
            glfw.window_hint(glfw.DECORATED, glfw.FALSE)
            glfw.window_hint(glfw.FLOATING, glfw.TRUE)
            self.mouse_pass_through = True
        elif self.stream_mode == "RTMP":
            glfw.window_hint(glfw.RESIZABLE, False)

        self.window = glfw.create_window(*self.window_size, self.title, None, None)
        if not self.window:
            raise RuntimeError("Failed to create GLFW window")
        glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_HIDDEN)
        glfw.set_key_callback(self.window, self.on_key_event)
        glfw.set_window_size_callback(self.window, self._on_window_resize)
        self.update_monitor_size()

        self.device = Metal.MTLCreateSystemDefaultDevice()
        if self.device is None:
            raise RuntimeError("No Metal device available")
        self.command_queue = self.device.newCommandQueue()
        self._attach_metal_layer()
        self._build_pipeline()

    def _glfw_cocoa_window(self):
        if not hasattr(glfw, "get_cocoa_window"):
            raise RuntimeError("GLFW Python package lacks get_cocoa_window")
        ptr = glfw.get_cocoa_window(self.window)
        if not ptr:
            raise RuntimeError("glfwGetCocoaWindow returned null")
        return objc.objc_object(c_void_p=ptr)

    def _attach_metal_layer(self):
        ns_window = self._glfw_cocoa_window()
        view = ns_window.contentView()
        layer = CAMetalLayer.layer()
        layer.setDevice_(self.device)
        layer.setPixelFormat_(MTLPixelFormatBGRA8Unorm)
        layer.setFramebufferOnly_(True)
        layer.setContentsScale_(ns_window.backingScaleFactor())
        if hasattr(layer, "setDisplaySyncEnabled_"):
            layer.setDisplaySyncEnabled_(bool(self.vsync))
        if hasattr(layer, "setOpaque_"):
            layer.setOpaque_(True)
        view.setWantsLayer_(True)
        view.setLayer_(layer)
        self._ns_window = ns_window
        self._ns_view = view
        self._metal_layer = layer
        self._resize_drawable()

    def _build_pipeline(self):
        library, error = self.device.newLibraryWithSource_options_error_(METAL_SHADER, None, None)
        if library is None:
            raise RuntimeError(f"Metal shader compile failed: {error}")
        desc = Metal.MTLRenderPipelineDescriptor.alloc().init()
        desc.setVertexFunction_(library.newFunctionWithName_("vertex_main"))
        desc.setFragmentFunction_(library.newFunctionWithName_("fragment_main"))
        desc.colorAttachments().objectAtIndexedSubscript_(0).setPixelFormat_(MTLPixelFormatBGRA8Unorm)
        self.pipeline, error = self.device.newRenderPipelineStateWithDescriptor_error_(desc, None)
        if self.pipeline is None:
            raise RuntimeError(f"Metal pipeline creation failed: {error}")

        overlay_desc = Metal.MTLRenderPipelineDescriptor.alloc().init()
        overlay_desc.setVertexFunction_(library.newFunctionWithName_("overlay_vertex_main"))
        overlay_desc.setFragmentFunction_(library.newFunctionWithName_("overlay_fragment_main"))
        overlay_attachment = overlay_desc.colorAttachments().objectAtIndexedSubscript_(0)
        overlay_attachment.setPixelFormat_(MTLPixelFormatBGRA8Unorm)
        overlay_attachment.setBlendingEnabled_(True)
        overlay_attachment.setRgbBlendOperation_(MTLBlendOperationAdd)
        overlay_attachment.setAlphaBlendOperation_(MTLBlendOperationAdd)
        overlay_attachment.setSourceRGBBlendFactor_(MTLBlendFactorSourceAlpha)
        overlay_attachment.setDestinationRGBBlendFactor_(MTLBlendFactorOneMinusSourceAlpha)
        overlay_attachment.setSourceAlphaBlendFactor_(MTLBlendFactorOne)
        overlay_attachment.setDestinationAlphaBlendFactor_(MTLBlendFactorOneMinusSourceAlpha)
        self.overlay_pipeline, error = self.device.newRenderPipelineStateWithDescriptor_error_(overlay_desc, None)
        if self.overlay_pipeline is None:
            raise RuntimeError(f"Metal overlay pipeline creation failed: {error}")

    def _resize_drawable(self):
        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        fb_w, fb_h = max(1, fb_w), max(1, fb_h)
        if hasattr(self, "_ns_window"):
            scale = self._ns_window.backingScaleFactor()
            self._metal_layer.setContentsScale_(scale)
        if hasattr(self, "_ns_view"):
            self._metal_layer.setFrame_(self._ns_view.bounds())
        self._metal_layer.setDrawableSize_((fb_w, fb_h))

    def _on_window_resize(self, window, width, height):
        self.window_size = (max(1, width), max(1, height))
        self._update_font()
        self._overlay_cache["image"] = None
        self._overlay_cache["fps_text"] = None
        self._resize_drawable()

    def _make_texture(self, width, height, pixel_format):
        desc = Metal.MTLTextureDescriptor.texture2DDescriptorWithPixelFormat_width_height_mipmapped_(
            pixel_format, width, height, False
        )
        return self.device.newTextureWithDescriptor_(desc)

    def _ensure_textures(self, width, height):
        if self._texture_size == (width, height):
            return
        self._color_tex = self._make_texture(width, height, MTLPixelFormatRGBA8Unorm)
        self._depth_tex = self._make_texture(width, height, MTLPixelFormatR32Float)
        self._texture_size = (width, height)

    def _upload_texture(self, texture, arr, bytes_per_row):
        arr = np.ascontiguousarray(arr)
        h, w = arr.shape[:2]
        region = Metal.MTLRegionMake2D(0, 0, w, h)
        texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow_(
            region,
            0,
            arr.tobytes(),
            bytes_per_row,
        )

    def _update_font(self):
        if self.font_type is None:
            self.font = None
            return
        base_height = 1080
        scale_factor = min(self.frame_size[0] / base_height, 2.0)
        self.current_font_size = int(self.base_font_size * scale_factor * 0.8)
        try:
            from PIL import ImageFont

            self.font = ImageFont.truetype(self.font_type, self.current_font_size)
        except Exception:
            try:
                from PIL import ImageFont

                self.font = ImageFont.load_default()
            except Exception:
                self.font = None
        self.text_padding = max(5, int(self.current_font_size * 0.4))
        self.text_spacing = max(2, int(self.current_font_size * 0.2))
        self._overlay_cache["pos"] = (self.text_padding, self.text_padding)

    def _generate_overlay_image(self, fps_text, latency_text, depth_text, mouse_text):
        if self.font is None:
            return None
        try:
            from PIL import Image, ImageDraw
        except Exception:
            return None

        lines = []
        if self.show_fps and fps_text:
            lines.append(fps_text)
        if self.show_fps and self.total_latency > 0:
            latency_text = f"Latency: {self.total_latency:.0f} ms"
            lines.append(latency_text)
        if self.show_depth_ratio and depth_text:
            lines.append(depth_text)
        if self.show_mouse_state and mouse_text:
            lines.append(mouse_text)
        if not lines:
            return None

        padding = self.text_padding
        spacing = self.text_spacing
        dummy_img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        draw = ImageDraw.Draw(dummy_img)
        widths = []
        heights = []
        for line in lines:
            try:
                bbox = draw.textbbox((0, 0), line, font=self.font)
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
            except Exception:
                width, height = draw.textsize(line, font=self.font)
            widths.append(width)
            heights.append(height)

        overlay_w = max(widths) + padding * 2
        overlay_h = sum(heights) + spacing * (len(lines) - 1) + padding * 2
        overlay_img = Image.new("RGBA", (overlay_w, overlay_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay_img)
        y = padding
        for i, line in enumerate(lines):
            if "FPS:" in line:
                color = (0, 255, 0, 255)
            elif "Latency:" in line:
                color = (0, 255, 255, 255)
            elif "Depth:" in line:
                color = (255, 255, 255, 255)
            elif "Mouse:" in line:
                color = (255, 255, 0, 255)
            else:
                color = (255, 255, 255, 255)
            draw.text((padding, y), line, font=self.font, fill=color)
            y += heights[i] + spacing
        return np.array(overlay_img, dtype=np.uint8)

    def _upload_overlay_texture(self, rgba):
        if rgba is None:
            self._overlay_tex = None
            self._overlay_size = None
            return
        h, w = rgba.shape[:2]
        if w <= 0 or h <= 0:
            self._overlay_tex = None
            self._overlay_size = None
            return
        if self._overlay_size != (w, h):
            self._overlay_tex = self._make_texture(w, h, MTLPixelFormatRGBA8Unorm)
            self._overlay_size = (w, h)
        self._upload_texture(self._overlay_tex, rgba, w * 4)

    def _update_overlay_texture(self):
        if self.stream_mode is not None:
            self._upload_overlay_texture(None)
            return
        if self.font is None:
            self._upload_overlay_texture(None)
            return
        if not (self.show_fps or self.show_depth_ratio or self.show_mouse_state):
            self._upload_overlay_texture(None)
            return

        current_time = time.perf_counter()
        self.show_depth_ratio = current_time - self.last_depth_change_time < self.depth_display_duration
        self.show_mouse_state = current_time - self.last_mouse_toggle_time < self.mouse_display_duration
        if not (self.show_fps or self.show_depth_ratio or self.show_mouse_state):
            self._upload_overlay_texture(None)
            return

        cache = self._overlay_cache
        if cache.get("disp_fps") is None or (current_time - cache.get("values_update", 0.0)) >= self.overlay_update_interval:
            cache["disp_fps"] = self.actual_fps
            cache["disp_latency"] = self.total_latency
            cache["values_update"] = current_time

        fps_text = f"FPS: {cache['disp_fps']:.1f}" if self.show_fps else ""
        latency_text = f"Latency: {cache['disp_latency']:.1f} ms" if self.show_fps else ""
        depth_text = f"Depth: {self.depth_ratio:.1f}" if self.show_depth_ratio else ""
        mouse_text = f"Mouse: {'Pass' if self.mouse_pass_through else 'Normal'}" if self.show_mouse_state else ""

        if (
            fps_text == cache.get("fps_text")
            and latency_text == cache.get("latency_text")
            and depth_text == cache.get("depth_text")
            and mouse_text == cache.get("mouse_text")
            and cache.get("image") is not None
        ):
            return

        overlay_arr = self._generate_overlay_image(fps_text, latency_text, depth_text, mouse_text)
        cache["image"] = overlay_arr
        cache["fps_text"] = fps_text
        cache["latency_text"] = latency_text
        cache["depth_text"] = depth_text
        cache["mouse_text"] = mouse_text
        cache["last_update"] = current_time
        self._upload_overlay_texture(overlay_arr)

    def update_frame(self, rgb, depth, current_fps=None, current_latency=None):
        if current_fps is not None:
            self.actual_fps = current_fps
        if current_latency is not None:
            self.total_latency = current_latency * 1000.0
        if self.stream_mode is None:
            self._update_overlay_texture()

        rgb_np = _as_numpy_rgb(rgb)
        depth_np = np.ascontiguousarray(_as_numpy_depth(depth).astype(np.float32, copy=False))
        h, w = rgb_np.shape[:2]
        if depth_np.shape[:2] != (h, w):
            depth_np = np.resize(depth_np, (h, w)).astype(np.float32, copy=False)

        self._ensure_textures(w, h)
        rgba = _rgba_from_rgb(rgb_np)
        self._upload_texture(self._color_tex, rgba, w * 4)
        self._upload_texture(self._depth_tex, depth_np, w * 4)

        if not self._has_real_frame:
            self._has_real_frame = True
            if self.stream_mode != "MJPEG":
                glfw.show_window(self.window)
            self.position_on_monitor(self.monitor_index)
            if self.use_3d:
                if not self.specify_display:
                    self.toggle_fullscreen()
                self.apply_3d_settings()
            if self.stream_mode == "RTMP":
                if not self.specify_display:
                    self.move_to_adjacent_monitor(direction=1)
                self.fix_aspect = True
                if (self._texture_size == (self.mon_w, self.mon_h) and self.display_mode != "Full-SBS") or (
                    self._texture_size == (self.mon_w // 2, self.mon_h) and self.display_mode == "Full-SBS"
                ):
                    glfw.set_window_attrib(self.window, glfw.RESIZABLE, True)
                    glfw.set_window_attrib(self.window, glfw.DECORATED, glfw.FALSE)
                    self.toggle_fullscreen()
            if self.specify_display:
                if self.stream_mode != "RTMP" or self.lossless_scaling:
                    self.toggle_fullscreen()

    def _mode_id(self):
        if self.display_mode == "Depth Map":
            return 0 if self.show_original_in_depth_mode else 3
        if self.display_mode == "Anaglyph":
            return 4
        if self.display_mode == "Interleaved":
            return 5
        if self.display_mode == "Interleaved-V":
            return 6
        if self.display_mode in ("Half-TAB", "Full-TAB", "TAB"):
            return 2
        return 1

    def _uniform_bytes(self, viewport, mode_id, eye_offset=None):
        eye = self.ipd_uv * 0.5 if eye_offset is None else eye_offset
        uniforms = np.array(
            [
                eye,
                self.depth_strength * self.depth_ratio,
                self.convergence,
                self.depth_exponent,
                float(viewport[0]),
                float(viewport[1]),
                float(viewport[2]),
                float(viewport[3]),
                float(mode_id),
                1.0 if self.feather_enabled else 0.0,
                self.feather_width,
                0.0,
            ],
            dtype=np.float32,
        )
        return uniforms.tobytes()

    def render(self):
        if self._color_tex is None or self._depth_tex is None:
            return
        win_w, win_h = glfw.get_framebuffer_size(self.window)
        tex_w, tex_h = self._texture_size
        if self.fix_aspect and self._texture_size:
            if self.display_mode == "Full-SBS":
                glfw.set_window_aspect_ratio(self.window, 2 * tex_w, tex_h)
            elif self.display_mode == "Full-TAB":
                glfw.set_window_aspect_ratio(self.window, tex_w, 2 * tex_h)
            else:
                glfw.set_window_aspect_ratio(self.window, tex_w, tex_h)
        elif self._texture_size:
            glfw.set_window_aspect_ratio(self.window, glfw.DONT_CARE, glfw.DONT_CARE)

        self._resize_drawable()
        drawable = self._metal_layer.nextDrawable()
        if drawable is None:
            return

        pass_desc = Metal.MTLRenderPassDescriptor.renderPassDescriptor()
        attachment = pass_desc.colorAttachments().objectAtIndexedSubscript_(0)
        attachment.setTexture_(drawable.texture())
        attachment.setLoadAction_(MTLLoadActionClear)
        attachment.setStoreAction_(MTLStoreActionStore)
        attachment.setClearColor_(Metal.MTLClearColorMake(0.0, 0.0, 0.0, 1.0))

        cmd = self.command_queue.commandBuffer()
        enc = cmd.renderCommandEncoderWithDescriptor_(pass_desc)
        enc.setRenderPipelineState_(self.pipeline)
        enc.setFragmentTexture_atIndex_(self._color_tex, 0)
        enc.setFragmentTexture_atIndex_(self._depth_tex, 1)
        if self.display_mode in ("Full-SBS", "Half-SBS", "Half-TAB", "Full-TAB"):
            for viewport, eye_offset in self._stereo_viewports(win_w, win_h, tex_w, tex_h):
                uniform_bytes = self._uniform_bytes(viewport, 7, eye_offset)
                enc.setFragmentBytes_length_atIndex_(uniform_bytes, len(uniform_bytes), 0)
                enc.drawPrimitives_vertexStart_vertexCount_(MTLPrimitiveTypeTriangleStrip, 0, 4)
        else:
            viewport = self._calculate_viewport(win_w, win_h, tex_w, tex_h)
            uniform_bytes = self._uniform_bytes(viewport, self._mode_id())
            enc.setFragmentBytes_length_atIndex_(uniform_bytes, len(uniform_bytes), 0)
            enc.drawPrimitives_vertexStart_vertexCount_(MTLPrimitiveTypeTriangleStrip, 0, 4)
        self._render_overlay(enc, win_w, win_h)
        enc.endEncoding()
        cmd.presentDrawable_(drawable)
        cmd.commit()
        if self.stream_mode is None:
            glfw.poll_events()

    def _overlay_positions(self, win_w, win_h):
        base_x, base_y = self._overlay_cache.get("pos", (self.text_padding, self.text_padding))
        positions = [(base_x, base_y)]

        if self.display_mode == "Full-SBS" and self._texture_size:
            tex_w, tex_h = self._texture_size
            max_w, max_h = win_w / 2.0, win_h
            render_w, render_h = self._compute_render_size(max_w, max_h, tex_w, tex_h)
            center_y = win_h / 2.0
            left_vp = (
                int(win_w / 4.0 - render_w / 2),
                int(center_y - render_h / 2),
                render_w,
                render_h,
            )
            right_vp = (
                int(3 * win_w / 4.0 - render_w / 2),
                int(center_y - render_h / 2),
                render_w,
                render_h,
            )
            positions = [
                (left_vp[0] + base_x, win_h - (left_vp[1] + left_vp[3]) + base_y),
                (right_vp[0] + base_x, win_h - (right_vp[1] + right_vp[3]) + base_y),
            ]
        elif self.display_mode == "Half-SBS":
            positions = [
                (base_x, base_y),
                (max(base_x, win_w // 2 + base_x), base_y),
            ]
        elif self.display_mode in ("Full-TAB", "Half-TAB"):
            positions = [
                (base_x, base_y),
                (base_x, max(base_y, win_h // 2 + base_y)),
            ]
        return positions

    def _render_overlay(self, enc, win_w, win_h):
        if self.stream_mode is not None or self._overlay_tex is None or self._overlay_size is None:
            return
        tex_w, tex_h = self._overlay_size
        if win_w <= 0 or win_h <= 0 or tex_w <= 0 or tex_h <= 0:
            return

        enc.setRenderPipelineState_(self.overlay_pipeline)
        enc.setFragmentTexture_atIndex_(self._overlay_tex, 0)
        for x, y in self._overlay_positions(win_w, win_h):
            left = (x / win_w) * 2.0 - 1.0
            right = ((x + tex_w) / win_w) * 2.0 - 1.0
            top = 1.0 - (y / win_h) * 2.0
            bottom = 1.0 - ((y + tex_h) / win_h) * 2.0
            verts = np.array(
                [
                    left,
                    bottom,
                    0.0,
                    1.0,
                    right,
                    bottom,
                    1.0,
                    1.0,
                    left,
                    top,
                    0.0,
                    0.0,
                    right,
                    top,
                    1.0,
                    0.0,
                ],
                dtype=np.float32,
            )
            vertex_bytes = verts.tobytes()
            enc.setVertexBytes_length_atIndex_(vertex_bytes, len(vertex_bytes), 0)
            enc.drawPrimitives_vertexStart_vertexCount_(MTLPrimitiveTypeTriangleStrip, 0, 4)

    def capture_glfw_image(self):
        return None

    def on_key_event(self, window, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ENTER or key == glfw.KEY_SPACE:
            if self.stream_mode is None and not self.use_3d:
                self.toggle_fullscreen()
        elif key == glfw.KEY_D:
            if self.display_mode == "Depth Map":
                self.show_original_in_depth_mode = not self.show_original_in_depth_mode
        elif key == glfw.KEY_RIGHT:
            self.move_to_adjacent_monitor(+1)
        elif key == glfw.KEY_LEFT:
            self.move_to_adjacent_monitor(-1)
        elif key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_DOWN:
            self.depth_ratio = max(0.0, self.depth_ratio - 0.5)
            self.last_depth_change_time = time.perf_counter()
        elif key == glfw.KEY_UP:
            self.depth_ratio = min(10.0, self.depth_ratio + 0.5)
            self.last_depth_change_time = time.perf_counter()
        elif key == glfw.KEY_0:
            self.depth_ratio = self.depth_ratio_original
            self.last_depth_change_time = time.perf_counter()
        elif key == glfw.KEY_TAB:
            idx = self._modes.index(self.display_mode) if self.display_mode in self._modes else 0
            self.display_mode = self._modes[(idx + 1) % len(self._modes)]
        elif key == glfw.KEY_F:
            self.show_fps = not self.show_fps
            self._overlay_cache["last_update"] = 0.0
        elif key == glfw.KEY_B:
            self.feather_enabled = not self.feather_enabled
            print(f"Edge feathering: {'ON' if self.feather_enabled else 'OFF'}")
        elif key == glfw.KEY_A:
            self.fill_16_9 = not self.fill_16_9
            self._overlay_cache["last_update"] = 0.0
        elif key == glfw.KEY_L:
            self.fix_aspect = not self.fix_aspect
            self._overlay_cache["last_update"] = 0.0
        elif key == glfw.KEY_M:
            current_state = glfw.get_window_attrib(self.window, glfw.MOUSE_PASSTHROUGH)
            new_state = not current_state
            glfw.set_window_attrib(self.window, glfw.MOUSE_PASSTHROUGH, new_state)
            self.mouse_pass_through = new_state
            self.last_mouse_toggle_time = time.perf_counter()
            self.show_mouse_state = True

    def get_glfw_mon_index_mss(self, mss_monitor_index=1):
        try:
            import mss
        except ImportError:
            print("[StereoWindow] mss not installed; using default monitor.")
            return 0

        with mss.mss() as sct:
            mss_monitors = sct.monitors

        if mss_monitor_index < 1 or mss_monitor_index >= len(mss_monitors):
            print(f"[StereoWindow] Invalid MSS monitor index {mss_monitor_index}, defaulting to 1.")
            mss_monitor_index = 1

        mss_mon = mss_monitors[mss_monitor_index]
        mss_x, mss_y = mss_mon["left"], mss_mon["top"]
        mss_w, mss_h = mss_mon["width"], mss_mon["height"]

        glfw_monitors = glfw.get_monitors()
        if not glfw_monitors:
            print("[StereoWindow] No GLFW monitors detected.")
            return 0

        for i, gmon in enumerate(glfw_monitors):
            gx, gy = glfw.get_monitor_pos(gmon)
            gvm = glfw.get_video_mode(gmon)
            gw, gh = gvm.size.width, gvm.size.height
            if abs(gx - mss_x) <= 5 and abs(gy - mss_y) <= 5 and abs(gw - mss_w) <= 5 and abs(gh - mss_h) <= 5:
                return i

        print("[StereoWindow] No matching GLFW monitor found for MSS index, defaulting to primary.")
        return 0

    def position_on_monitor(self, monitor_index=0):
        monitors = glfw.get_monitors()
        if monitor_index >= len(monitors):
            return
        monitor = monitors[monitor_index]
        mon_x, mon_y = glfw.get_monitor_pos(monitor)
        vidmode = glfw.get_video_mode(monitor)
        mon_w, mon_h = vidmode.size.width, vidmode.size.height
        if self.window_size[0] >= mon_w or self.window_size[1] >= mon_h:
            if self.window_size[0] >= mon_w:
                target_w = mon_w
                target_h = int(self.window_size[1] * mon_w / self.window_size[0])
                self.window_size = (target_w, target_h)
            if self.window_size[1] >= mon_h:
                target_h = mon_h
                target_w = int(self.window_size[0] * mon_h / self.window_size[1])
                self.window_size = (target_w, target_h)
            glfw.set_window_size(self.window, self.window_size[0], self.window_size[1])
        x = mon_x + (mon_w - self.window_size[0]) // 2
        y = mon_y + (mon_h - self.window_size[1]) // 2
        glfw.set_window_pos(self.window, x, y)
        self._resize_drawable()

    def _display_frame_size(self, tex_w, tex_h):
        if self.display_mode == "Full-SBS":
            return 2 * tex_w, tex_h
        if self.display_mode == "Full-TAB":
            return tex_w, 2 * tex_h
        return tex_w, tex_h

    def _compute_render_size(self, max_w, max_h, src_w, src_h):
        if src_w == 0 or src_h == 0:
            return 0, 0
        scale = min(max_w / src_w, max_h / src_h)
        return (
            max(1, int(round(src_w * scale))),
            max(1, int(round(src_h * scale))),
        )

    def _calculate_viewport(self, win_w, win_h, tex_w, tex_h):
        disp_w, disp_h = self._display_frame_size(tex_w, tex_h)
        if self.fill_16_9:
            view_w, view_h = self._compute_render_size(win_w, win_h, disp_w, disp_h)
        else:
            target_aspect = disp_h / disp_w
            try:
                window_aspect = win_h / win_w
            except ZeroDivisionError:
                window_aspect = 9.0 / 16.0
            if window_aspect <= target_aspect:
                view_h = win_h
                view_w = int(view_h / target_aspect)
            else:
                view_w = win_w
                view_h = int(view_w * target_aspect)
        offset_x = (win_w - view_w) // 2
        offset_y = (win_h - view_h) // 2
        return (offset_x, offset_y, view_w, view_h)

    def _stereo_viewports(self, win_w, win_h, tex_w, tex_h):
        if self.fill_16_9:
            if self.display_mode == "Full-SBS":
                src_w, src_h = tex_w, tex_h
                max_w, max_h = win_w / 2.0, win_h
                render_w, render_h = self._compute_render_size(max_w, max_h, src_w, src_h)
                center_y = win_h / 2.0
                return [
                    ((int(win_w / 4.0 - render_w / 2), int(center_y - render_h / 2), render_w, render_h), -self.ipd_uv / 2.0),
                    ((int(3 * win_w / 4.0 - render_w / 2), int(center_y - render_h / 2), render_w, render_h), self.ipd_uv / 2.0),
                ]
            if self.display_mode == "Half-SBS":
                src_w, src_h = tex_w / 2.0, tex_h
                max_w, max_h = win_w / 2.0, win_h
                render_w, render_h = self._compute_render_size(max_w, max_h, src_w, src_h)
                center_y = win_h / 2.0
                return [
                    ((int(win_w / 4.0 - render_w / 2), int(center_y - render_h / 2), render_w, render_h), -self.ipd_uv / 2.0),
                    ((int(3 * win_w / 4.0 - render_w / 2), int(center_y - render_h / 2), render_w, render_h), self.ipd_uv / 2.0),
                ]
            if self.display_mode == "Half-TAB":
                src_w, src_h = tex_w, tex_h / 2.0
                max_w, max_h = win_w, win_h / 2.0
                render_w, render_h = self._compute_render_size(max_w, max_h, src_w, src_h)
                return [
                    ((int(win_w / 2.0 - render_w / 2), int(win_h / 4.0 - render_h / 2), render_w, render_h), -self.ipd_uv / 2.0),
                    ((int(win_w / 2.0 - render_w / 2), int(3 * win_h / 4.0 - render_h / 2), render_w, render_h), self.ipd_uv / 2.0),
                ]
            if self.display_mode == "Full-TAB":
                src_w, src_h = tex_w, tex_h
                max_w, max_h = win_w, win_h / 2.0
                render_w, render_h = self._compute_render_size(max_w, max_h, src_w, src_h)
                return [
                    ((int(win_w / 2.0 - render_w / 2), int(win_h / 4.0 - render_h / 2), render_w, render_h), -self.ipd_uv / 2.0),
                    ((int(win_w / 2.0 - render_w / 2), int(3 * win_h / 4.0 - render_h / 2), render_w, render_h), self.ipd_uv / 2.0),
                ]

        disp_w, disp_h = self._display_frame_size(tex_w, tex_h)
        target_aspect = disp_h / disp_w
        try:
            window_aspect = win_h / win_w
        except ZeroDivisionError:
            window_aspect = 9.0 / 16.0
        if window_aspect <= target_aspect:
            view_h = win_h
            view_w = int(view_h / target_aspect)
        else:
            view_w = win_w
            view_h = int(view_w * target_aspect)
        offset_x = (win_w - view_w) // 2
        offset_y = (win_h - view_h) // 2
        if self.display_mode in ("Full-SBS", "Half-SBS"):
            return [
                ((offset_x, offset_y, view_w // 2, view_h), -self.ipd_uv / 2.0),
                ((offset_x + view_w // 2, offset_y, view_w // 2, view_h), self.ipd_uv / 2.0),
            ]
        return [
            ((offset_x, offset_y + view_h // 2, view_w, view_h // 2), -self.ipd_uv / 2.0),
            ((offset_x, offset_y, view_w, view_h // 2), self.ipd_uv / 2.0),
        ]

    def update_monitor_size(self):
        monitors = glfw.get_monitors()
        if not monitors:
            self.mon_w, self.mon_h = self.window_size
            return
        monitor_index = min(self.monitor_index, len(monitors) - 1)
        vidmode = glfw.get_video_mode(monitors[monitor_index])
        self.mon_w, self.mon_h = vidmode.size.width, vidmode.size.height

    def apply_3d_settings(self):
        if self.monitor_index == self.get_glfw_mon_index_mss(self.input_monitor_index):
            glfw.set_window_attrib(self.window, glfw.FLOATING, glfw.TRUE)
        else:
            glfw.set_window_attrib(self.window, glfw.FLOATING, glfw.FALSE)

    def get_current_monitor(self):
        monitors = glfw.get_monitors()
        if not monitors:
            return None

        win_x, win_y = glfw.get_window_pos(self.window)
        win_w, win_h = glfw.get_window_size(self.window)
        window_center_x = win_x + win_w // 2
        window_center_y = win_y + win_h // 2

        for monitor in monitors:
            monitor_x, monitor_y = glfw.get_monitor_pos(monitor)
            vidmode = glfw.get_video_mode(monitor)
            if (
                monitor_x <= window_center_x < monitor_x + vidmode.size.width
                and monitor_y <= window_center_y < monitor_y + vidmode.size.height
            ):
                return monitor
        return monitors[0]

    def move_to_adjacent_monitor(self, direction):
        monitors = glfw.get_monitors()
        if len(monitors) > 1:
            new_index = (self.monitor_index + direction) % len(monitors)
            self.position_on_monitor(new_index)
            self.monitor_index = new_index
        else:
            self.position_on_monitor(self.monitor_index)
        if self.use_3d:
            self.apply_3d_settings()
        self.update_monitor_size()

    def get_glfw_monitor_index(self, monitor):
        monitors = glfw.get_monitors()
        for idx, mon in enumerate(monitors):
            if mon == monitor:
                return idx
        return -1

    def toggle_fullscreen(self):
        current_monitor = self.get_current_monitor()
        if not current_monitor:
            return

        if not self._fullscreen:
            if (
                not self.use_3d
                and self.capture_mode == "Window"
                and self.get_glfw_mon_index_mss(self.input_monitor_index) == self.get_glfw_monitor_index(current_monitor)
            ):
                glfw.set_window_attrib(self.window, glfw.MOUSE_PASSTHROUGH, True)
                glfw.set_window_attrib(self.window, glfw.FLOATING, glfw.TRUE)
            if send_ctrl_cmd_f is not None:
                send_ctrl_cmd_f()
            self._fullscreen = True
        else:
            if send_ctrl_cmd_f is not None:
                send_ctrl_cmd_f()
            self._fullscreen = False
            glfw.set_window_attrib(self.window, glfw.MOUSE_PASSTHROUGH, False)
            self.mouse_pass_through = False
        self.window_size = glfw.get_window_size(self.window)
        self._resize_drawable()

    def stop(self):
        pass