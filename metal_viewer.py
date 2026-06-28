import ctypes
import platform
import time

import glfw
import numpy as np


if platform.system() == "Darwin":
    import objc
    import Metal
    import Quartz

    objc.loadBundle(
        "QuartzCore",
        globals(),
        bundle_path=objc.pathForFramework("/System/Library/Frameworks/QuartzCore.framework"),
    )
    import QuartzCore
else:
    Metal = None
    Quartz = None
    QuartzCore = None


MTLPixelFormatRGBA8Unorm = 70
MTLPixelFormatBGRA8Unorm = 80
MTLPixelFormatR32Float = 55
MTLLoadActionClear = 2
MTLStoreActionStore = 1
MTLPrimitiveTypeTriangleStrip = 4


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
    int mode;
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

static float2 displaced_uv(float2 uv, float eye, texture2d<float> depthTex, sampler s, constant Uniforms& u) {
    float d = depthTex.sample(s, uv).r;
    d = pow(clamp(d, 0.0, 1.0), u.depthExponent);
    float shift = (u.convergence - d) * u.depthStrength * eye;
    return float2(clamp(uv.x + shift, 0.0, 1.0), uv.y);
}

fragment float4 fragment_main(
    VertexOut in [[stage_in]],
    texture2d<float> colorTex [[texture(0)]],
    texture2d<float> depthTex [[texture(1)]],
    constant Uniforms& u [[buffer(0)]]
) {
    constexpr sampler s(address::clamp_to_edge, filter::linear);
    float2 uv = in.uv;

    if (u.mode == 3) {
        float d = depthTex.sample(s, uv).r;
        return float4(d, d, d, 1.0);
    }

    if (u.mode == 4) {
        float2 luv = displaced_uv(uv, -u.eyeOffset, depthTex, s, u);
        float2 ruv = displaced_uv(uv,  u.eyeOffset, depthTex, s, u);
        float4 lc = colorTex.sample(s, luv);
        float4 rc = colorTex.sample(s, ruv);
        return float4(lc.r, rc.g, rc.b, 1.0);
    }

    if (u.mode == 5) {
        float eye = (fmod(floor(in.position.y), 2.0) < 1.0) ? -u.eyeOffset : u.eyeOffset;
        return colorTex.sample(s, displaced_uv(uv, eye, depthTex, s, u));
    }

    if (u.mode == 6) {
        float eye = (fmod(floor(in.position.x), 2.0) < 1.0) ? -u.eyeOffset : u.eyeOffset;
        return colorTex.sample(s, displaced_uv(uv, eye, depthTex, s, u));
    }

    if (u.mode == 2) {
        float eye = uv.y < 0.5 ? u.eyeOffset : -u.eyeOffset;
        float2 src = float2(uv.x, uv.y < 0.5 ? uv.y * 2.0 : (uv.y - 0.5) * 2.0);
        return colorTex.sample(s, displaced_uv(src, eye, depthTex, s, u));
    }

    float eye = uv.x < 0.5 ? -u.eyeOffset : u.eyeOffset;
    float2 src = float2(uv.x < 0.5 ? uv.x * 2.0 : (uv.x - 0.5) * 2.0, uv.y);
    return colorTex.sample(s, displaced_uv(src, eye, depthTex, s, u));
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
        self.window_size = frame_size
        self.stream_mode = stream_mode
        self._texture_size = None
        self._color_tex = None
        self._depth_tex = None
        self._has_real_frame = False
        self._modes = ["Full-SBS", "Half-SBS", "Half-TAB", "Depth Map", "Full-TAB", "Anaglyph", "Interleaved", "Interleaved-V"]

        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        glfw.window_hint(glfw.RESIZABLE, True)
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self.window = glfw.create_window(*self.window_size, self.title, None, None)
        if not self.window:
            raise RuntimeError("Failed to create GLFW window")
        glfw.set_key_callback(self.window, self.on_key_event)

        self.device = Metal.MTLCreateSystemDefaultDevice()
        if self.device is None:
            raise RuntimeError("No Metal device available")
        self.command_queue = self.device.newCommandQueue()
        self._attach_metal_layer()
        self._build_pipeline()

    def _glfw_cocoa_window(self):
        fn = getattr(glfw._glfw, "glfwGetCocoaWindow", None)
        if fn is None:
            raise RuntimeError("GLFW library lacks glfwGetCocoaWindow")
        fn.argtypes = [ctypes.c_void_p]
        fn.restype = ctypes.c_void_p
        ptr = fn(self.window)
        if not ptr:
            raise RuntimeError("glfwGetCocoaWindow returned null")
        return objc.objc_object(c_void_p=ptr)

    def _attach_metal_layer(self):
        ns_window = self._glfw_cocoa_window()
        view = ns_window.contentView()
        layer = QuartzCore.CAMetalLayer.layer()
        layer.setDevice_(self.device)
        layer.setPixelFormat_(MTLPixelFormatBGRA8Unorm)
        layer.setFramebufferOnly_(True)
        layer.setContentsScale_(ns_window.backingScaleFactor())
        view.setWantsLayer_(True)
        view.setLayer_(layer)
        self._ns_window = ns_window
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

    def _resize_drawable(self):
        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        fb_w, fb_h = max(1, fb_w), max(1, fb_h)
        self._metal_layer.setDrawableSize_((fb_w, fb_h))

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
        h, w = arr.shape[:2]
        region = Metal.MTLRegionMake2D(0, 0, w, h)
        texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow_(
            region,
            0,
            arr.ctypes.data_as(ctypes.c_void_p),
            bytes_per_row,
        )

    def update_frame(self, rgb, depth, current_fps=None, current_latency=None):
        if current_fps is not None:
            self.actual_fps = current_fps
        if current_latency is not None:
            self.total_latency = current_latency * 1000.0

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
            glfw.show_window(self.window)

    def _mode_id(self):
        if self.display_mode == "Depth Map":
            return 3
        if self.display_mode == "Anaglyph":
            return 4
        if self.display_mode == "Interleaved":
            return 5
        if self.display_mode == "Interleaved-V":
            return 6
        if self.display_mode in ("Half-TAB", "Full-TAB", "TAB"):
            return 2
        return 1

    def render(self):
        if self._color_tex is None or self._depth_tex is None:
            return
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

        uniforms = np.array(
            [
                self.ipd_uv * 0.5,
                self.depth_strength * self.depth_ratio,
                self.convergence,
                self.depth_exponent,
            ],
            dtype=np.float32,
        )
        mode = np.array([self._mode_id()], dtype=np.int32)
        uniform_bytes = uniforms.tobytes() + mode.tobytes() + b"\x00" * 12

        cmd = self.command_queue.commandBuffer()
        enc = cmd.renderCommandEncoderWithDescriptor_(pass_desc)
        enc.setRenderPipelineState_(self.pipeline)
        enc.setFragmentTexture_atIndex_(self._color_tex, 0)
        enc.setFragmentTexture_atIndex_(self._depth_tex, 1)
        enc.setFragmentBytes_length_atIndex_(uniform_bytes, len(uniform_bytes), 0)
        enc.drawPrimitives_vertexStart_vertexCount_(MTLPrimitiveTypeTriangleStrip, 0, 4)
        enc.endEncoding()
        cmd.presentDrawable_(drawable)
        cmd.commit()

    def capture_glfw_image(self):
        return None

    def on_key_event(self, window, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_TAB:
            idx = self._modes.index(self.display_mode) if self.display_mode in self._modes else 0
            self.display_mode = self._modes[(idx + 1) % len(self._modes)]
        elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
            self.depth_ratio = min(10.0, self.depth_ratio + 0.5)
        elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
            self.depth_ratio = max(0.0, self.depth_ratio - 0.5)
        elif key == glfw.KEY_0:
            self.depth_ratio = self.depth_ratio_original

    def stop(self):
        pass
