# viewer.py
import ctypes, os, sys
import glfw, torch
import moderngl
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont
from OpenGL.GL import *
# Get OS name and settings
from utils import OS_NAME, crop_icon, get_font_type, DEVICE_INFO
# 3D monitor mode to hide viewer
if OS_NAME == "Windows":
    from utils import hide_window_from_capture, show_window_in_capture
elif OS_NAME == "Darwin":
    from utils import send_ctrl_cmd_f
BACKEND = None
# NVIDIA CUDA Version
if "NVIDIA" in DEVICE_INFO:
    BACKEND = "CUDA"
    class CUDART_GL:
        """CUDA-OpenGL interop helper (based on local_viewer.py)."""

        # cudaGraphicsRegisterFlags
        CUDA_GRAPHICS_REGISTER_FLAGS_NONE = 0
        CUDA_GRAPHICS_REGISTER_FLAGS_READ_ONLY = 1
        CUDA_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD = 2
        # cudaMemcpyKind
        CUDA_MEMCPY_HOST_TO_DEVICE = 1
        CUDA_MEMCPY_DEVICE_TO_DEVICE = 3

        def __init__(self, device_id=0):
            # Locate cudart library
            torch_dir = os.path.dirname(torch.__file__)
            site_packages = os.path.dirname(torch_dir)
            candidates = [
                os.path.join(torch_dir, "lib"),
                os.path.join(site_packages, "nvidia", "cuda_runtime", "lib"),
                os.path.join(site_packages, "nvidia", "cuda_runtime", "bin"),
            ]
            cudart_path = None
            for lib_dir in candidates:
                if not os.path.exists(lib_dir):
                    continue
                for f in os.listdir(lib_dir):
                    if sys.platform == "win32":
                        if f.startswith("cudart64") and f.endswith(".dll"):
                            cudart_path = os.path.join(lib_dir, f)
                            break
                    else:
                        if f.startswith("libcudart") and ".so" in f:
                            cudart_path = os.path.join(lib_dir, f)
                            break
                if cudart_path:
                    break
            if not cudart_path:
                raise RuntimeError("Could not find cudart in torch/lib or nvidia/cuda_runtime/lib")

            # Load library
            if sys.platform == "win32":
                self.lib = ctypes.WinDLL(cudart_path)
            else:
                self.lib = ctypes.CDLL(cudart_path)

            # Set argument types
            self.lib.cudaSetDevice.argtypes = [ctypes.c_int]
            self.lib.cudaSetDevice.restype = ctypes.c_int
            self.lib.cudaGraphicsGLRegisterBuffer.argtypes = [
                ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_uint
            ]
            self.lib.cudaGraphicsGLRegisterBuffer.restype = ctypes.c_int
            self.lib.cudaGraphicsUnregisterResource.argtypes = [ctypes.c_void_p]
            self.lib.cudaGraphicsUnregisterResource.restype = ctypes.c_int
            self.lib.cudaGraphicsMapResources.argtypes = [
                ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p
            ]
            self.lib.cudaGraphicsMapResources.restype = ctypes.c_int
            self.lib.cudaGraphicsUnmapResources.argtypes = [
                ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p
            ]
            self.lib.cudaGraphicsUnmapResources.restype = ctypes.c_int
            self.lib.cudaGraphicsResourceGetMappedPointer.argtypes = [
                ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p
            ]
            self.lib.cudaGraphicsResourceGetMappedPointer.restype = ctypes.c_int
            self.lib.cudaMemcpy.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
            ]
            self.lib.cudaMemcpy.restype = ctypes.c_int
            # Async variant (used with the PyTorch stream to avoid a full device sync)
            self.lib.cudaMemcpyAsync.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p
            ]
            self.lib.cudaMemcpyAsync.restype = ctypes.c_int
            self.lib.cudaSetDevice(device_id)

        def register_buffer(self, pbo_id):
            resource = ctypes.c_void_p()
            # WriteDiscard: we overwrite the whole buffer every frame -> fastest, and
            # semantically correct (we write to the mapped pointer, not read from it).
            res = self.lib.cudaGraphicsGLRegisterBuffer(
                ctypes.byref(resource), pbo_id,
                self.CUDA_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD
            )
            if res != 0:
                raise RuntimeError(f"cudaGraphicsGLRegisterBuffer failed: {res}")
            return resource

        def unregister_resource(self, resource):
            self.lib.cudaGraphicsUnregisterResource(resource)

        def map_resource(self, resource, stream=None):
            stream_ptr = ctypes.c_void_p(stream) if stream else None
            res = self.lib.cudaGraphicsMapResources(1, ctypes.byref(resource), stream_ptr)
            if res != 0:
                raise RuntimeError(f"cudaGraphicsMapResources failed: {res}")
            ptr = ctypes.c_void_p()
            size = ctypes.c_size_t()
            res = self.lib.cudaGraphicsResourceGetMappedPointer(ctypes.byref(ptr), ctypes.byref(size), resource)
            if res != 0:
                self.lib.cudaGraphicsUnmapResources(1, ctypes.byref(resource), stream_ptr)
                raise RuntimeError(f"cudaGraphicsResourceGetMappedPointer failed: {res}")
            return ptr.value

        def unmap_resource(self, resource, stream=None):
            stream_ptr = ctypes.c_void_p(stream) if stream else None
            self.lib.cudaGraphicsUnmapResources(1, ctypes.byref(resource), stream_ptr)

        def memcpy_d2d(self, dst_ptr, src_ptr, size):
            # cudaMemcpyDeviceToDevice = 3 (synchronous w.r.t. host)
            res = self.lib.cudaMemcpy(dst_ptr, src_ptr, size, self.CUDA_MEMCPY_DEVICE_TO_DEVICE)
            if res != 0:
                raise RuntimeError(f"cudaMemcpy failed: {res}")

        def memcpy_h2d(self, dst_ptr, src_ptr, size):
            # Host(pinned)->Device upload straight into the mapped PBO (synchronous).
            res = self.lib.cudaMemcpy(dst_ptr, src_ptr, size, self.CUDA_MEMCPY_HOST_TO_DEVICE)
            if res != 0:
                raise RuntimeError(f"cudaMemcpy (H2D) failed: {res}")


elif "AMD" in DEVICE_INFO:
    BACKEND = "HIP"
    class CUDART_GL:
        """
        HIP-OpenGL interop helper, performance-tuned for AMD GPUs.
        Equivalent to the CUDA version, but uses the HIP runtime.
        """

        HIP_SUCCESS = 0
        # hipGraphicsRegisterFlagsWriteDiscard == 2 (we overwrite the whole buffer
        # every frame, so WriteDiscard is both correct and the fastest mode).
        HIP_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD = 2
        HIP_MEMCPY_HOST_TO_DEVICE = 1            # hipMemcpyHostToDevice
        HIP_MEMCPY_DEVICE_TO_DEVICE = 3          # hipMemcpyDeviceToDevice

        def __init__(self, device_id=0):
            # 1. Locate the HIP runtime library (libamdhip64.so / amdhip64.dll)
            torch_dir = os.path.dirname(torch.__file__)
            site_packages = os.path.dirname(torch_dir)

            # Common install locations; extend as needed.
            candidates = [
                os.path.join(torch_dir, "lib"),
                os.path.join(site_packages, "_rocm_sdk_core", "bin"),
                os.path.join(site_packages, "_rocm_sdk_core", "lib"),
                os.path.join(site_packages, "_rocm_sdk_devel", "bin"),
                os.path.join(site_packages, "_rocm_sdk_devel", "lib"),
                
                # system level hip
                "/opt/rocm/lib/",
                "/usr/lib/x86_64-linux-gnu",
            ]

            # check for HIP path in environment variables (common on Windows with AMD drivers)
            if sys.platform == "win32":
                if "HIP_PATH" in os.environ:
                    candidates.append(os.path.join(os.environ["HIP_PATH"], "bin"))

            hip_path = None
            for lib_dir in candidates:
                if not os.path.exists(lib_dir):
                    continue
                for f in os.listdir(lib_dir):
                    if sys.platform == "win32":
                        if f.startswith("amdhip64") and f.endswith(".dll"):
                            hip_path = os.path.join(lib_dir, f)
                            break
                    else:
                        if f.startswith("libamdhip64") and ".so" in f:
                            hip_path = os.path.join(lib_dir, f)
                            break
                if hip_path:
                    break

            if not hip_path:
                raise RuntimeError(
                    "Could not find libamdhip64.so (Linux) or amdhip64.dll (Windows) "
                    "in torch/lib or elsewhere."
                )

            # 2. Load the library
            if sys.platform == "win32":
                self.lib = ctypes.WinDLL(hip_path)
            else:
                self.lib = ctypes.CDLL(hip_path)

            # 3. Set argument types and return types
            # hipSetDevice
            self.lib.hipSetDevice.argtypes = [ctypes.c_int]
            self.lib.hipSetDevice.restype = ctypes.c_int

            # hipGraphicsGLRegisterBuffer
            self.lib.hipGraphicsGLRegisterBuffer.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),   # resource (output)
                ctypes.c_uint,                     # GLuint buffer
                ctypes.c_uint,                     # flags
            ]
            self.lib.hipGraphicsGLRegisterBuffer.restype = ctypes.c_int

            # hipGraphicsUnregisterResource
            self.lib.hipGraphicsUnregisterResource.argtypes = [ctypes.c_void_p]
            self.lib.hipGraphicsUnregisterResource.restype = ctypes.c_int

            # hipGraphicsMapResources: stream now accepted (0 = default stream).
            self.lib.hipGraphicsMapResources.argtypes = [
                ctypes.c_int,                      # count
                ctypes.POINTER(ctypes.c_void_p),   # *resources
                ctypes.c_void_p,                   # hipStream_t (0 or stream pointer)
            ]
            self.lib.hipGraphicsMapResources.restype = ctypes.c_int

            # hipGraphicsUnmapResources
            self.lib.hipGraphicsUnmapResources.argtypes = [
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_void_p,                   # hipStream_t
            ]
            self.lib.hipGraphicsUnmapResources.restype = ctypes.c_int

            # hipGraphicsResourceGetMappedPointer
            self.lib.hipGraphicsResourceGetMappedPointer.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),   # devPtr (output)
                ctypes.POINTER(ctypes.c_size_t),   # size (output)
                ctypes.c_void_p,                   # resource
            ]
            self.lib.hipGraphicsResourceGetMappedPointer.restype = ctypes.c_int

            # hipMemcpy
            self.lib.hipMemcpy.argtypes = [
                ctypes.c_void_p,   # dst
                ctypes.c_void_p,   # src
                ctypes.c_size_t,   # sizeBytes
                ctypes.c_int,      # kind (e.g. hipMemcpyDeviceToDevice)
            ]
            self.lib.hipMemcpy.restype = ctypes.c_int

            # optional: hipStreamCreate / hipStreamDestroy for advanced pipelining
            # (not used by default, kept for future use)
            self.lib.hipStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
            self.lib.hipStreamCreate.restype = ctypes.c_int
            self.lib.hipStreamDestroy.argtypes = [ctypes.c_void_p]
            self.lib.hipStreamDestroy.restype = ctypes.c_int

            # 4. Set the active device
            res = self.lib.hipSetDevice(device_id)
            if res != self.HIP_SUCCESS:
                raise RuntimeError(f"hipSetDevice({device_id}) failed with error {res}")

        def register_buffer(self, pbo_id: int):
            """
            Register an OpenGL buffer for HIP access.
            Uses the WRITE_DISCARD flag because we always overwrite the whole buffer.
            """
            resource = ctypes.c_void_p()
            res = self.lib.hipGraphicsGLRegisterBuffer(
                ctypes.byref(resource),
                pbo_id,
                self.HIP_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD,
            )
            if res != self.HIP_SUCCESS:
                raise RuntimeError(f"hipGraphicsGLRegisterBuffer failed: {res}")
            return resource

        def unregister_resource(self, resource):
            """Unregister a previously registered graphics resource."""
            res = self.lib.hipGraphicsUnregisterResource(resource)
            if res != self.HIP_SUCCESS:
                raise RuntimeError(f"hipGraphicsUnregisterResource failed: {res}")

        def map_resource(self, resource, stream=None):
            """
            Map a graphics resource and return the device pointer.
            Optionally provide a HIP stream for asynchronous operation.
            If stream is None, the default (null) stream is used.
            """
            stream_ptr = stream if stream is not None else ctypes.c_void_p(0)
            res = self.lib.hipGraphicsMapResources(1, ctypes.byref(resource), stream_ptr)
            if res != self.HIP_SUCCESS:
                raise RuntimeError(f"hipGraphicsMapResources failed: {res}")

            dev_ptr = ctypes.c_void_p()
            size = ctypes.c_size_t()
            res = self.lib.hipGraphicsResourceGetMappedPointer(
                ctypes.byref(dev_ptr),
                ctypes.byref(size),
                resource,
            )
            if res != self.HIP_SUCCESS:
                # Unmap to leave a clean state on failure
                self.lib.hipGraphicsUnmapResources(1, ctypes.byref(resource), stream_ptr)
                raise RuntimeError(f"hipGraphicsResourceGetMappedPointer failed: {res}")
            return dev_ptr.value

        def unmap_resource(self, resource, stream=None):
            """Unmap a previously mapped graphics resource."""
            stream_ptr = stream if stream is not None else ctypes.c_void_p(0)
            res = self.lib.hipGraphicsUnmapResources(1, ctypes.byref(resource), stream_ptr)
            if res != self.HIP_SUCCESS:
                raise RuntimeError(f"hipGraphicsUnmapResources failed: {res}")

        def memcpy_d2d(self, dst_ptr, src_ptr, size):
            """Device-to-device memory copy (hipMemcpyDeviceToDevice)."""
            res = self.lib.hipMemcpy(
                dst_ptr,
                src_ptr,
                size,
                self.HIP_MEMCPY_DEVICE_TO_DEVICE,
            )
            if res != self.HIP_SUCCESS:
                raise RuntimeError(f"hipMemcpy failed: {res}")

        def memcpy_h2d(self, dst_ptr, src_ptr, size):
            """Host(pinned)->Device upload straight into the mapped PBO (synchronous)."""
            res = self.lib.hipMemcpy(
                dst_ptr,
                src_ptr,
                size,
                self.HIP_MEMCPY_HOST_TO_DEVICE,
            )
            if res != self.HIP_SUCCESS:
                raise RuntimeError(f"hipMemcpy (H2D) failed: {res}")

# Shaders as constants (unchanged)
VERTEX_SHADER = """
    #version 330
    in vec2 in_position;
    in vec2 in_uv;
    out vec2 uv;
    void main() {
        uv = in_uv;
        gl_Position = vec4(in_position, 0.0, 1.0);
    }
    """

FRAGMENT_SHADER = """
    #version 330
    // Optimized stereoscopic inpainting with push-pull method
    
    in vec2 uv;
    out vec4 frag_color;

    uniform sampler2D tex_color;   // RGB image
    uniform sampler2D tex_depth;   // Single-channel depth (0 = near, 1 = far)
    uniform vec2 u_resolution;     // viewport resolution
    uniform float u_eye_offset;    // e.g. +/-0.03 (positive = right eye)
    uniform float u_depth_strength;// parallax intensity
    uniform float u_convergence;   // depth value at screen plane (0..1)
    uniform float u_roll;          // screen roll (radians), rotates parallax direction

    // Optimized inpainting controls
    uniform float u_search_radius = 12.0;  // horizontal search distance
    uniform float u_depth_tolerance = 0.012; // background detection threshold
    uniform float u_blur_radius = 2.5;     // final smoothing
    // Edge feathering controls
    uniform int u_feather_enabled;
    uniform float u_feather_width;
    uniform vec4 u_viewport;

    // Rounded corners
    uniform float u_corner_radius = 0.0;

    vec2 pixel_size = 1.0 / u_resolution;

    // Precomputed parallax direction (set once in main, reused by helpers)
    vec2  g_par_dir;     // normalized roll direction (cos,sin) * sign(eye_offset)
    float g_sweep_sign;  // +1 / -1: side the background is revealed from

    // SOFT DISOCCLUSION CONFIDENCE (0 = none, 1 = full)
    // Returns a smooth value instead of a hard bool -> blends inpaint in to hide seams.
    float disocclusion_confidence(vec2 base_uv, vec2 shifted_uv) {
        // Out-of-bounds shifted UV -> fully disoccluded
        if (shifted_uv.x < 0.0 || shifted_uv.x > 1.0 ||
            shifted_uv.y < 0.0 || shifted_uv.y > 1.0)
            return 1.0;

        // Fast depth discontinuity check (2-tap) along parallax direction
        vec2 step2 = g_par_dir * pixel_size * 2.0;
        float d_left  = texture(tex_depth, base_uv - step2).r;
        float d_right = texture(tex_depth, base_uv + step2).r;
        float jump = abs(d_left - d_right);

        // Smooth ramp instead of hard threshold (soft edges -> fewer seams)
        return smoothstep(0.06, 0.12, jump);
    }

    vec4 push_pull_inpaint(vec2 uv_coord, float center_depth_inv) {
        vec4 best_color = vec4(0.0);
        float best_weight = 0.0;
        int search_range = int(u_search_radius);
        // Sweep toward the side the background is revealed from (trig precomputed)
        vec2 sweep = g_par_dir * pixel_size.x * g_sweep_sign;
        
        // Phase 1: Directional sweep (most samples here)
        for (int i = 1; i <= search_range; i++) {
            vec2 sample_uv = uv_coord + sweep * float(i);
            if (sample_uv.x < 0.0 || sample_uv.y < 0.0 || sample_uv.x > 1.0 || sample_uv.y > 1.0) continue;
            float sample_depth_inv = 1.0 - texture(tex_depth, sample_uv).r;
            
            // Only accept background pixels (farther than current)
            if (sample_depth_inv > center_depth_inv + u_depth_tolerance) {
                vec4 sample_color = texture(tex_color, sample_uv);
                
                // Weight by: distance + depth similarity to other background
                float dist_weight = exp(-float(i) * 0.15);
                float depth_weight = 1.0 + (sample_depth_inv - center_depth_inv) * 10.0;
                float w = dist_weight * depth_weight;
                
                best_color += sample_color * w;
                best_weight += w;
                
                // Early exit if we found strong background
                if (best_weight > 5.0) break;
            }
        }

        // Phase 2: opposite sweep fallback if not enough background found
        if (best_weight < 2.0) {
            int opposite_range = search_range;
            for (int i = 1; i <= opposite_range; i++) {
                vec2 sample_uv = uv_coord - sweep * float(i);
                if (sample_uv.x < 0.0 || sample_uv.y < 0.0 || sample_uv.x > 1.0 || sample_uv.y > 1.0) continue;
                float sample_depth_inv = 1.0 - texture(tex_depth, sample_uv).r;
                if (sample_depth_inv > center_depth_inv + u_depth_tolerance) {
                    vec4 sample_color = texture(tex_color, sample_uv);
                    float w = exp(-float(i) * 0.2);
                    best_color += sample_color * w;
                    best_weight += w;
                }
            }
        }
        
        // Phase 3: Small vertical blur for smoothness (3 taps)
        if (best_weight > 0.01) {
            vec4 blurred = best_color / best_weight;
            vec4 vert_accum = blurred * 0.5;
            float vert_weight = 0.5;
            
            for (int dy = -1; dy <= 1; dy += 2) {
                vec2 vert_uv = uv_coord + vec2(0.0, dy * pixel_size.y * u_blur_radius);
                if (vert_uv.y >= 0.0 && vert_uv.y <= 1.0) {
                    float vert_depth_inv = 1.0 - texture(tex_depth, vert_uv).r;
                    if (vert_depth_inv > center_depth_inv + u_depth_tolerance * 0.5) {
                        float w = 0.25;
                        vert_accum += texture(tex_color, vert_uv) * w;
                        vert_weight += w;
                    }
                }
            }
            
            return vert_accum / vert_weight;
        }
        
        // Fallback: return original pixel if no background found
        return texture(tex_color, uv_coord);
    }

    // ALTERNATIVE: FAST SEPARABLE BLUR (Uncomment to use instead)
    vec4 separable_inpaint(vec2 uv_coord, float center_depth_inv) {
        vec4 accum = vec4(0.0);
        float total_w = 0.0;
        int R = 4;
        
        // Horizontal pass only (vertical would be a second shader pass)
        for (int x = -R; x <= R; ++x) {
            vec2 sample_uv = uv_coord + vec2(x * pixel_size.x, 0.0);
            
            if (sample_uv.x < 0.0 || sample_uv.x > 1.0) continue;
            
            float sample_depth_inv = 1.0 - texture(tex_depth, sample_uv).r;
            
            // Background filter
            float depth_ok = step(center_depth_inv + 0.015, sample_depth_inv);
            float w = exp(-float(x*x) * 0.25) * (1.0 + depth_ok * 10.0);
            
            accum += texture(tex_color, sample_uv) * w;
            total_w += w;
        }
        
        return total_w > 0.01 ? accum / total_w : texture(tex_color, uv_coord);
    }

    // MAIN
    void main() {
        vec2 flipped_uv = vec2(uv.x, 1.0 - uv.y);

        // Precompute parallax direction once (perf win: was recomputed per-loop)
        float c = cos(u_roll);
        float s = sin(u_roll);
        g_par_dir = vec2(c, s) * sign(u_eye_offset);
        g_sweep_sign = (u_eye_offset > 0.0) ? -1.0 : 1.0;

        // DIBR asymmetric depth smoothing: 3-tap Gaussian along parallax dir.
        // Per Fehn 2004, smooths sharp depth edges to narrow disocclusion width.
        vec2 ds_dir = g_par_dir * pixel_size * 1.5;
        float d0 = texture(tex_depth, flipped_uv).r;
        float dm = texture(tex_depth, flipped_uv - ds_dir).r;
        float dp = texture(tex_depth, flipped_uv + ds_dir).r;
        float depth = d0 * 0.7 + dm * 0.15 + dp * 0.15;
        float depth_inv = -depth;

        // Enhanced 3D: mild non-linear depth curve boosts perceived pop
        // for nearer objects without pushing parallax into artifact-heavy ranges.
        float depth_shaped = depth_inv * (1.0 + 0.35 * (1.0 - depth));

        // Calculate parallax shift with edge-aware border constraint.
        // Smoothly reduces parallax near left/right edges to prevent sampling
        // beyond image boundaries (standard DIBR border handling).
        float shift = (depth_shaped + u_convergence);
        float edge_margin = 0.02;
        float edge_falloff = smoothstep(0.0, edge_margin, flipped_uv.x)
                           * smoothstep(1.0, 1.0 - edge_margin, flipped_uv.x);
        float px = u_eye_offset * shift * u_depth_strength * edge_falloff;
        vec2 shifted_uv = flipped_uv - vec2(px * c, px * s);

        // Soft disocclusion: blend inpaint in by confidence to hide hard seams
        float conf = disocclusion_confidence(flipped_uv, shifted_uv);

        // Normal sampling
        vec4 color = texture(tex_color, shifted_uv);
        if (conf > 0.001) {
            // Disoccluded region: optimized inpainting, blended by confidence
            vec4 filled = push_pull_inpaint(flipped_uv, depth_inv);
            // Alternative: vec4 filled = separable_inpaint(flipped_uv, depth_inv);
            color = mix(color, filled, conf);
        }

        // Screen-edge alpha clip: keep the out-of-bounds safety net (alpha -> 0
        // if parallax somehow over-shoots into negative UV) but use a
        // sub-pixel fade band so the user does not see a visible soft border
        // between the desktop image and the screen edge. 
        vec2 border = smoothstep(-0.001, 0.001, shifted_uv) * smoothstep(1.001, 0.999, shifted_uv);
        color.a = min(border.x, border.y);
        frag_color = color;

        // Natural edge feathering
        if (u_feather_enabled == 1) {

            vec2 uv = (gl_FragCoord.xy - u_viewport.xy) / u_viewport.zw;

            // Distance to each edge
            float left   = uv.x;
            float right  = 1.0 - uv.x;
            float top    = uv.y;
            float bottom = 1.0 - uv.y;

            float feather = u_feather_width;

            // Smooth edge fades
            float fadeL = smoothstep(0.0, feather, left);
            float fadeR = smoothstep(0.0, feather, right);
            float fadeT = smoothstep(0.0, feather, top);
            float fadeB = smoothstep(0.0, feather, bottom);

            // Combine smoothly
            float falloff = fadeL * fadeR * fadeT * fadeB;

            // Perceptual shaping
            // <1 softer
            // >1 harder
            falloff = pow(falloff, 0.7);

            color.rgb *= falloff;
        }

        // Rounded corners via 2D SDF (screen-space uv, not shifted_uv)
        // Inigo Quilez rounded-box SDF
        float corner_r = u_corner_radius;
        vec2 d = abs(uv - 0.5) - 0.5 + corner_r;
        float corner_sdf = length(max(d, vec2(0.0))) + min(max(d.x, d.y), 0.0) - corner_r;

        // Mask out rounded corners
        float corner_alpha = 1.0 - smoothstep(0.0, 0.01, corner_sdf);
        color.a = min(color.a, corner_alpha);

        // Glow outside the screen edge only (sdf > 0 = outside the rounded rect).
        // Moved to dedicated glow quad in XR view.

        frag_color = color;
    }
"""

DEPTH_FRAGMENT = """
    #version 330
    in vec2 uv;
    out vec4 frag_color;
    uniform sampler2D tex_depth;
    
    // Optimized spectral-like colormap without normalization
    vec3 spectral_r_ultrafast(float t) {
        // Precomputed key colors for spectral_r approximation
        // Blue (far) -> Green -> Yellow -> Red (near)
        
        // Single branch - optimized for GPU parallelism
        vec3 color1 = vec3(0.0, 0.298, 0.651);      // Blue
        vec3 color2 = vec3(0.0, 0.5, 0.0);          // Green
        vec3 color3 = vec3(1.0, 0.851, 0.0);        // Yellow
        vec3 color4 = vec3(0.988, 0.0, 0.0);        // Red
        
        // Piecewise linear interpolation without conditional branches
        float w1 = max(0.0, 1.0 - abs(t - 0.125) * 4.0);
        float w2 = max(0.0, 1.0 - abs(t - 0.375) * 4.0);
        float w3 = max(0.0, 1.0 - abs(t - 0.625) * 4.0);
        float w4 = max(0.0, 1.0 - abs(t - 0.875) * 4.0);
        
        // Normalize weights
        float total = w1 + w2 + w3 + w4;
        if (total > 0.0) {
            w1 /= total; w2 /= total; w3 /= total; w4 /= total;
        }
        
        return color1 * w1 + color2 * w2 + color3 * w3 + color4 * w4;
    }

    void main() {
        vec2 flipped_uv = vec2(uv.x, 1.0 - uv.y);
        float depth = texture(tex_depth, flipped_uv).r;
        
        // Depth is already in [0,1] range, no normalization needed
        // Apply optimized spectral colormap directly
        vec3 color = spectral_r_ultrafast(depth);
        
        frag_color = vec4(color, 1.0);
    }
"""

# Anaglyph red-cyan composite shader: samples both eyes and blends.
ANAGLYPH_FRAGMENT = """
    #version 330
    in vec2 uv;
    out vec4 frag_color;
    uniform sampler2D tex_color;
    uniform sampler2D tex_depth;
    uniform vec2 u_resolution;
    uniform float u_eye_offset;
    uniform float u_depth_strength;
    uniform float u_convergence;
    uniform float u_roll;
    uniform float u_search_radius = 12.0;
    uniform float u_depth_tolerance = 0.012;
    uniform float u_blur_radius = 2.5;
    uniform int u_feather_enabled;
    uniform float u_feather_width;
    uniform vec4 u_viewport;
    uniform float u_corner_radius = 0.0;

    vec2 pixel_size = 1.0 / u_resolution;

    bool is_disoccluded(vec2 base_uv, vec2 shifted_uv, float center_depth, float eye_dir) {
        if (shifted_uv.x < 0.0 || shifted_uv.x > 1.0 ||
            shifted_uv.y < 0.0 || shifted_uv.y > 1.0)
            return true;
        vec2 grad_dir = vec2(cos(u_roll), sin(u_roll)) * eye_dir;
        float d_left  = texture(tex_depth, base_uv - grad_dir * pixel_size * 2.0).r;
        float d_right = texture(tex_depth, base_uv + grad_dir * pixel_size * 2.0).r;
        if (abs(d_left - d_right) > 0.08)
            return true;
        return false;
    }

    vec4 push_pull_inpaint(vec2 uv_coord, float center_depth_inv, float eye_dir) {
        vec4 best_color = vec4(0.0);
        float best_weight = 0.0;
        int search_range = int(u_search_radius);
        int search_dir = eye_dir > 0.0 ? -1 : 1;

        for (int i = 1; i <= search_range; i++) {
            float c = cos(u_roll);
            float s = sin(u_roll);
            vec2 sample_uv = uv_coord + vec2(search_dir * i * pixel_size.x * c, search_dir * i * pixel_size.x * s);
            if (sample_uv.x < 0.0 || sample_uv.y < 0.0 || sample_uv.x > 1.0 || sample_uv.y > 1.0) continue;
            float sample_depth_inv = 1.0 - texture(tex_depth, sample_uv).r;
            if (sample_depth_inv > center_depth_inv + u_depth_tolerance) {
                vec4 sample_color = texture(tex_color, sample_uv);
                float dist_weight = exp(-float(i) * 0.15);
                float depth_weight = 1.0 + (sample_depth_inv - center_depth_inv) * 10.0;
                float w = dist_weight * depth_weight;
                best_color += sample_color * w;
                best_weight += w;
                if (best_weight > 5.0) break;
            }
        }

        if (best_weight < 2.0) {
            int opposite_range = search_range;
            for (int i = 1; i <= opposite_range; i++) {
                float c = cos(u_roll);
                float s = sin(u_roll);
                vec2 sample_uv = uv_coord - vec2(search_dir * i * pixel_size.x * c, search_dir * i * pixel_size.x * s);
                if (sample_uv.x < 0.0 || sample_uv.y < 0.0 || sample_uv.x > 1.0 || sample_uv.y > 1.0) continue;
                float sample_depth_inv = 1.0 - texture(tex_depth, sample_uv).r;
                if (sample_depth_inv > center_depth_inv + u_depth_tolerance) {
                    vec4 sample_color = texture(tex_color, sample_uv);
                    float w = exp(-float(i) * 0.2);
                    best_color += sample_color * w;
                    best_weight += w;
                }
            }
        }

        if (best_weight > 0.01) {
            vec4 blurred = best_color / best_weight;
            vec4 vert_accum = blurred * 0.5;
            float vert_weight = 0.5;
            for (int dy = -1; dy <= 1; dy += 2) {
                vec2 vert_uv = uv_coord + vec2(0.0, dy * pixel_size.y * u_blur_radius);
                if (vert_uv.y >= 0.0 && vert_uv.y <= 1.0) {
                    float vert_depth_inv = 1.0 - texture(tex_depth, vert_uv).r;
                    if (vert_depth_inv > center_depth_inv + u_depth_tolerance * 0.5) {
                        float w = 0.25;
                        vert_accum += texture(tex_color, vert_uv) * w;
                        vert_weight += w;
                    }
                }
            }
            return vert_accum / vert_weight;
        }
        return texture(tex_color, uv_coord);
    }

    void main() {
        vec2 flipped_uv = vec2(uv.x, 1.0 - uv.y);
        float c = cos(u_roll);
        float s = sin(u_roll);
        // DIBR asymmetric depth smoothing along parallax direction
        vec2 ds_dir = vec2(c, s) * pixel_size * 1.5;
        float d0 = texture(tex_depth, flipped_uv).r;
        float dm = texture(tex_depth, flipped_uv - ds_dir).r;
        float dp = texture(tex_depth, flipped_uv + ds_dir).r;
        float depth = d0 * 0.7 + dm * 0.15 + dp * 0.15;
        float depth_inv = -depth;
        float shift_amount = (depth_inv + u_convergence) * u_depth_strength;
        // Edge-aware border constraint: both eyes rendered in one pass
        float edge_margin = 0.02;
        float edge_falloff = smoothstep(0.0, edge_margin, flipped_uv.x)
                           * smoothstep(1.0, 1.0 - edge_margin, flipped_uv.x);
        shift_amount *= edge_falloff;

        vec2 left_uv  = flipped_uv + vec2(u_eye_offset * shift_amount * c, u_eye_offset * shift_amount * s);
        vec2 right_uv = flipped_uv - vec2(u_eye_offset * shift_amount * c, u_eye_offset * shift_amount * s);

        vec4 left_color, right_color;
        if (is_disoccluded(flipped_uv, left_uv, depth, -1.0))
            left_color = push_pull_inpaint(flipped_uv, depth_inv, -1.0);
        else
            left_color = texture(tex_color, left_uv);

        if (is_disoccluded(flipped_uv, right_uv, depth, 1.0))
            right_color = push_pull_inpaint(flipped_uv, depth_inv, 1.0);
        else
            right_color = texture(tex_color, right_uv);

        frag_color = vec4(left_color.r, right_color.g, right_color.b, 1.0);

        vec2 border = smoothstep(0.0, 0.015, left_uv) * smoothstep(1.0, 0.985, left_uv);
        vec2 border_r = smoothstep(0.0, 0.015, right_uv) * smoothstep(1.0, 0.985, right_uv);
        frag_color.a = min(min(border.x, border.y), min(border_r.x, border_r.y));

        vec2 fuv = (gl_FragCoord.xy - u_viewport.xy) / u_viewport.zw;

        if (u_feather_enabled == 1) {
            float left   = fuv.x;
            float right  = 1.0 - fuv.x;
            float top    = fuv.y;
            float bottom = 1.0 - fuv.y;
            float feather = u_feather_width;
            float fadeL = smoothstep(0.0, feather, left);
            float fadeR = smoothstep(0.0, feather, right);
            float fadeT = smoothstep(0.0, feather, top);
            float fadeB = smoothstep(0.0, feather, bottom);
            float falloff = fadeL * fadeR * fadeT * fadeB;
            falloff = pow(falloff, 0.7);
            frag_color.rgb *= falloff;
        }

        float corner_r = u_corner_radius;
        vec2 d = abs(fuv - 0.5) - 0.5 + corner_r;
        float corner_sdf = length(max(d, vec2(0.0))) + min(max(d.x, d.y), 0.0) - corner_r;
        float corner_alpha = 1.0 - smoothstep(0.0, 0.01, corner_sdf);
        frag_color.a = min(frag_color.a, corner_alpha);
    }
"""

# Row-interleaved stereo shader: eye determined by row parity.
INTERLEAVED_FRAGMENT = """
    #version 330
    in vec2 uv;
    out vec4 frag_color;
    uniform sampler2D tex_color;
    uniform sampler2D tex_depth;
    uniform vec2 u_resolution;
    uniform float u_eye_offset;
    uniform float u_depth_strength;
    uniform float u_convergence;
    uniform float u_roll;
    uniform float u_search_radius = 12.0;
    uniform float u_depth_tolerance = 0.012;
    uniform float u_blur_radius = 2.5;
    uniform int u_feather_enabled;
    uniform float u_feather_width;
    uniform vec4 u_viewport;
    uniform float u_corner_radius = 0.0;

    vec2 pixel_size = 1.0 / u_resolution;
    float eye_dir;

    bool is_disoccluded(vec2 base_uv, vec2 shifted_uv, float center_depth) {
        if (shifted_uv.x < 0.0 || shifted_uv.x > 1.0 ||
            shifted_uv.y < 0.0 || shifted_uv.y > 1.0)
            return true;
        vec2 grad_dir = vec2(cos(u_roll), sin(u_roll)) * eye_dir;
        float d_left  = texture(tex_depth, base_uv - grad_dir * pixel_size * 2.0).r;
        float d_right = texture(tex_depth, base_uv + grad_dir * pixel_size * 2.0).r;
        if (abs(d_left - d_right) > 0.08)
            return true;
        return false;
    }

    vec4 push_pull_inpaint(vec2 uv_coord, float center_depth_inv) {
        vec4 best_color = vec4(0.0);
        float best_weight = 0.0;
        int search_range = int(u_search_radius);
        vec2 sweep = vec2(cos(u_roll), sin(u_roll)) * eye_dir;

        // Phase 1: Directional sweep (most samples here)
        for (int i = 1; i <= search_range; i++) {
            vec2 sample_uv = uv_coord + sweep * pixel_size.x * float(i);
            if (sample_uv.x < 0.0 || sample_uv.y < 0.0 || sample_uv.x > 1.0 || sample_uv.y > 1.0) continue;
            float sample_depth_inv = 1.0 - texture(tex_depth, sample_uv).r;
            if (sample_depth_inv > center_depth_inv + u_depth_tolerance) {
                vec4 sample_color = texture(tex_color, sample_uv);
                float dist_weight = exp(-float(i) * 0.15);
                float depth_weight = 1.0 + (sample_depth_inv - center_depth_inv) * 10.0;
                float w = dist_weight * depth_weight;
                best_color += sample_color * w;
                best_weight += w;
                if (best_weight > 5.0) break;
            }
        }

        // Phase 2: opposite sweep fallback if not enough background found
        if (best_weight < 2.0) {
            for (int i = 1; i <= search_range; i++) {
                vec2 sample_uv = uv_coord - sweep * pixel_size.x * float(i);
                if (sample_uv.x < 0.0 || sample_uv.y < 0.0 || sample_uv.x > 1.0 || sample_uv.y > 1.0) continue;
                float sample_depth_inv = 1.0 - texture(tex_depth, sample_uv).r;
                if (sample_depth_inv > center_depth_inv + u_depth_tolerance) {
                    vec4 sample_color = texture(tex_color, sample_uv);
                    float w = exp(-float(i) * 0.2);
                    best_color += sample_color * w;
                    best_weight += w;
                }
            }
        }

        // Phase 3: Small vertical blur for smoothness (3 taps)
        if (best_weight > 0.01) {
            vec4 blurred = best_color / best_weight;
            vec4 vert_accum = blurred * 0.5;
            float vert_weight = 0.5;

            for (int dy = -1; dy <= 1; dy += 2) {
                vec2 vert_uv = uv_coord + vec2(0.0, dy * pixel_size.y * u_blur_radius);
                if (vert_uv.y >= 0.0 && vert_uv.y <= 1.0) {
                    float vert_depth_inv = 1.0 - texture(tex_depth, vert_uv).r;
                    if (vert_depth_inv > center_depth_inv + u_depth_tolerance * 0.5) {
                        float w = 0.25;
                        vert_accum += texture(tex_color, vert_uv) * w;
                        vert_weight += w;
                    }
                }
            }

            return vert_accum / vert_weight;
        }

        // Fallback: return original pixel if no background found
        return texture(tex_color, uv_coord);
    }

    void main() {
        eye_dir = (int(mod(gl_FragCoord.y, 2.0)) == 0) ? -1.0 : 1.0;
        float my_offset = eye_dir * u_eye_offset;

        vec2 flipped_uv = vec2(uv.x, 1.0 - uv.y);

        // Precompute parallax direction once (perf win: was recomputed per-loop)
        float c = cos(u_roll);
        float s = sin(u_roll);
        vec2 parallax_dir = vec2(c, s) * eye_dir;

        // DIBR asymmetric depth smoothing: 3-tap Gaussian along parallax dir
        // Per Fehn 2004, smooths sharp depth edges to narrow disocclusion width
        vec2 ds_dir = parallax_dir * pixel_size * 1.5;
        float d0 = texture(tex_depth, flipped_uv).r;
        float dm = texture(tex_depth, flipped_uv - ds_dir).r;
        float dp = texture(tex_depth, flipped_uv + ds_dir).r;
        float depth = d0 * 0.7 + dm * 0.15 + dp * 0.15;
        float depth_inv = -depth;

        // Enhanced 3D: mild non-linear depth curve boosts perceived pop
        // for nearer objects without pushing parallax into artifact-heavy ranges
        float depth_shaped = depth_inv * (1.0 + 0.35 * (1.0 - depth));

        // Calculate parallax shift with edge-aware border constraint
        // Smoothly reduces parallax near left/right edges to prevent sampling
        // beyond image boundaries (standard DIBR border handling)
        float shift = (depth_shaped + u_convergence);
        float edge_margin = 0.02;
        float edge_falloff = smoothstep(0.0, edge_margin, flipped_uv.x)
                           * smoothstep(1.0, 1.0 - edge_margin, flipped_uv.x);
        float px = my_offset * shift * u_depth_strength * edge_falloff;
        vec2 shifted_uv = flipped_uv - vec2(px * c, px * s);

        // Soft disocclusion: blend inpaint in by confidence to hide hard seams
        float conf = 0.0;
        if (shifted_uv.x < 0.0 || shifted_uv.x > 1.0 ||
            shifted_uv.y < 0.0 || shifted_uv.y > 1.0) {
            conf = 1.0;
        } else {
            vec2 step2 = parallax_dir * pixel_size * 2.0;
            float d_left  = texture(tex_depth, flipped_uv - step2).r;
            float d_right = texture(tex_depth, flipped_uv + step2).r;
            float jump = abs(d_left - d_right);
            conf = smoothstep(0.06, 0.12, jump);
        }

        vec4 color;
        if (conf > 0.001) {
            color = push_pull_inpaint(flipped_uv, depth_inv);
        } else {
            color = texture(tex_color, shifted_uv);
        }

        // Screen-edge alpha clip: keep the out-of-bounds safety net (alpha -> 0
        // if parallax somehow over-shoots into negative UV) but use a
        // sub-pixel fade band so the user does not see a visible soft border
        vec2 border = smoothstep(-0.001, 0.001, shifted_uv) * smoothstep(1.001, 0.999, shifted_uv);
        color.a = min(border.x, border.y);
        frag_color = color;

        // Natural edge feathering
        if (u_feather_enabled == 1) {
            vec2 fuv = (gl_FragCoord.xy - u_viewport.xy) / u_viewport.zw;
            float left   = fuv.x;
            float right  = 1.0 - fuv.x;
            float top    = fuv.y;
            float bottom = 1.0 - fuv.y;
            float feather = u_feather_width;
            float fadeL = smoothstep(0.0, feather, left);
            float fadeR = smoothstep(0.0, feather, right);
            float fadeT = smoothstep(0.0, feather, top);
            float fadeB = smoothstep(0.0, feather, bottom);
            float falloff = fadeL * fadeR * fadeT * fadeB;
            falloff = pow(falloff, 0.7);
            frag_color.rgb *= falloff;
        }

        // Rounded corners via 2D SDF
        float corner_r = u_corner_radius;
        vec2 fuv = (gl_FragCoord.xy - u_viewport.xy) / u_viewport.zw;
        vec2 d = abs(fuv - 0.5) - 0.5 + corner_r;
        float corner_sdf = length(max(d, vec2(0.0))) + min(max(d.x, d.y), 0.0) - corner_r;
        float corner_alpha = 1.0 - smoothstep(0.0, 0.01, corner_sdf);
        frag_color.a = min(frag_color.a, corner_alpha);
    }
"""

# Interleaved-V: column-interleaved stereo (alternating columns per eye).
VERTICAL_INTERLEAVED_FRAGMENT = """
    #version 330
    in vec2 uv;
    out vec4 frag_color;
    uniform sampler2D tex_color;
    uniform sampler2D tex_depth;
    uniform vec2 u_resolution;
    uniform float u_eye_offset;
    uniform float u_depth_strength;
    uniform float u_convergence;
    uniform float u_roll;
    uniform float u_search_radius = 12.0;
    uniform float u_depth_tolerance = 0.012;
    uniform float u_blur_radius = 2.5;
    uniform int u_feather_enabled;
    uniform float u_feather_width;
    uniform vec4 u_viewport;
    uniform float u_corner_radius = 0.0;

    vec2 pixel_size = 1.0 / u_resolution;
    float eye_dir;

    bool is_disoccluded(vec2 base_uv, vec2 shifted_uv, float center_depth) {
        if (shifted_uv.x < 0.0 || shifted_uv.x > 1.0 ||
            shifted_uv.y < 0.0 || shifted_uv.y > 1.0)
            return true;
        vec2 grad_dir = vec2(cos(u_roll), sin(u_roll)) * eye_dir;
        float d_left  = texture(tex_depth, base_uv - grad_dir * pixel_size * 2.0).r;
        float d_right = texture(tex_depth, base_uv + grad_dir * pixel_size * 2.0).r;
        if (abs(d_left - d_right) > 0.08)
            return true;
        return false;
    }

    vec4 push_pull_inpaint(vec2 uv_coord, float center_depth_inv) {
        vec4 best_color = vec4(0.0);
        float best_weight = 0.0;
        int search_range = int(u_search_radius);
        int search_dir = eye_dir > 0.0 ? -1 : 1;

        for (int i = 1; i <= search_range; i++) {
            float c = cos(u_roll);
            float s = sin(u_roll);
            vec2 sample_uv = uv_coord + vec2(search_dir * i * pixel_size.x * c, search_dir * i * pixel_size.x * s);
            if (sample_uv.x < 0.0 || sample_uv.y < 0.0 || sample_uv.x > 1.0 || sample_uv.y > 1.0) continue;
            float sample_depth_inv = 1.0 - texture(tex_depth, sample_uv).r;
            if (sample_depth_inv > center_depth_inv + u_depth_tolerance) {
                vec4 sample_color = texture(tex_color, sample_uv);
                float dist_weight = exp(-float(i) * 0.15);
                float depth_weight = 1.0 + (sample_depth_inv - center_depth_inv) * 10.0;
                float w = dist_weight * depth_weight;
                best_color += sample_color * w;
                best_weight += w;
                if (best_weight > 5.0) break;
            }
        }

        if (best_weight < 2.0) {
            int opposite_range = search_range;
            for (int i = 1; i <= opposite_range; i++) {
                float c = cos(u_roll);
                float s = sin(u_roll);
                vec2 sample_uv = uv_coord - vec2(search_dir * i * pixel_size.x * c, search_dir * i * pixel_size.x * s);
                if (sample_uv.x < 0.0 || sample_uv.y < 0.0 || sample_uv.x > 1.0 || sample_uv.y > 1.0) continue;
                float sample_depth_inv = 1.0 - texture(tex_depth, sample_uv).r;
                if (sample_depth_inv > center_depth_inv + u_depth_tolerance) {
                    vec4 sample_color = texture(tex_color, sample_uv);
                    float w = exp(-float(i) * 0.2);
                    best_color += sample_color * w;
                    best_weight += w;
                }
            }
        }

        if (best_weight > 0.01) {
            vec4 blurred = best_color / best_weight;
            vec4 vert_accum = blurred * 0.5;
            float vert_weight = 0.5;
            for (int dy = -1; dy <= 1; dy += 2) {
                vec2 vert_uv = uv_coord + vec2(0.0, dy * pixel_size.y * u_blur_radius);
                if (vert_uv.y >= 0.0 && vert_uv.y <= 1.0) {
                    float vert_depth_inv = 1.0 - texture(tex_depth, vert_uv).r;
                    if (vert_depth_inv > center_depth_inv + u_depth_tolerance * 0.5) {
                        float w = 0.25;
                        vert_accum += texture(tex_color, vert_uv) * w;
                        vert_weight += w;
                    }
                }
            }
            return vert_accum / vert_weight;
        }
        return texture(tex_color, uv_coord);
    }

    void main() {
        // Interleaved-V: alternate columns by X coordinate (odd cols = left eye, even cols = right eye)
        eye_dir = (int(mod(gl_FragCoord.x, 2.0)) == 0) ? -1.0 : 1.0;
        float my_offset = eye_dir * u_eye_offset;

        vec2 flipped_uv = vec2(uv.x, 1.0 - uv.y);

        // DIBR asymmetric depth smoothing: 3-tap Gaussian along parallax dir
        // Per Fehn 2004, smooths sharp depth edges to narrow disocclusion width
        float c = cos(u_roll);
        float s = sin(u_roll);
        vec2 parallax_dir = vec2(c, s) * eye_dir;
        vec2 ds_dir = parallax_dir * pixel_size * 1.5;
        float d0 = texture(tex_depth, flipped_uv).r;
        float dm = texture(tex_depth, flipped_uv - ds_dir).r;
        float dp = texture(tex_depth, flipped_uv + ds_dir).r;
        float depth = d0 * 0.7 + dm * 0.15 + dp * 0.15;
        float depth_inv = -depth;

        // Enhanced 3D: mild non-linear depth curve boosts perceived pop
        // for nearer objects without pushing parallax into artifact-heavy ranges
        float depth_shaped = depth_inv * (1.0 + 0.35 * (1.0 - depth));

        // Calculate parallax shift with edge-aware border constraint
        // Smoothly reduces parallax near left/right edges to prevent sampling
        // beyond image boundaries (standard DIBR border handling)
        float shift = (depth_shaped + u_convergence);
        float edge_margin = 0.02;
        float edge_falloff = smoothstep(0.0, edge_margin, flipped_uv.x)
                           * smoothstep(1.0, 1.0 - edge_margin, flipped_uv.x);
        float px = my_offset * shift * u_depth_strength * edge_falloff;
        vec2 shifted_uv = flipped_uv - vec2(px * c, px * s);

        // Soft disocclusion confidence (smooth ramp, hides hard seams)
        float conf = 0.0;
        if (shifted_uv.x < 0.0 || shifted_uv.x > 1.0 ||
            shifted_uv.y < 0.0 || shifted_uv.y > 1.0) {
            conf = 1.0;
        } else {
            vec2 step2 = parallax_dir * pixel_size * 2.0;
            float d_left  = texture(tex_depth, flipped_uv - step2).r;
            float d_right = texture(tex_depth, flipped_uv + step2).r;
            float jump = abs(d_left - d_right);
            conf = smoothstep(0.06, 0.12, jump);
        }

        vec4 color;
        if (conf > 0.001) {
            color = push_pull_inpaint(flipped_uv, depth_inv);
        } else {
            color = texture(tex_color, shifted_uv);
        }

        // Screen-edge alpha clip
        vec2 border = smoothstep(-0.001, 0.001, shifted_uv) * smoothstep(1.001, 0.999, shifted_uv);
        color.a = min(border.x, border.y);
        frag_color = color;

        // Natural edge feathering
        if (u_feather_enabled == 1) {
            vec2 fuv = (gl_FragCoord.xy - u_viewport.xy) / u_viewport.zw;
            float left   = fuv.x;
            float right  = 1.0 - fuv.x;
            float top    = fuv.y;
            float bottom = 1.0 - fuv.y;
            float feather = u_feather_width;
            float fadeL = smoothstep(0.0, feather, left);
            float fadeR = smoothstep(0.0, feather, right);
            float fadeT = smoothstep(0.0, feather, top);
            float fadeB = smoothstep(0.0, feather, bottom);
            float falloff = fadeL * fadeR * fadeT * fadeB;
            falloff = pow(falloff, 0.7);
            frag_color.rgb *= falloff;
        }

        // Rounded corners via 2D SDF
        float corner_r = u_corner_radius;
        vec2 fuv = (gl_FragCoord.xy - u_viewport.xy) / u_viewport.zw;
        vec2 d = abs(fuv - 0.5) - 0.5 + corner_r;
        float corner_sdf = length(max(d, vec2(0.0))) + min(max(d.x, d.y), 0.0) - corner_r;
        float corner_alpha = 1.0 - smoothstep(0.0, 0.01, corner_sdf);
        frag_color.a = min(frag_color.a, corner_alpha);
    }
"""

def add_logo(window):
    """Optimized logo loading with lazy imports"""
    from PIL import Image
    glfw_img = Image.open("icon2.ico")  # Path to your icon file
    if OS_NAME != "Darwin":
        glfw_img = crop_icon(glfw_img)
        glfw.set_window_icon(window, 1, [glfw_img])


class OverlayTextureRenderer:
    """Small RGBA overlay rendered as a separate GL texture."""

    VERTEX_SHADER = """
        #version 330
        in vec2 in_position;
        in vec2 in_uv;
        out vec2 uv;
        void main() {
            uv = in_uv;
            gl_Position = vec4(in_position, 0.0, 1.0);
        }
    """

    FRAGMENT_SHADER = """
        #version 330
        in vec2 uv;
        out vec4 frag_color;
        uniform sampler2D u_overlay;
        void main() {
            frag_color = texture(u_overlay, uv);
        }
    """

    def __init__(self, ctx):
        self.ctx = ctx
        self.prog = ctx.program(
            vertex_shader=self.VERTEX_SHADER,
            fragment_shader=self.FRAGMENT_SHADER,
        )
        self.prog["u_overlay"].value = 0
        self.texture = None
        self.size = None
        self.vbo = None
        self.vao = None
        self._geometry_key = None

    def update_texture(self, rgba):
        if rgba is None:
            return False
        h, w = rgba.shape[:2]
        if w <= 0 or h <= 0:
            return False
        if self.texture is None or self.size != (w, h):
            if self.texture is not None:
                self.texture.release()
            self.texture = self.ctx.texture((w, h), 4, dtype="f1")
            self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.size = (w, h)
        self.texture.write(np.ascontiguousarray(rgba).tobytes())
        return True

    def clear(self):
        if self.texture is not None:
            try:
                self.texture.release()
            except Exception:
                pass
        self.texture = None
        self.size = None
        self._geometry_key = None

    def render(self, window_size, position):
        if self.texture is None or self.size is None:
            return
        win_w, win_h = window_size
        tex_w, tex_h = self.size
        if win_w <= 0 or win_h <= 0:
            return
        x, y = position
        left = (x / win_w) * 2.0 - 1.0
        right = ((x + tex_w) / win_w) * 2.0 - 1.0
        top = 1.0 - (y / win_h) * 2.0
        bottom = 1.0 - ((y + tex_h) / win_h) * 2.0
        geometry_key = (win_w, win_h, tex_w, tex_h, x, y)
        if self.vao is None or self._geometry_key != geometry_key:
            verts = np.array([
                left, bottom, 0.0, 1.0,
                right, bottom, 1.0, 1.0,
                left, top, 0.0, 0.0,
                right, top, 1.0, 0.0,
            ], dtype="f4")
            if self.vbo is None:
                self.vbo = self.ctx.buffer(verts.tobytes())
            else:
                self.vbo.orphan(verts.nbytes)
                self.vbo.write(verts.tobytes())
            if self.vao is None:
                self.vao = self.ctx.vertex_array(
                    self.prog, [(self.vbo, "2f 2f", "in_position", "in_uv")]
                )
            self._geometry_key = geometry_key

        previous_viewport = self.ctx.viewport
        self.ctx.viewport = (0, 0, win_w, win_h)
        self.texture.use(location=0)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.viewport = previous_viewport

    def release(self):
        for obj in (self.vao, self.vbo, self.texture, self.prog):
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
        self.vao = None
        self.vbo = None
        self.texture = None
        self.prog = None
        self.size = None

class StereoWindow:
    """Optimized stereo viewer with performance improvements"""

    def __init__(self, capture_mode="Monitor", monitor_index=0, ipd=0.064, depth_ratio=1.0, convergence=0.0, display_mode="Half-SBS", fill_16_9=True, show_fps=True, use_3d=False, fix_aspect=False, stream_mode=None, lossless_scaling=False, specify_display=False, stereo_display_index=0, feather_enabled=False, frame_size=(1280, 720), use_cuda=False, cuda_device_id=0, vsync=False, **kwargs):
        # Initialize with default values
        self._has_real_frame = False
        self.use_3d = use_3d
        self.title = "Stereo Viewer"
        self.capture_mode = capture_mode
        self.input_monitor_index = monitor_index
        self.ipd_uv = ipd
        self.depth_strength = 0.1
        self._last_window_position = None
        self._last_window_size = None
        self._fullscreen = False
        self.depth_ratio = depth_ratio
        self.depth_ratio_original = depth_ratio
        self._modes = ["Full-SBS", "Half-SBS", "Half-TAB", "Depth Map", "Full-TAB", "Anaglyph", "Interleaved", "Interleaved-V"]
        # Edge feathering toggle
        self.feather_enabled = feather_enabled
        self.feather_width = 0.02       # 2% of view width
        self.display_mode = display_mode
        self._texture_size = None
        self.fill_16_9 = fill_16_9
        self.frame_size = frame_size
        self.aspect = self.frame_size[0] / self.frame_size[1]
        self.fix_aspect = fix_aspect
        self.show_fps = show_fps
        self.vsync = vsync
        self.stream_mode = stream_mode
        self.window_size = self.frame_size
        self.convergence = convergence
        self.lossless_scaling = lossless_scaling

        # FPS and latency tracking variables, will be set externally
        self.actual_fps = 0.0
        self.total_latency = 0.0
        
        # Add PBO for streamer
        self._pbo_ids = None
        self._pbo_index = 0
        self._pbo_initialized = False
        
        # Add toggle state for depth map mode
        self.show_original_in_depth_mode = False
        
        # Cache for uniforms to avoid redundant updates
        self._last_eye_offset_set = 0.0
        self._last_depth_strength_set = 0.0
        
        # Depth ratio display variables
        self.last_depth_change_time = 0
        self.show_depth_ratio = False
        self.depth_display_duration = 2.0  # Show depth for 2 seconds after change
        
        # Font and text sizing
        self.font = None
        self.font_type = get_font_type()
        self.base_font_size = 60  # Base size for 1280x720 window
        self.current_font_size = self.base_font_size
        self.text_padding = 10
        self.text_spacing = 5

        # Mouse Passthrough
        self.last_mouse_toggle_time = 0
        self.show_mouse_state = False
        self.mouse_display_duration = 2.0

        # Cache for viewport calculations
        self._viewport_cache = {
            'win_size': (0, 0),
            'tex_size': (0, 0),
            'display_mode': None,
            'fill_16_9': None,
            'viewport': None,
            'show_original': None
        }

        # Overlay cache & throttle
        self.overlay_update_interval = 0.5  # seconds, throttle overlay regeneration
        self._overlay_cache = {
            'image': None,         # numpy RGBA image
            'overlay_rgb_f': None, # cached float32 RGB of overlay (avoids per-frame convert)
            'alpha_f': None,       # cached float32 alpha (0..1) of overlay
            'fps_text': None,
            'latency_text': None,
            'depth_text': None,
            'mouse_text': None,
            'last_update': 0.0,
            'values_update': 0.0,  # last time the displayed FPS/latency numbers refreshed
            'disp_fps': None,      # throttled FPS value actually shown
            'disp_latency': None,  # throttled latency value actually shown
            'pos': (self.text_padding, self.text_padding)
        }
        
        # Stereo Display Settings
        self.specify_display = specify_display
        self.stereo_display_index = stereo_display_index

        # Fullscreen mode
        self._fullscreen = False

        # Initialize GLFW
        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW")

        self.monitor_index = self.get_glfw_mon_index_mss(self.input_monitor_index)
        if self.specify_display:
            self.monitor_index = self.get_glfw_mon_index_mss(self.stereo_display_index)
        # Configure window
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, True)
        glfw.window_hint(glfw.DECORATED, glfw.TRUE) # window decoration

        if (self.use_3d or self.capture_mode == "Window" and self.specify_display) and self.input_monitor_index == self.stereo_display_index:
            glfw.window_hint(glfw.MOUSE_PASSTHROUGH, glfw.TRUE)  # clicks pass through
            glfw.window_hint(glfw.DECORATED, glfw.FALSE) # remove window decoration
            glfw.window_hint(glfw.FLOATING, glfw.TRUE)  # enable fullscreen
        elif self.stream_mode == "RTMP":
            glfw.window_hint(glfw.RESIZABLE, False)  # Disable resizing
        elif self.stream_mode == "MJPEG":
            glfw.window_hint(glfw.RESIZABLE, False)  # Disable resizing
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)  # Hide Window
        # Create window
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self.window = glfw.create_window(*self.window_size, self.title, None, None)
        add_logo(self.window)
        
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Could not create window")
        
        # Hide OS cursor inside GLFW window but DO NOT disable it globally
        glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_HIDDEN)

        # Set up OpenGL context
        glfw.make_context_current(self.window)
        self.ctx = moderngl.create_context()
        glfw.swap_interval(1 if self.vsync else 0)
        
        # Precompile shaders and create VAO
        self.prog = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER
        )
        self.prog['u_convergence'].value = self.convergence  # e.g. self.convergence = 0.5
        vertices = np.array([
            -1, -1, 0, 0,
            1, -1, 1, 0,
            -1,  1, 0, 1,
            1,  1, 1, 1,
        ], dtype='f4')
        self.vbo = self.ctx.buffer(vertices)
        self.quad_vao = self.ctx.vertex_array(
            self.prog, [(self.vbo, '2f 2f', 'in_position', 'in_uv')]
        )
        self.depth_prog = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=DEPTH_FRAGMENT
        )
        self.depth_vao = self.ctx.vertex_array(
            self.depth_prog, [(self.vbo, '2f 2f', 'in_position', 'in_uv')]
        )

        # Anaglyph shader program
        self.anaglyph_prog = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=ANAGLYPH_FRAGMENT
        )
        self.anaglyph_prog['u_convergence'].value = self.convergence
        self.anaglyph_vao = self.ctx.vertex_array(
            self.anaglyph_prog, [(self.vbo, '2f 2f', 'in_position', 'in_uv')]
        )

        # Interleaved (row-interleaved) shader program
        self.interleaved_prog = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=INTERLEAVED_FRAGMENT
        )
        self.interleaved_prog['u_convergence'].value = self.convergence
        self.interleaved_vao = self.ctx.vertex_array(
            self.interleaved_prog, [(self.vbo, '2f 2f', 'in_position', 'in_uv')]
        )

        # Interleaved-V shader program: columns alternate per eye.
        self.vertical_interleaved_prog = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=VERTICAL_INTERLEAVED_FRAGMENT
        )
        self.vertical_interleaved_prog['u_convergence'].value = self.convergence
        self.vertical_interleaved_vao = self.ctx.vertex_array(
            self.vertical_interleaved_prog, [(self.vbo, '2f 2f', 'in_position', 'in_uv')]
        )
        self.overlay_renderer = OverlayTextureRenderer(self.ctx)

        # Initialize textures as None
        self.color_tex = None
        self.depth_tex = None
        
        # Cache for display mode and original toggle state
        self._last_display_mode = None
        self._last_show_original = None
        
        # Set callbacks
        glfw.set_key_callback(self.window, self.on_key_event)
        glfw.set_window_size_callback(self.window, self._on_window_resize)
        
        # Load initial font
        self._update_font()
        vidmode = glfw.get_video_mode(glfw.get_monitors()[self.monitor_index])
        self.mon_w, self.mon_h = vidmode.size.width, vidmode.size.height

        self.use_cuda = use_cuda
        self.cuda_device_id = cuda_device_id
        self._cudart = None
        self._pbo_color = None        # PBO ID for colour texture
        self._pbo_depth = None        # PBO ID for depth texture
        self._cuda_resource_color = None
        self._cuda_resource_depth = None
        # Persistent page-locked (pinned) staging buffer for the colour upload.
        # Reused every frame to avoid per-frame device allocations and to make the
        # host->PBO copy fast (pinned memory has much higher H2D bandwidth).
        self._pinned_rgb = None
        self._pinned_rgb_ptr = None
        # Whether the active GPU is integrated (APU / unified memory). Selects the
        # fastest colour-upload strategy (see _upload_color). Set in _init_cuda_pbos.
        self._cuda_integrated = False

    def __del__(self):
        if hasattr(self, "overlay_renderer") and self.overlay_renderer is not None:
            self.overlay_renderer.release()
        self.cleanup_cuda()

    def _init_cuda_pbos(self, width, height):
        """Create PBOs and register them with CUDA. Disable CUDA on failure."""
        if not self.use_cuda or self._cudart is not None:
            return
        try:
            self._cudart = CUDART_GL(self.cuda_device_id)

            # Detect integrated GPUs (APUs with unified memory). On those there is
            # no PCIe bus, so a pinned Host->Device staging copy for the colour
            # frame is pure overhead (benchmarked slower than a direct texture
            # write). We therefore only use the pinned PBO path on discrete GPUs.
            self._cuda_integrated = False
            try:
                props = torch.cuda.get_device_properties(self.cuda_device_id)
                self._cuda_integrated = bool(getattr(props, "is_integrated", 0))
            except Exception:
                self._cuda_integrated = False

            # Colour PBO (RGB8, 3 bytes/pixel): only needed for the discrete-GPU
            # pinned upload path. Integrated GPUs upload colour via texture.write.
            self._pbo_color = glGenBuffers(1)
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self._pbo_color)
            glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 3, None, GL_STREAM_DRAW)
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
            self._cuda_resource_color = self._cudart.register_buffer(self._pbo_color)

            # Depth PBO (float32, 4 bytes/pixel). Depth always lives on the GPU,
            # so a device-to-device copy is optimal on every backend/GPU type.
            self._pbo_depth = glGenBuffers(1)
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self._pbo_depth)
            glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4, None, GL_STREAM_DRAW)
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
            self._cuda_resource_depth = self._cudart.register_buffer(self._pbo_depth)

            # Persistent pinned staging buffer for the (overlay-composited) colour
            # frame only on discrete GPUs, where pinned H2D bandwidth pays off.
            if not self._cuda_integrated:
                try:
                    self._pinned_rgb = torch.empty(
                        (height, width, 3), dtype=torch.uint8, pin_memory=True
                    )
                    self._pinned_rgb_ptr = self._pinned_rgb.data_ptr()
                except Exception:
                    # Pinned allocation can fail on some setups; fall back to no staging
                    self._pinned_rgb = None
                    self._pinned_rgb_ptr = None

            gpu_kind = "integrated" if self._cuda_integrated else "discrete"
            print(f"[Main] Enabled {BACKEND}-GL interop acceleration "
                  f"({BACKEND}: {self.cuda_device_id}, {gpu_kind} GPU)")
        except Exception as e:
            print(f"[Main] {BACKEND}-GL interop initialization failed: {e}")
            print(f"[Main] Falling back to CPU-GL interop.")
            self.use_cuda = False
            self.cleanup_cuda()   # clean up any partial allocations

    def _upload_color(self, rgb_host):
        """Adaptive colour upload for the overlay-composited host frame.

        - Discrete GPU: stage into a persistent pinned buffer and do a single fast
          Host->Device copy into the registered PBO (PCIe-optimal, no per-frame
          device allocation).
        - Integrated GPU (APU): there is no PCIe bus, so the pinned/PBO dance is
          pure overhead; a direct ModernGL texture write is as fast or faster.
        """
        # Integrated GPU (or no pinned staging / colour PBO): direct texture write.
        if self._cuda_integrated or self._pinned_rgb is None or self._cuda_resource_color is None:
            self.color_tex.write(np.ascontiguousarray(rgb_host).tobytes())
            return

        h, w, _ = rgb_host.shape
        nbytes = h * w * 3
        if self._pinned_rgb.shape[:2] == (h, w):
            # In-place CPU copy numpy -> pinned torch buffer
            self._pinned_rgb.copy_(torch.from_numpy(np.ascontiguousarray(rgb_host)))
            src_ptr = self._pinned_rgb_ptr
        else:
            # Size mismatch fallback: copy directly from the (pageable) numpy buffer
            rgb_c = np.ascontiguousarray(rgb_host)
            src_ptr = rgb_c.ctypes.data
            nbytes = rgb_c.nbytes

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self._pbo_color)
        ptr = self._cudart.map_resource(self._cuda_resource_color)
        try:
            self._cudart.memcpy_h2d(ptr, src_ptr, nbytes)
        finally:
            self._cudart.unmap_resource(self._cuda_resource_color)

        # Update ModernGL texture from the PBO
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.color_tex.glo)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h,
                        GL_RGB, GL_UNSIGNED_BYTE, None)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

    def _upload_color_cuda(self, rgb_gpu):
        """Upload a GPU tensor (H,W,3) uint8 to colour texture using PBO."""
        h, w, _ = rgb_gpu.shape
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self._pbo_color)
        ptr = self._cudart.map_resource(self._cuda_resource_color)
        try:
            self._cudart.memcpy_d2d(ptr, rgb_gpu.data_ptr(), rgb_gpu.nbytes)
        finally:
            self._cudart.unmap_resource(self._cuda_resource_color)

        # Update ModernGL texture from the PBO
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.color_tex.glo)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h,
                        GL_RGB, GL_UNSIGNED_BYTE, None)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

    def _upload_depth_cuda(self, depth_gpu):
        """Upload a GPU tensor (H,W) float32 to depth texture using PBO."""
        h, w = depth_gpu.shape
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self._pbo_depth)
        ptr = self._cudart.map_resource(self._cuda_resource_depth)
        try:
            self._cudart.memcpy_d2d(ptr, depth_gpu.data_ptr(), depth_gpu.nbytes)
        finally:
            self._cudart.unmap_resource(self._cuda_resource_depth)

        # Update ModernGL texture from the PBO
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.depth_tex.glo)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h,
                        GL_RED, GL_FLOAT, None)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

    def cleanup_cuda(self):
        """Release CUDA resources and delete PBOs."""
        if self._cudart:
            try:
                if self._cuda_resource_color:
                    self._cudart.unregister_resource(self._cuda_resource_color)
            except Exception:
                pass
            try:
                if self._cuda_resource_depth:
                    self._cudart.unregister_resource(self._cuda_resource_depth)
            except Exception:
                pass
            try:
                if self._pbo_color and bool(glDeleteBuffers):
                    glDeleteBuffers(1, [self._pbo_color])
            except Exception:
                pass
            try:
                if self._pbo_depth and bool(glDeleteBuffers):
                    glDeleteBuffers(1, [self._pbo_depth])
            except Exception:
                pass
            # Release the persistent pinned staging buffer
            self._pinned_rgb = None
            self._pinned_rgb_ptr = None
            self._cudart = None
            self._pbo_color = None
            self._pbo_depth = None
            self._cuda_resource_color = None
            self._cuda_resource_depth = None
    
    def _calculate_depth_map_viewport(self, win_w, win_h, tex_w, tex_h):
        """Cached viewport calculation for depth map mode"""
        cache_key = (win_w, win_h, tex_w, tex_h, self.display_mode, 
                     self.fill_16_9, self.show_original_in_depth_mode)
        
        # Check if we can reuse cached viewport
        if (self._viewport_cache['win_size'] == (win_w, win_h) and
            self._viewport_cache['tex_size'] == (tex_w, tex_h) and
            self._viewport_cache['display_mode'] == self.display_mode and
            self._viewport_cache['fill_16_9'] == self.fill_16_9 and
            self._viewport_cache['show_original'] == self.show_original_in_depth_mode):
            return self._viewport_cache['viewport']
        
        # Calculate viewport
        if self.fill_16_9:
            src_w, src_h = tex_w, tex_h
            max_w, max_h = win_w, win_h
            render_w, render_h = self._compute_render_size(max_w, max_h, src_w, src_h)
            center_x = win_w / 2.0
            center_y = win_h / 2.0
            viewport = (
                int(center_x - render_w / 2),
                int(center_y - render_h / 2),
                render_w, render_h
            )
        else:
            disp_w, disp_h = tex_w, tex_h
            target_aspect = disp_h / disp_w
            try:
                window_aspect = win_h / win_w
            except ZeroDivisionError:
                window_aspect = 9/16

            # Scale to fit window, preserving aspect ratio
            if window_aspect <= target_aspect:
                # Window is wider than content
                view_h = win_h
                view_w = int(view_h / target_aspect)
            else:
                # Window is taller than content
                view_w = win_w
                view_h = int(view_w * target_aspect)

            offset_x = (win_w - view_w) // 2
            offset_y = (win_h - view_h) // 2
            viewport = (offset_x, offset_y, view_w, view_h)
        
        # Update cache
        self._viewport_cache = {
            'win_size': (win_w, win_h),
            'tex_size': (tex_w, tex_h),
            'display_mode': self.display_mode,
            'fill_16_9': self.fill_16_9,
            'show_original': self.show_original_in_depth_mode,
            'viewport': viewport
        }
        
        return viewport
    
    def update_monitor_size(self):
        vidmode = glfw.get_video_mode(glfw.get_monitors()[self.monitor_index])
        self.mon_w, self.mon_h = vidmode.size.width, vidmode.size.height
    
    def apply_3d_settings(self):
        if self.monitor_index == self.get_glfw_mon_index_mss(self.input_monitor_index):
            if OS_NAME == "Windows" and self.capture_mode == "Monitor":
                hide_window_from_capture(self.window)
            glfw.set_window_attrib(self.window, glfw.FLOATING, glfw.TRUE)    # Always on top
        else:
            if OS_NAME == "Windows":
                show_window_in_capture(self.window)
            glfw.set_window_attrib(self.window, glfw.FLOATING, glfw.FALSE)    # Disable

    def get_glfw_mon_index_mss(self, mss_monitor_index=1):
        """
        Map an MSS monitor index (1-based) to a GLFW monitor handle.

        MSS provides monitors like:
            monitors[0] -> all monitors bounding box
            monitors[1] -> first display
            monitors[2] -> second display, etc.

        GLFW gives a list of monitor handles that we can position windows on.
        This function matches them by position and size.
        """
        try:
            import mss
        except ImportError:
            print("[StereoWindow] mss not installed; using default monitor.")
            return 0

        with mss.mss() as sct:
            mss_monitors = sct.monitors

        # MSS uses index 1-based (0 = virtual bounding box)
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
                # Found matching GLFW monitor
                return i

        print("[StereoWindow] No matching GLFW monitor found for MSS index, defaulting to primary.")
        return 0
    
    def _on_window_resize(self, window, width, height):
        """Handle window resize events"""
        self.window_size = (width, height)
        self._update_font()
        self._overlay_cache['image'] = None
        self._overlay_cache['fps_text'] = None

    def _update_font(self):
        """Update font size based on window dimensions"""
        # Calculate dynamic font size based on window height
        base_height = 1080  # Reference height
        scale_factor = min(self.frame_size[0] / base_height, 2.0)  # Cap scaling at 2x
        self.current_font_size = int(self.base_font_size * scale_factor * 0.8)  # Slightly smaller than linear scale
        
        try:
            self.font = ImageFont.truetype(self.font_type, self.current_font_size)
        except Exception:
            try:
                # Try default font (Pillow default)
                self.font = ImageFont.load_default()
            except Exception:
                self.font = None
        
        # Update padding and spacing based on font size
        self.text_padding = max(5, int(self.current_font_size * 0.4))
        self.text_spacing = max(2, int(self.current_font_size * 0.2))
        
        # Update overlay cache position in case padding changed
        self._overlay_cache['pos'] = (self.text_padding, self.text_padding)

    def _generate_overlay_image(self, fps_text, latency_text, depth_text, mouse_text):
        """Rasterize the small overlay to RGBA numpy array (transparent background)."""
        if self.font is None:
            return None

        # Compose lines
        lines = []
        if self.show_fps and fps_text:
            lines.append(fps_text)
        if self.show_fps and self.total_latency > 0:
            latency_text = f"Latency: {self.total_latency:.0f} ms"
            lines.append(latency_text)
        if self.show_depth_ratio and depth_text:
            lines.append(depth_text)
        if self.show_mouse_state and mouse_text:  # Add mouse state line
            lines.append(mouse_text)
        
        if not lines:
            return None

        # Estimate text size using PIL
        # small margin for padding/spacing
        padding = self.text_padding
        spacing = self.text_spacing

        # measure bounding boxes
        dummy_img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
        draw = ImageDraw.Draw(dummy_img)
        widths = []
        heights = []
        bboxes = []
        for line in lines:
            try:
                # textbbox if available provides better metrics
                bbox = draw.textbbox((0, 0), line, font=self.font)
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
            except Exception:
                w, h = draw.textsize(line, font=self.font)
            widths.append(w)
            heights.append(h)
            bboxes.append((w, h))
        overlay_w = max(widths) + padding * 2
        overlay_h = sum(heights) + spacing * (len(lines) - 1) + padding * 2

        # Create overlay image and draw text
        overlay_img = Image.new('RGBA', (overlay_w, overlay_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay_img)
        x = padding
        y = padding
        for i, line in enumerate(lines):
            # Assign specific colors based on line content
            if "FPS:" in line:
                color = (0, 255, 0, 255)        # Green for FPS
            elif "Latency:" in line:
                color = (0, 255, 255, 255)        # Blue for Latency
            elif "Depth:" in line:
                color = (255, 255, 255, 255)      # Cyan for Depth
            elif "Mouse:" in line:
                color = (255, 255, 0, 255)      # Yellow for Mouse
            else:
                color = (255, 255, 255, 255)    # White fallback for other text
            
            draw.text((x, y), line, font=self.font, fill=color)
            y += heights[i] + spacing

        overlay_arr = np.array(overlay_img, dtype=np.uint8)  # H x W x 4
        return overlay_arr

    def _add_overlay(self, rgb_frame):
        """Add FPS and depth ratio overlay to the frame with minimal allocations."""
        # Skip overlay for depth map mode
        if self.display_mode == "Depth Map":
            return rgb_frame
        
        # If nothing to show or no font available, do nothing fast
        if not (self.show_fps or self.show_depth_ratio or self.show_mouse_state) or self.font is None:
            return rgb_frame

        h, w, _ = rgb_frame.shape
        
        # freeze window size for rtmp streaming
        if self.display_mode == "Full-SBS":
            w = 2 * w
        self.frame_size = (w, h)
                
        # Depth ratio visibility check
        current_time = time.perf_counter()
        if current_time - self.last_depth_change_time < self.depth_display_duration:
            self.show_depth_ratio = True
        else:
            self.show_depth_ratio = False
        
        # Mouse state visibility check
        if current_time - self.last_mouse_toggle_time < self.mouse_display_duration:
            self.show_mouse_state = True
        else:
            self.show_mouse_state = False

        cache = self._overlay_cache

        # Throttle the *displayed* FPS/latency numbers. These change every frame
        # (e.g. "FPS: 59.8" -> "FPS: 60.1"), and the live values were previously
        # forcing a full PIL text re-rasterization on every single frame, which is
        # the main overlay-induced FPS drop. We only refresh the shown numbers a
        # couple of times per second; the frame's real FPS is unaffected.
        if (cache.get('disp_fps') is None or
                (current_time - cache.get('values_update', 0.0)) >= self.overlay_update_interval):
            cache['disp_fps'] = self.actual_fps
            cache['disp_latency'] = self.total_latency
            cache['values_update'] = current_time

        # Compose the strings to display (from the throttled snapshot values)
        fps_text = f"FPS: {cache['disp_fps']:.1f}" if self.show_fps else ""
        latency_text = f"Latency: {cache['disp_latency']:.1f} ms" if self.show_fps else ""
        depth_text = f"Depth: {self.depth_ratio:.1f}" if self.show_depth_ratio else ""
        mouse_text = f"Mouse: {'Pass' if self.mouse_pass_through else 'Normal'}" if self.show_mouse_state else ""

        # Decide whether to regenerate the rasterized overlay. Because the numbers
        # above are throttled, the text strings only change a few times per second,
        # so regeneration (the expensive part) is naturally rate-limited.
        needs_regen = False
        if cache['image'] is None:
            needs_regen = True
        elif (fps_text != cache.get('fps_text') or
              latency_text != cache.get('latency_text') or
              depth_text != cache.get('depth_text') or
              mouse_text != cache.get('mouse_text')):
            needs_regen = True

        if needs_regen:
            overlay_arr = self._generate_overlay_image(fps_text, latency_text, depth_text, mouse_text)
            cache['image'] = overlay_arr
            cache['fps_text'] = fps_text
            cache['latency_text'] = latency_text
            cache['depth_text'] = depth_text
            cache['mouse_text'] = mouse_text
            cache['last_update'] = current_time
            # Pre-convert the overlay to float32 once so the per-frame blend below
            # doesn't repeat the RGBA->float conversion every frame.
            if overlay_arr is not None:
                cache['alpha_f'] = overlay_arr[..., 3:4].astype(np.float32) / 255.0
                cache['overlay_rgb_f'] = overlay_arr[..., :3].astype(np.float32)
            else:
                cache['alpha_f'] = None
                cache['overlay_rgb_f'] = None

        overlay_arr = cache['image']
        if overlay_arr is None:
            return rgb_frame

        ov_h, ov_w = overlay_arr.shape[:2]
        pos_x, pos_y = cache.get('pos', (self.text_padding, self.text_padding))

        # Clip overlay to frame boundaries
        if pos_x >= w or pos_y >= h:
            return rgb_frame
        end_x = min(w, pos_x + ov_w)
        end_y = min(h, pos_y + ov_h)
        ov_w_clipped = end_x - pos_x
        ov_h_clipped = end_y - pos_y
        if ov_w_clipped <= 0 or ov_h_clipped <= 0:
            return rgb_frame

        # Use the cached float32 overlay (already converted at regen time)
        alpha = cache['alpha_f'][0:ov_h_clipped, 0:ov_w_clipped]
        overlay_rgb = cache['overlay_rgb_f'][0:ov_h_clipped, 0:ov_w_clipped]
        frame_region = rgb_frame[pos_y:end_y, pos_x:end_x]

        # Alpha blending: result = overlay.rgb * alpha + frame * (1 - alpha)
        blended = overlay_rgb * alpha + frame_region.astype(np.float32) * (1.0 - alpha)
        # write back blended region into original frame (as uint8)
        rgb_frame[pos_y:end_y, pos_x:end_x] = np.clip(blended, 0, 255).astype(np.uint8)

        return rgb_frame

    def _update_overlay_texture(self):
        """Refresh the small text overlay texture without touching the RGB frame."""
        if self.font is None:
            if self.overlay_renderer is not None:
                self.overlay_renderer.clear()
            return
        if not (self.show_fps or self.show_depth_ratio or self.show_mouse_state):
            if self.overlay_renderer is not None:
                self.overlay_renderer.clear()
            return

        current_time = time.perf_counter()
        self.show_depth_ratio = current_time - self.last_depth_change_time < self.depth_display_duration
        self.show_mouse_state = current_time - self.last_mouse_toggle_time < self.mouse_display_duration
        if not (self.show_fps or self.show_depth_ratio or self.show_mouse_state):
            if self.overlay_renderer is not None:
                self.overlay_renderer.clear()
            return

        cache = self._overlay_cache
        if (cache.get('disp_fps') is None or
                (current_time - cache.get('values_update', 0.0)) >= self.overlay_update_interval):
            cache['disp_fps'] = self.actual_fps
            cache['disp_latency'] = self.total_latency
            cache['values_update'] = current_time

        fps_text = f"FPS: {cache['disp_fps']:.1f}" if self.show_fps else ""
        latency_text = f"Latency: {cache['disp_latency']:.1f} ms" if self.show_fps else ""
        depth_text = f"Depth: {self.depth_ratio:.1f}" if self.show_depth_ratio else ""
        mouse_text = f"Mouse: {'Pass' if getattr(self, 'mouse_pass_through', False) else 'Normal'}" if self.show_mouse_state else ""

        if (fps_text == cache.get('fps_text') and
                latency_text == cache.get('latency_text') and
                depth_text == cache.get('depth_text') and
                mouse_text == cache.get('mouse_text') and
                cache.get('image') is not None):
            return

        overlay_arr = self._generate_overlay_image(fps_text, latency_text, depth_text, mouse_text)
        cache['image'] = overlay_arr
        cache['fps_text'] = fps_text
        cache['latency_text'] = latency_text
        cache['depth_text'] = depth_text
        cache['mouse_text'] = mouse_text
        cache['last_update'] = current_time
        if overlay_arr is not None and self.overlay_renderer is not None:
            self.overlay_renderer.update_texture(overlay_arr)
        elif self.overlay_renderer is not None:
            self.overlay_renderer.clear()

    def _render_overlay(self):
        if self.stream_mode is not None or self.overlay_renderer is None:
            return
        window_size = glfw.get_framebuffer_size(self.window)
        win_w, win_h = window_size
        base_x, base_y = self._overlay_cache.get(
            'pos', (self.text_padding, self.text_padding)
        )
        positions = [(base_x, base_y)]

        if self.display_mode == "Full-SBS" and self._texture_size:
            tex_w, tex_h = self._texture_size
            src_w, src_h = tex_w, tex_h
            max_w, max_h = win_w / 2.0, win_h
            render_w, render_h = self._compute_render_size(max_w, max_h, src_w, src_h)
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

        for position in positions:
            self.overlay_renderer.render(window_size, position)

    def position_on_monitor(self, monitor_index=0):
        """Optimized monitor positioning"""
        monitors = glfw.get_monitors()
        if monitor_index < len(monitors):
            monitor = monitors[monitor_index]
            mon_x, mon_y = glfw.get_monitor_pos(monitor)
            vidmode = glfw.get_video_mode(monitor)
            mon_w, mon_h = vidmode.size.width, vidmode.size.height
            if self.stream_mode == "RTMP" and OS_NAME=="Linux":
                x = mon_x + mon_w // 2
                y = mon_y + mon_h // 2
            else:
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
                

    def get_current_monitor(self):
        """Optimized monitor detection with early returns"""
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
            if (monitor_x <= window_center_x < monitor_x + vidmode.size.width and
                monitor_y <= window_center_y < monitor_y + vidmode.size.height):
                return monitor
        return monitors[0]

    def move_to_adjacent_monitor(self, direction):
        """Optimized monitor switching"""
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
        """Get the GLFW monitor index for a given monitor."""
        monitors = glfw.get_monitors()
        for idx, mon in enumerate(monitors):
            if mon == monitor:
                return idx
        return -1

    def toggle_fullscreen(self):
        """Optimized fullscreen toggle with reduced GLFW calls"""
        current_monitor = self.get_current_monitor()
        if not current_monitor:
            return

        if not self._fullscreen:
            if not self.use_3d and self.capture_mode == "Window" and self.get_glfw_mon_index_mss(self.input_monitor_index) == self.get_glfw_monitor_index(current_monitor):
                glfw.set_window_attrib(self.window, glfw.MOUSE_PASSTHROUGH, True)
                glfw.set_window_attrib(self.window, glfw.FLOATING, glfw.TRUE)
            if OS_NAME == "Darwin":
                send_ctrl_cmd_f() # MacOS full screen
            else:
                # Enter fullscreen
                self._last_window_position = glfw.get_window_pos(self.window)
                self._last_window_size = glfw.get_window_size(self.window)

                # Get monitor info
                mon_x, mon_y = glfw.get_monitor_pos(current_monitor)
                vidmode = glfw.get_video_mode(current_monitor)
                full_w, full_h = vidmode.size.width, vidmode.size.height

                # Make the window undecorated and floating
                glfw.set_window_attrib(self.window, glfw.DECORATED, glfw.FALSE)
                if self.fix_aspect:
                    monitor_aspect = full_w / full_h
                    if monitor_aspect > self.aspect:
                        # Monitor is wider than target aspect
                        new_h = full_h
                        new_w = int(new_h * self.aspect)

                    else:
                        # Screen is taller: fit by width.
                        new_w = full_w
                        new_h = int(full_w / self.aspect)

                    glfw.set_window_size(self.window, new_w, new_h)

                    # Center window on screen
                    center_x = mon_x + (full_w - new_w) // 2
                    center_y = mon_y + (full_h - new_h) // 2
                    glfw.set_window_pos(self.window, center_x, center_y)
                else:
                    glfw.set_window_size(self.window, full_w, full_h)
                    glfw.set_window_pos(self.window, mon_x, mon_y)
            self._fullscreen = True
        else:
            if OS_NAME == "Darwin":
                send_ctrl_cmd_f() # MacOS full screen
            else:
                # Exit fullscreen
                glfw.set_window_attrib(self.window, glfw.DECORATED, glfw.TRUE)
                glfw.set_window_attrib(self.window, glfw.FLOATING, glfw.FALSE)

                restore_w, restore_h = self._last_window_size or self.window_size
                restore_x, restore_y = self._last_window_position or (0, 0)
                
                if not self._last_window_position:
                    vidmode = glfw.get_video_mode(current_monitor)
                    mon_x, mon_y = glfw.get_monitor_pos(current_monitor)
                    restore_x = mon_x + (vidmode.size.width - restore_w) // 2
                    restore_y = mon_y + (vidmode.size.height - restore_h) // 2

                glfw.set_window_size(self.window, restore_w, restore_h)
                glfw.set_window_pos(self.window, restore_x, restore_y)
            self._fullscreen = False
            glfw.set_window_attrib(self.window, glfw.MOUSE_PASSTHROUGH, False)
        self.window_size = glfw.get_window_size(self.window)
    
    def on_key_event(self, window, key, scancode, action, mods):
        """Optimized key event handling, disable some keys for rtmp and 3d monitor"""
        if action == glfw.PRESS:
            if key == glfw.KEY_ENTER or key == glfw.KEY_SPACE:
                if self.stream_mode is None and not self.use_3d:
                    self.toggle_fullscreen()
            elif key == glfw.KEY_D:
                 # Toggle between depth map and original RGB when in Depth Map mode
                if self.display_mode == "Depth Map":
                    self.show_original_in_depth_mode = not self.show_original_in_depth_mode
                    # print(f"Depth Map Mode: {'Showing Original RGB' if self.show_original_in_depth_mode else 'Showing Depth Map'}")
            elif key == glfw.KEY_RIGHT:
                self.move_to_adjacent_monitor(+1)
            elif key == glfw.KEY_LEFT:
                self.move_to_adjacent_monitor(-1)
            elif key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
            elif key == glfw.KEY_DOWN:
                self.depth_ratio = max(0, self.depth_ratio - 0.5)
                self.last_depth_change_time = time.perf_counter()
            elif key == glfw.KEY_UP:
                self.depth_ratio = min(10, self.depth_ratio + 0.5)
                self.last_depth_change_time = time.perf_counter()
            elif key == glfw.KEY_0:
                self.depth_ratio = self.depth_ratio_original
                self.last_depth_change_time = time.perf_counter()
            elif key == glfw.KEY_TAB:
                idx = self._modes.index(self.display_mode)
                self.display_mode = self._modes[(idx + 1) % len(self._modes)]
            elif key == glfw.KEY_F:  # Add FPS toggle with F key
                self.show_fps = not self.show_fps
                # Force overlay regen when toggling show_fps
                self._overlay_cache['last_update'] = 0.0
            elif key == glfw.KEY_B:
                self.feather_enabled = not self.feather_enabled
                print(f"Edge feathering: {'ON' if self.feather_enabled else 'OFF'}")
            elif key == glfw.KEY_A:  # Toggle fill 16:0 with A key
                self.fill_16_9 = not self.fill_16_9
                # Force overlay regen to show aspect ratio status
                self._overlay_cache['last_update'] = 0.0
            elif key == glfw.KEY_L:  # Toggle viewer aspect ratio lock with L key
                self.fix_aspect = not self.fix_aspect
                # Force overlay regen to show aspect ratio status
                self._overlay_cache['last_update'] = 0.0
            elif key == glfw.KEY_M:
                # Toggle mouse pass-through mode
                current_state = glfw.get_window_attrib(self.window, glfw.MOUSE_PASSTHROUGH)
                new_state = not current_state
                glfw.set_window_attrib(self.window, glfw.MOUSE_PASSTHROUGH, new_state)
                self.mouse_pass_through = new_state
                
                # Record time for OSD display
                self.last_mouse_toggle_time = time.perf_counter()
                self.show_mouse_state = True

    def update_frame(self, rgb, depth, current_fps=None, current_latency=None):
        """Update frame textures. CUDA tensors stay on GPU when GL interop works."""
        if current_fps is not None:
            self.actual_fps = current_fps
        if current_latency is not None:
            self.total_latency = current_latency * 1000

        if hasattr(rgb, "detach"):
            rgb_shape = tuple(rgb.shape)
            if len(rgb_shape) == 3 and rgb_shape[0] in (3, 4):
                h, w = int(rgb_shape[1]), int(rgb_shape[2])
            else:
                h, w = int(rgb_shape[0]), int(rgb_shape[1])
        else:
            rgb_np = np.asarray(rgb)
            h, w = rgb_np.shape[:2]

        if self.stream_mode is None:
            self._update_overlay_texture()

        if self._texture_size != (w, h):
            if self.color_tex:
                self.color_tex.release()
            if self.depth_tex:
                self.depth_tex.release()
            self.color_tex = self.ctx.texture((w, h), 3, dtype='f1')
            self.depth_tex = self.ctx.texture((w, h), 1, dtype='f4')
            self.prog['tex_color'].value = 0
            self.prog['tex_depth'].value = 1
            self._texture_size = (w, h)

            # Reinit CUDA PBOs if needed (may disable CUDA on failure)
            if self.use_cuda:
                try:
                    self.cleanup_cuda()
                    self._init_cuda_pbos(w, h)
                except Exception as e:
                    print(f"[update_frame] Error initializing CUDA PBOs: {e}")
                    self.use_cuda = False

        # Try GPU-GL interop path first.
        cuda_success = False
        if self.stream_mode is None and self.use_cuda and self._cudart is not None:
            try:
                if not (hasattr(rgb, 'is_cuda') and rgb.is_cuda):
                    raise RuntimeError("RGB tensor not on GPU")
                if not (hasattr(depth, 'is_cuda') and depth.is_cuda):
                    raise RuntimeError("Depth tensor not on GPU")
                if self._cuda_resource_color is None or self._pbo_color is None:
                    raise RuntimeError("Colour PBO not available")

                rgb_gpu = rgb.detach()
                if rgb_gpu.ndim == 3 and rgb_gpu.shape[0] in (3, 4):
                    rgb_gpu = rgb_gpu[:3].permute(1, 2, 0)
                elif rgb_gpu.ndim == 3 and rgb_gpu.shape[-1] >= 3:
                    rgb_gpu = rgb_gpu[..., :3]
                else:
                    raise RuntimeError(f"Unsupported RGB tensor shape: {tuple(rgb_gpu.shape)}")
                rgb_gpu = rgb_gpu.contiguous().clamp(0, 255).to(torch.uint8)
                depth_gpu = depth.contiguous().float()

                torch.cuda.current_stream(self.cuda_device_id).synchronize()
                self._upload_color_cuda(rgb_gpu)
                self._upload_depth_cuda(depth_gpu)
                cuda_success = True
            except Exception as e:
                print(f"[update_frame] CUDA-GL upload disabled: {e}")
                self.use_cuda = False
                self.cleanup_cuda()

        # Fallback to CPU path if CUDA failed or not available
        if not cuda_success:
            if hasattr(depth, 'detach'):
                depth_np = depth.detach().cpu().contiguous().float().numpy()
            else:
                depth_np = np.asarray(depth, dtype=np.float32)

            if hasattr(rgb, 'detach'):
                try:
                    rgb_np = rgb.cpu().detach().contiguous().permute(1, 2, 0).clamp(0, 255).to(torch.uint8).numpy()
                except RuntimeError as e:
                    print(f"[update_frame] RuntimeError converting RGB tensor to numpy: {e}")
                    return
                except Exception as e:
                    print(f"[update_frame] Error converting RGB tensor to numpy: {e}")
                    return
            else:
                rgb_np = np.asarray(rgb)

            if self.stream_mode is not None and self.display_mode != "Depth Map":
                rgb_np = self._add_overlay(rgb_np)

            if self.display_mode == "Depth Map" and not self.show_original_in_depth_mode:
                self.depth_tex.write(depth_np.tobytes())
                if self._last_display_mode != "Depth Map" or self._last_show_original != self.show_original_in_depth_mode:
                    self.color_tex.write(rgb_np.astype('uint8', copy=False).tobytes())
            else:
                self.color_tex.write(rgb_np.astype('uint8', copy=False).tobytes())
                self.depth_tex.write(depth_np.tobytes())

        # Cache current state
        self._last_display_mode = self.display_mode
        self._last_show_original = self.show_original_in_depth_mode

        # Show window after first frame.
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
                if OS_NAME != "Darwin":
                    if not self.lossless_scaling:
                        glfw.set_window_opacity(self.window, 0.0)
                    glfw.set_window_attrib(self.window, glfw.DECORATED, glfw.FALSE)
                else:
                    self.fix_aspect = True
                    if (self._texture_size == (self.mon_w, self.mon_h) and self.display_mode != "Full-SBS") or (self._texture_size == (self.mon_w // 2, self.mon_h) and self.display_mode == "Full-SBS"):
                        glfw.set_window_attrib(self.window, glfw.RESIZABLE, True)
                        glfw.set_window_attrib(self.window, glfw.DECORATED, glfw.FALSE)
                        self.toggle_fullscreen()
            if self.specify_display:
                if not self.stream_mode == "RTMP" or self.lossless_scaling:
                    self.toggle_fullscreen()

    def _compute_render_size(self, max_w, max_h, src_w, src_h):
        """Calculate render size maintaining aspect ratio"""
        if src_w == 0 or src_h == 0:
            return 0, 0
        scale = min(max_w / src_w, max_h / src_h)
        return (
            max(1, int(round(src_w * scale))),
            max(1, int(round(src_h * scale)))
        )
        
    def _init_pbos(self, width, height):
        """Initialize two PBOs for asynchronous pixel transfers."""
        if self._pbo_initialized:
            return
        self._pbo_ids = glGenBuffers(2)
        buffer_size = width * height * 3  # RGB8
        for pbo in self._pbo_ids:
            glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo)
            glBufferData(GL_PIXEL_PACK_BUFFER, buffer_size, None, GL_STREAM_READ)
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
        self._pbo_index = 0
        self._pbo_initialized = True
        # print(f"[StereoWindow] Initialized {len(self._pbo_ids)} PBOs ({buffer_size/1e6:.2f} MB each)")

    def capture_glfw_image(self):
        """Asynchronous GPU readback using double-buffered PBOs."""
        width, height = glfw.get_framebuffer_size(self.window)
        if not self._pbo_initialized:
            self._init_pbos(width, height)
        next_index = (self._pbo_index + 1) % 2
        glPixelStorei(GL_PACK_ALIGNMENT, 1)

        # Bind current PBO and start async read
        glBindBuffer(GL_PIXEL_PACK_BUFFER, self._pbo_ids[self._pbo_index])
        glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, ctypes.c_void_p(0))

        # Bind previous PBO to map and read CPU data (async from previous frame)
        glBindBuffer(GL_PIXEL_PACK_BUFFER, self._pbo_ids[next_index])
        ptr = glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY)
        image = None
        if ptr:
            try:
                # Copy from GPU memory into numpy array
                buf = (GLubyte * (width * height * 3)).from_address(int(ptr))
                image = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
                image = np.flipud(image.copy())  # Flip vertically; .copy() detaches from mapped memory
            finally:
                glUnmapBuffer(GL_PIXEL_PACK_BUFFER)

        # Unbind PBO
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
        # Swap PBO indices for next frame
        self._pbo_index = next_index

        # Note: the very first call will return None (no previous frame ready)
        return image

    def render(self):
        """Ultra-optimized rendering with minimal GL calls"""
        if not self.color_tex or not self.depth_tex:
            return

        # Get window dimensions once
        win_w, win_h = glfw.get_framebuffer_size(self.window)
        tex_w, tex_h = self._texture_size
        
        # Early exit for depth map mode - fully optimized path
        if self.display_mode == "Depth Map":
            # Fast path - minimal operations for depth map mode
            
            # Skip aspect ratio updates for depth map mode if not needed
            if self.fix_aspect:
                glfw.set_window_aspect_ratio(self.window, tex_w, tex_h)
            
            # Clear screen with optimized clear color
            self.ctx.clear(0.0, 0.0, 0.0)  # Black background for better contrast
            
            # Use cached viewport calculation
            viewport = self._calculate_depth_map_viewport(win_w, win_h, tex_w, tex_h)
            self.ctx.viewport = viewport
            
            # Toggle between depth map and original RGB with minimal state changes
            if self.show_original_in_depth_mode:
                # Fast RGB rendering path
                self.color_tex.use(location=0)
                
                # Update uniforms only if changed
                if self._last_eye_offset_set != 0.0:
                    self.prog['u_eye_offset'].value = 0.0
                    self._last_eye_offset_set = 0.0
                
                if self._last_depth_strength_set != 0.0:
                    self.prog['u_depth_strength'].value = 0.0
                    self._last_depth_strength_set = 0.0
                
                # Direct render call without extra state changes
                self.quad_vao.render(moderngl.TRIANGLE_STRIP)
            else:
                # Ultra-fast depth map rendering
                self.depth_tex.use(location=0)
                
                # Direct render call - no uniform updates needed for depth shader
                self.depth_vao.render(moderngl.TRIANGLE_STRIP)
            
            # Skip swap buffers if frame hasn't changed (for headless/streaming modes)
            self._render_overlay()
            if self.stream_mode is None:
                glfw.poll_events()
            return

        # Composite display modes (Mono, Anaglyph, Interleaved, Interleaved-V)
        if self.display_mode in ["Anaglyph", "Interleaved", "Interleaved-V"]:
            if self.fix_aspect:
                glfw.set_window_aspect_ratio(self.window, tex_w, tex_h)
            else:
                glfw.set_window_aspect_ratio(self.window, glfw.DONT_CARE, glfw.DONT_CARE)

            self.ctx.clear(0.0, 0.0, 0.0)

            if self.fill_16_9:
                render_w, render_h = self._compute_render_size(win_w, win_h, tex_w, tex_h)
                center_x, center_y = win_w / 2.0, win_h / 2.0
                viewport = (int(center_x - render_w / 2), int(center_y - render_h / 2), render_w, render_h)
            else:
                target_aspect = tex_h / tex_w
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
                viewport = (offset_x, offset_y, view_w, view_h)

            self.ctx.viewport = viewport
            half_ipd = self.ipd_uv / 2.0

            if self.display_mode == "Anaglyph":
                self.color_tex.use(location=0)
                self.depth_tex.use(location=1)
                self.anaglyph_prog['u_eye_offset'].value = half_ipd
                self.anaglyph_prog['u_depth_strength'].value = self.depth_strength * self.depth_ratio
                self.anaglyph_prog['u_feather_enabled'].value = self.feather_enabled
                self.anaglyph_prog['u_feather_width'].value = self.feather_width
                self.anaglyph_prog['u_viewport'].value = viewport
                self.anaglyph_vao.render(moderngl.TRIANGLE_STRIP)
            elif self.display_mode == "Interleaved":
                self.color_tex.use(location=0)
                self.depth_tex.use(location=1)
                self.interleaved_prog['u_eye_offset'].value = half_ipd
                self.interleaved_prog['u_depth_strength'].value = self.depth_strength * self.depth_ratio
                self.interleaved_prog['u_feather_enabled'].value = self.feather_enabled
                self.interleaved_prog['u_feather_width'].value = self.feather_width
                self.interleaved_prog['u_viewport'].value = viewport
                self.interleaved_vao.render(moderngl.TRIANGLE_STRIP)
            elif self.display_mode == "Interleaved-V":
                self.color_tex.use(location=0)
                self.depth_tex.use(location=1)
                self.vertical_interleaved_prog['u_eye_offset'].value = half_ipd
                self.vertical_interleaved_prog['u_depth_strength'].value = self.depth_strength * self.depth_ratio
                self.vertical_interleaved_prog['u_feather_enabled'].value = self.feather_enabled
                self.vertical_interleaved_prog['u_feather_width'].value = self.feather_width
                self.vertical_interleaved_prog['u_viewport'].value = viewport
                self.vertical_interleaved_vao.render(moderngl.TRIANGLE_STRIP)

            self._render_overlay()
            if self.stream_mode is None:
                glfw.poll_events()
            return

        # Rest of the rendering for other modes...
        if self.fix_aspect:
            if self.display_mode == "Full-SBS":
                glfw.set_window_aspect_ratio(self.window, 2*tex_w, tex_h)
            elif self.display_mode == "Full-TAB":
                glfw.set_window_aspect_ratio(self.window, tex_w, 2*tex_h)
            else:
                glfw.set_window_aspect_ratio(self.window, tex_w, tex_h)
        else:
            glfw.set_window_aspect_ratio(self.window, glfw.DONT_CARE, glfw.DONT_CARE)
        
        # Clear screen once
        self.ctx.clear(0.0, 0.0, 0.0)
        
        # Handle other display modes (non-depth map)
        if self.fill_16_9:
            if self.display_mode in ["Full-SBS", "Half-SBS", "Half-TAB", "Full-TAB"]:
                self.color_tex.use(location=0)
                self.depth_tex.use(location=1)
                self.prog['u_depth_strength'].value = self.depth_strength * self.depth_ratio

                if self.display_mode == "Full-SBS":
                    src_w, src_h = tex_w, tex_h
                    max_w, max_h = win_w / 2.0, win_h
                    render_w, render_h = self._compute_render_size(max_w, max_h, src_w, src_h)
                    center_y = win_h / 2.0

                    # Left view
                    left_vp = (
                        int(win_w / 4.0 - render_w / 2),
                        int(center_y - render_h / 2),
                        render_w, render_h
                    )
                    self.ctx.viewport = left_vp
                    self.prog['u_eye_offset'].value = -self.ipd_uv / 2.0
                    self.prog['u_feather_enabled'].value = self.feather_enabled
                    self.prog['u_feather_width'].value = self.feather_width
                    self.prog['u_viewport'].value = left_vp
                    self.quad_vao.render(moderngl.TRIANGLE_STRIP)

                    # Right view
                    right_vp = (
                        int(3 * win_w / 4.0 - render_w / 2),
                        int(center_y - render_h / 2),
                        render_w, render_h
                    )
                    self.ctx.viewport = right_vp
                    self.prog['u_eye_offset'].value = self.ipd_uv / 2.0
                    self.prog['u_feather_enabled'].value = self.feather_enabled
                    self.prog['u_feather_width'].value = self.feather_width
                    self.prog['u_viewport'].value = right_vp
                    self.quad_vao.render(moderngl.TRIANGLE_STRIP)

                elif self.display_mode == "Half-SBS":
                    src_w, src_h = tex_w / 2.0, tex_h
                    max_w, max_h = win_w / 2.0, win_h
                    render_w, render_h = self._compute_render_size(max_w, max_h, src_w, src_h)
                    center_y = win_h / 2.0

                    # Left view
                    left_vp = (
                        int(win_w / 4.0 - render_w / 2),
                        int(center_y - render_h / 2),
                        render_w, render_h
                    )
                    self.ctx.viewport = left_vp
                    self.prog['u_eye_offset'].value = -self.ipd_uv / 2.0
                    self.prog['u_feather_enabled'].value = self.feather_enabled
                    self.prog['u_feather_width'].value = self.feather_width
                    self.prog['u_viewport'].value = left_vp
                    self.quad_vao.render(moderngl.TRIANGLE_STRIP)

                    # Right view
                    right_vp = (
                        int(3 * win_w / 4.0 - render_w / 2),
                        int(center_y - render_h / 2),
                        render_w, render_h
                    )
                    self.ctx.viewport = right_vp
                    self.prog['u_eye_offset'].value = self.ipd_uv / 2.0
                    self.prog['u_feather_enabled'].value = self.feather_enabled
                    self.prog['u_feather_width'].value = self.feather_width
                    self.prog['u_viewport'].value = right_vp
                    self._last_eye_offset_set = self.ipd_uv / 2.0
                    self.quad_vao.render(moderngl.TRIANGLE_STRIP)

                elif self.display_mode in ["Half-TAB"]:
                    src_w, src_h = tex_w, tex_h / 2.0
                    max_w, max_h = win_w, win_h / 2.0
                    render_w, render_h = self._compute_render_size(max_w, max_h, src_w, src_h)

                    # Top view (left eye)
                    top_vp = (
                        int(win_w / 2.0 - render_w / 2),
                        int(win_h / 4.0 - render_h / 2),
                        render_w, render_h
                    )
                    self.ctx.viewport = top_vp
                    self.prog['u_eye_offset'].value = -self.ipd_uv / 2.0
                    self.prog['u_feather_enabled'].value = self.feather_enabled
                    self.prog['u_feather_width'].value = self.feather_width
                    self.prog['u_viewport'].value = top_vp
                    self.quad_vao.render(moderngl.TRIANGLE_STRIP)

                    # Bottom view (right eye)
                    bottom_vp = (
                        int(win_w / 2.0 - render_w / 2),
                        int(3 * win_h / 4.0 - render_h / 2),
                        render_w, render_h
                    )
                    self.ctx.viewport = bottom_vp
                    self.prog['u_eye_offset'].value = self.ipd_uv / 2.0
                    self.prog['u_feather_enabled'].value = self.feather_enabled
                    self.prog['u_feather_width'].value = self.feather_width
                    self.prog['u_viewport'].value = bottom_vp
                    self.quad_vao.render(moderngl.TRIANGLE_STRIP)

                elif self.display_mode == "Full-TAB":
                    src_w, src_h = tex_w, tex_h
                    max_w, max_h = win_w, win_h / 2.0
                    render_w, render_h = self._compute_render_size(max_w, max_h, src_w, src_h)

                    # Top view (left eye)
                    top_vp = (
                        int(win_w / 2.0 - render_w / 2),
                        int(win_h / 4.0 - render_h / 2),
                        render_w, render_h
                    )
                    self.ctx.viewport = top_vp
                    self.prog['u_eye_offset'].value = -self.ipd_uv / 2.0
                    self.prog['u_feather_enabled'].value = self.feather_enabled
                    self.prog['u_feather_width'].value = self.feather_width
                    self.prog['u_viewport'].value = top_vp
                    self.quad_vao.render(moderngl.TRIANGLE_STRIP)

                    # Bottom view (right eye)
                    bottom_vp = (
                        int(win_w / 2.0 - render_w / 2),
                        int(3 * win_h / 4.0 - render_h / 2),
                        render_w, render_h
                    )
                    self.ctx.viewport = bottom_vp
                    self.prog['u_eye_offset'].value = self.ipd_uv / 2.0
                    self.prog['u_feather_enabled'].value = self.feather_enabled
                    self.prog['u_feather_width'].value = self.feather_width
                    self.prog['u_viewport'].value = bottom_vp
                    self.quad_vao.render(moderngl.TRIANGLE_STRIP)

        else:
            # Determine effective stereo frame size by display mode
            if self.display_mode == "Full-SBS":
                disp_w, disp_h = 2 * tex_w, tex_h
            elif self.display_mode == "Half-SBS":
                disp_w, disp_h = tex_w, tex_h
            elif self.display_mode in ["Half-TAB"]:
                disp_w, disp_h = tex_w, tex_h
            elif self.display_mode == "Full-TAB":
                disp_w, disp_h = tex_w, 2 * tex_h
            elif self.display_mode == "Depth Map":
                disp_w, disp_h = tex_w, tex_h
            else:
                disp_w, disp_h = 2 * tex_w, tex_h  # default full SBS

            target_aspect = disp_h / disp_w
            try:
                window_aspect = win_h / win_w
            except ZeroDivisionError:
                window_aspect = 9/16

            # Scale to fit window, preserving aspect ratio
            if window_aspect <= target_aspect:
                # Window is wider than content
                view_h = win_h
                view_w = int(view_h / target_aspect)
            else:
                # Window is taller than content
                view_w = win_w
                view_h = int(view_w * target_aspect)

            offset_x = (win_w - view_w) // 2
            offset_y = (win_h - view_h) // 2

            if self.display_mode in ["Full-SBS", "Half-SBS", "Half-TAB", "Full-TAB"]:
                self.color_tex.use(0)
                self.depth_tex.use(1)
                self.prog['u_depth_strength'].value = self.depth_strength * self.depth_ratio

                if self.display_mode == "Full-SBS":
                    # Left eye
                    left_vp = (offset_x, offset_y, view_w // 2, view_h)
                    self.ctx.viewport = left_vp
                    self.prog['u_eye_offset'].value = -self.ipd_uv / 2.0
                    self.prog['u_feather_enabled'].value = self.feather_enabled
                    self.prog['u_feather_width'].value = self.feather_width
                    self.prog['u_viewport'].value = left_vp
                    self.quad_vao.render(moderngl.TRIANGLE_STRIP)

                    # Right eye
                    right_vp = (offset_x + view_w // 2, offset_y, view_w // 2, view_h)
                    self.ctx.viewport = right_vp
                    self.prog['u_eye_offset'].value = self.ipd_uv / 2.0
                    self.prog['u_feather_enabled'].value = self.feather_enabled
                    self.prog['u_feather_width'].value = self.feather_width
                    self.prog['u_viewport'].value = right_vp
                    self.quad_vao.render(moderngl.TRIANGLE_STRIP)

                elif self.display_mode == "Half-SBS":
                    # Left eye
                    left_vp = (offset_x, offset_y, view_w // 2, view_h)
                    self.ctx.viewport = left_vp
                    self.prog['u_eye_offset'].value = -self.ipd_uv / 2.0
                    self.prog['u_feather_enabled'].value = self.feather_enabled
                    self.prog['u_feather_width'].value = self.feather_width
                    self.prog['u_viewport'].value = left_vp
                    self.quad_vao.render(moderngl.TRIANGLE_STRIP)

                    # Right eye
                    right_vp = (offset_x + view_w // 2, offset_y, view_w // 2, view_h)
                    self.ctx.viewport = right_vp
                    self.prog['u_eye_offset'].value = self.ipd_uv / 2.0
                    self.prog['u_feather_enabled'].value = self.feather_enabled
                    self.prog['u_feather_width'].value = self.feather_width
                    self.prog['u_viewport'].value = right_vp
                    self.quad_vao.render(moderngl.TRIANGLE_STRIP)

                elif self.display_mode == "Half-TAB":
                    # Top eye (left)
                    top_vp = (offset_x, offset_y + view_h // 2, view_w, view_h // 2)
                    self.ctx.viewport = top_vp
                    self.prog['u_eye_offset'].value = -self.ipd_uv / 2.0
                    self.prog['u_feather_enabled'].value = self.feather_enabled
                    self.prog['u_feather_width'].value = self.feather_width
                    self.prog['u_viewport'].value = top_vp
                    self.quad_vao.render(moderngl.TRIANGLE_STRIP)

                    # Bottom eye (right)
                    bottom_vp = (offset_x, offset_y, view_w, view_h // 2)
                    self.ctx.viewport = bottom_vp
                    self.prog['u_eye_offset'].value = self.ipd_uv / 2.0
                    self.prog['u_feather_enabled'].value = self.feather_enabled
                    self.prog['u_feather_width'].value = self.feather_width
                    self.prog['u_viewport'].value = bottom_vp
                    self.quad_vao.render(moderngl.TRIANGLE_STRIP)

                elif self.display_mode == "Full-TAB":
                    # Top eye (left)
                    top_vp = (offset_x, offset_y + view_h // 2, view_w, view_h // 2)
                    self.ctx.viewport = top_vp
                    self.prog['u_eye_offset'].value = -self.ipd_uv / 2.0
                    self.prog['u_feather_enabled'].value = self.feather_enabled
                    self.prog['u_feather_width'].value = self.feather_width
                    self.prog['u_viewport'].value = top_vp
                    self.quad_vao.render(moderngl.TRIANGLE_STRIP)

                    # Bottom eye (right)
                    bottom_vp = (offset_x, offset_y, view_w, view_h // 2)
                    self.ctx.viewport = bottom_vp
                    self.prog['u_eye_offset'].value = self.ipd_uv / 2.0
                    self.prog['u_feather_enabled'].value = self.feather_enabled
                    self.prog['u_feather_width'].value = self.feather_width
                    self.prog['u_viewport'].value = bottom_vp
                    self.quad_vao.render(moderngl.TRIANGLE_STRIP)

        # Skip swap buffers if window is not visible (for headless/streaming)
        self._render_overlay()
        if self.stream_mode is None:
            glfw.poll_events()
