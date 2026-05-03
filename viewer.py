# viewer.py
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

import ctypes, os, sys

BACKEND = None
# NVIDIA CUDA Version
if "NVIDIA" in DEVICE_INFO:
    BACKEND = "CUDA"
    class CUDART_GL:
        """CUDA-OpenGL interop helper (based on local_viewer.py)."""
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
            self.lib.cudaGraphicsGLRegisterBuffer.argtypes = [
                ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_uint
            ]
            self.lib.cudaGraphicsUnregisterResource.argtypes = [ctypes.c_void_p]
            self.lib.cudaGraphicsMapResources.argtypes = [
                ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p
            ]
            self.lib.cudaGraphicsUnmapResources.argtypes = [
                ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p
            ]
            self.lib.cudaGraphicsResourceGetMappedPointer.argtypes = [
                ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p
            ]
            self.lib.cudaMemcpy.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
            ]
            self.lib.cudaSetDevice(device_id)

        def register_buffer(self, pbo_id):
            resource = ctypes.c_void_p()
            res = self.lib.cudaGraphicsGLRegisterBuffer(ctypes.byref(resource), pbo_id, 1)  # 1 = cudaGraphicsRegisterFlagsNone
            if res != 0:
                raise RuntimeError(f"cudaGraphicsGLRegisterBuffer failed: {res}")
            return resource

        def unregister_resource(self, resource):
            self.lib.cudaGraphicsUnregisterResource(resource)

        def map_resource(self, resource):
            res = self.lib.cudaGraphicsMapResources(1, ctypes.byref(resource), None)
            if res != 0:
                raise RuntimeError(f"cudaGraphicsMapResources failed: {res}")
            ptr = ctypes.c_void_p()
            size = ctypes.c_size_t()
            res = self.lib.cudaGraphicsResourceGetMappedPointer(ctypes.byref(ptr), ctypes.byref(size), resource)
            if res != 0:
                self.lib.cudaGraphicsUnmapResources(1, ctypes.byref(resource), None)
                raise RuntimeError(f"cudaGraphicsResourceGetMappedPointer failed: {res}")
            return ptr.value

        def unmap_resource(self, resource):
            self.lib.cudaGraphicsUnmapResources(1, ctypes.byref(resource), None)

        def memcpy_d2d(self, dst_ptr, src_ptr, size):
            # cudaMemcpyDeviceToDevice = 3
            res = self.lib.cudaMemcpy(dst_ptr, src_ptr, size, 3)
            if res != 0:
                raise RuntimeError(f"cudaMemcpy failed: {res}")

elif "AMD" in DEVICE_INFO:
    BACKEND = "HIP"
    class CUDART_GL:
        """
        HIP-OpenGL interop helper – performance‑tuned for AMD GPUs.
        Equivalent to the CUDA version, but uses the HIP runtime.
        """

        HIP_SUCCESS = 0
        # IMPORTANT: use the same flag as CUDA's cudaGraphicsRegisterFlagsWriteDiscard
        HIP_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD = 1
        HIP_MEMCPY_DEVICE_TO_DEVICE = 3          # hipMemcpyDeviceToDevice

        def __init__(self, device_id=0):
            # 1. Locate the HIP runtime library (libamdhip64.so / amdhip64.dll)
            torch_dir = os.path.dirname(torch.__file__)
            site_packages = os.path.dirname(torch_dir)

            # Common install locations – extend as needed
            candidates = [
                os.path.join(torch_dir, "lib"),
                os.path.join(site_packages, "_rocm_sdk_core", "bin"),
                os.path.join(site_packages, "_rocm_sdk_core", "lib"),
                os.path.join(site_packages, "_rocm_sdk_devel", "bin"),
                os.path.join(site_packages, "_rocm_sdk_devel", "lib"),
            ]

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

            # hipGraphicsMapResources – stream now accepted (0 = default stream)
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
    uniform float u_eye_offset;    // e.g. ±0.03 (positive = right eye)
    uniform float u_depth_strength;// parallax intensity
    uniform float u_convergence;   // depth value at screen plane (0–1)

    // Optimized inpainting controls
    uniform float u_search_radius = 12.0;  // horizontal search distance
    uniform float u_depth_tolerance = 0.012; // background detection threshold
    uniform float u_blur_radius = 2.5;     // final smoothing

    vec2 pixel_size = 1.0 / u_resolution;

    // FAST DISOCCLUSION DETECTION
    bool is_disoccluded(vec2 base_uv, vec2 shifted_uv, float center_depth) {
        // Bounds check
        if (shifted_uv.x < 0.0 || shifted_uv.x > 1.0 ||
            shifted_uv.y < 0.0 || shifted_uv.y > 1.0)
            return true;

        // Fast depth discontinuity check (3-tap only)
        vec2 grad_dir = vec2(sign(u_eye_offset), 0.0);
        float d_left  = texture(tex_depth, base_uv - grad_dir * pixel_size * 2.0).r;
        float d_right = texture(tex_depth, base_uv + grad_dir * pixel_size * 2.0).r;
        
        // Depth jump threshold (tuned for sharp edges)
        if (abs(d_left - d_right) > 0.08)
            return true;

        return false;
    }

    // Uses directional search to find background pixels efficiently
    vec4 push_pull_inpaint(vec2 uv_coord, float center_depth_inv) {
        vec4 best_color = vec4(0.0);
        float best_weight = 0.0;
        int search_range = int(u_search_radius);
        
        // Directional search: prioritize searching opposite to eye shift
        // (background pixels are more likely in that direction)
        int search_dir = u_eye_offset > 0.0 ? -1 : 1;
        
        // Phase 1: Directional sweep (most samples here)
        for (int i = 1; i <= search_range; i++) {
            vec2 sample_uv = uv_coord + vec2(search_dir * i * pixel_size.x, 0.0);
            
            if (sample_uv.x < 0.0 || sample_uv.x > 1.0) continue;
            
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
        
        // Phase 2: Opposite direction (fewer samples)
        if (best_weight < 2.0) {
            int opposite_range = search_range / 2;
            for (int i = 1; i <= opposite_range; i++) {
                vec2 sample_uv = uv_coord - vec2(search_dir * i * pixel_size.x, 0.0);
                
                if (sample_uv.x < 0.0 || sample_uv.x > 1.0) continue;
                
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
    /*
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
    */

    // MAIN
    void main() {
        vec2 flipped_uv = vec2(uv.x, 1.0 - uv.y);

        // Sample depth
        float depth = texture(tex_depth, flipped_uv).r;
        float depth_inv = -depth;

        // Calculate parallax shift
        float shift = (depth_inv + u_convergence);
        vec2 shifted_uv = flipped_uv - vec2(u_eye_offset * shift * u_depth_strength, 0.0);

        vec4 color;
        
        // Check if this pixel needs inpainting
        if (is_disoccluded(flipped_uv, shifted_uv, depth)) {
            // Disoccluded region → use optimized inpainting
            color = push_pull_inpaint(flipped_uv, depth_inv);
            // Alternative: color = separable_inpaint(flipped_uv, depth_inv);
        } else {
            // Normal sampling
            color = texture(tex_color, shifted_uv);
        }

        // Subtle edge fade to hide artifacts
        vec2 border = smoothstep(0.0, 0.015, shifted_uv) * 
                      smoothstep(1.0, 0.985, shifted_uv);
        color.a = min(border.x, border.y);

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

def add_logo(window):
    """Optimized logo loading with lazy imports"""
    from PIL import Image
    glfw_img = Image.open("icon2.ico")  # Path to your icon file
    if OS_NAME != "Darwin":
        glfw_img = crop_icon(glfw_img)
        glfw.set_window_icon(window, 1, [glfw_img])

class StereoWindow:
    """Optimized stereo viewer with performance improvements"""

    def __init__(self, capture_mode="Monitor", monitor_index=0, ipd=0.064, depth_ratio=1.0, convergence=0.0, display_mode="Half-SBS", fill_16_9=True, show_fps=True, use_3d=False, fix_aspect=False, stream_mode=None, lossless_scaling=False, specify_display=False, stereo_display_index=0, frame_size=(1280, 720), use_cuda=False, cuda_device_id=0, **kwargs):
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
        self._modes = ["Full-SBS", "Half-SBS", "TAB", "Depth Map"]
        self.display_mode = display_mode
        self._texture_size = None
        self.fill_16_9 = fill_16_9
        self.frame_size = frame_size
        self.aspect = self.frame_size[0] / self.frame_size[1]
        self.fix_aspect = fix_aspect
        self.show_fps = show_fps
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
            'fps_text': None,
            'depth_text': None,
            'last_update': 0.0,
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
        glfw.swap_interval(1) # VSync on
        
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

    def __del__(self):
        self.cleanup_cuda()

    def _init_cuda_pbos(self, width, height):
        """Create PBOs and register them with CUDA. Disable CUDA on failure."""
        if not self.use_cuda or self._cudart is not None:
            return
        try:
            self._cudart = CUDART_GL(self.cuda_device_id)

            # Colour PBO (RGB8, 3 bytes/pixel)
            self._pbo_color = glGenBuffers(1)
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self._pbo_color)
            glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 3, None, GL_STREAM_DRAW)
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
            self._cuda_resource_color = self._cudart.register_buffer(self._pbo_color)

            # Depth PBO (float32, 4 bytes/pixel)
            self._pbo_depth = glGenBuffers(1)
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self._pbo_depth)
            glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4, None, GL_STREAM_DRAW)
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
            self._cuda_resource_depth = self._cudart.register_buffer(self._pbo_depth)

            print(f"[Main] Enabled {BACKEND}-GL interop acceleration ({BACKEND}: {self.cuda_device_id})")
        except Exception as e:
            print(f"[Main] {BACKEND}-GL interop initialization failed: {e}")
            print(f"[Main] Falling back to CPU-GL interop.")
            self.use_cuda = False
            self.cleanup_cuda()   # clean up any partial allocations

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
            if self._cuda_resource_color:
                self._cudart.unregister_resource(self._cuda_resource_color)
            if self._cuda_resource_depth:
                self._cudart.unregister_resource(self._cuda_resource_depth)
            if self._pbo_color and bool(glDeleteBuffers):
                glDeleteBuffers(1, [self._pbo_color])
            if self._pbo_depth and bool(glDeleteBuffers):
                glDeleteBuffers(1, [self._pbo_depth])
            self._cudart = None
    
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

        # Compose the strings to display
        fps_text = f"FPS: {self.actual_fps:.1f}" if self.show_fps else ""
        latency_text = f"Latency: {self.total_latency:.1f} ms" if self.show_fps else ""
        depth_text = f"Depth: {self.depth_ratio:.1f}" if self.show_depth_ratio else ""
        mouse_text = f"Mouse: {'Pass' if self.mouse_pass_through else 'Normal'}" if self.show_mouse_state else ""

        # Decide whether to regenerate overlay
        cache = self._overlay_cache
        needs_regen = False
        if cache['image'] is None:
            needs_regen = True
        elif latency_text != cache.get('latency_text'):
            needs_regen = True
        elif fps_text != cache.get('fps_text') or depth_text != cache.get('depth_text') or mouse_text != cache.get('mouse_text'):
            needs_regen = True
        elif (current_time - cache.get('last_update', 0.0)) >= self.overlay_update_interval:
            # periodic regen in case font metrics or size changed
            needs_regen = True

        if needs_regen:
            overlay_arr = self._generate_overlay_image(fps_text, latency_text, depth_text, mouse_text)
            cache['image'] = overlay_arr
            cache['fps_text'] = fps_text
            cache['latency_text'] = latency_text
            cache['depth_text'] = depth_text
            cache['mouse_text'] = mouse_text
            cache['last_update'] = current_time

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

        # Slice overlay and frame
        overlay_slice = overlay_arr[0:ov_h_clipped, 0:ov_w_clipped]
        frame_region = rgb_frame[pos_y:end_y, pos_x:end_x]

        # Alpha blending: result = overlay.rgb * alpha + frame * (1 - alpha)
        alpha = overlay_slice[..., 3:4].astype(np.float32) / 255.0  # H x W x 1
        overlay_rgb = overlay_slice[..., :3].astype(np.float32)
        frame_rgb = frame_region.astype(np.float32)

        blended = (overlay_rgb * alpha) + (frame_rgb * (1.0 - alpha))
        # write back blended region into original frame (as uint8)
        rgb_frame[pos_y:end_y, pos_x:end_x] = np.clip(blended, 0, 255).astype(np.uint8)

        return rgb_frame
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
                        # Screen is taller — fit by width
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
        """Optimized frame update – tries CUDA zero‑copy, falls back to CPU."""
        # Convert tensors to numpy for overlay (always CPU)
        if hasattr(depth, 'detach'):
            depth_np = depth.detach().cpu().contiguous().float().numpy()
        else:
            depth_np = depth

        if hasattr(rgb, 'detach'):
            try:
                rgb_np = rgb.permute(1, 2, 0).detach().contiguous().clamp(0, 255).to(torch.uint8).cpu().numpy()
            except RuntimeError as e:
                print(f"[update_frame] RuntimeError converting RGB tensor to numpy: {e}")
                rgb_np = None
            except Exception as e:
                print(f"[update_frame] Error converting RGB tensor to numpy: {e}")
                rgb_np = None
        else:
            rgb_np = rgb

        # Add overlay (CPU only)
        if self.display_mode != "Depth Map":
            rgb_with_overlay = self._add_overlay(rgb_np)
        else:
            rgb_with_overlay = rgb_np

        # Update FPS/Latency
        if current_fps is not None:
            self.actual_fps = current_fps
        if current_latency is not None:
            self.total_latency = current_latency * 1000

        h, w, _ = rgb_with_overlay.shape

        # Recreate textures and PBOs if size changed
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

        # Try CUDA zero‑copy path first
        cuda_success = False
        if self.use_cuda and self._cudart is not None:
            try:
                # Check that tensors are on GPU and compatible
                if not (hasattr(rgb, 'is_cuda') and rgb.is_cuda and
                        hasattr(depth, 'is_cuda') and depth.is_cuda):
                    raise RuntimeError("Tensors not on GPU")

                # Convert overlay (CPU) to GPU tensor (uint8)
                rgb_gpu = torch.from_numpy(rgb_with_overlay).cuda(self.cuda_device_id).contiguous()
                depth_gpu = depth.contiguous().float()

                # Upload using PBOs
                self._upload_color_cuda(rgb_gpu)
                self._upload_depth_cuda(depth_gpu)
                cuda_success = True
            except Exception as e:
                self.use_cuda = False   # permanently disable CUDA for this instance
                self.cleanup_cuda()

        # Fallback to CPU path if CUDA failed or not available
        if not cuda_success:
            if self.display_mode == "Depth Map" and not self.show_original_in_depth_mode:
                self.depth_tex.write(depth_np.tobytes())
                if self._last_display_mode != "Depth Map" or self._last_show_original != self.show_original_in_depth_mode:
                    rgb_u8 = rgb_with_overlay.astype('uint8', copy=False)
                    self.color_tex.write(rgb_u8.tobytes())
            else:
                rgb_u8 = rgb_with_overlay.astype('uint8', copy=False)
                self.color_tex.write(rgb_u8.tobytes())
                self.depth_tex.write(depth_np.tobytes())

        # Cache current state
        self._last_display_mode = self.display_mode
        self._last_show_original = self.show_original_in_depth_mode

        # Show window after first frame (unchanged – keep as is)
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
            if self.stream_mode is None:
                glfw.poll_events()
            return
        
        # Rest of the rendering for other modes...
        if self.fix_aspect:
            if self.display_mode == "Full-SBS":
                glfw.set_window_aspect_ratio(self.window, 2*tex_w, tex_h)
            else:
                glfw.set_window_aspect_ratio(self.window, tex_w, tex_h)
        else:
            glfw.set_window_aspect_ratio(self.window, glfw.DONT_CARE, glfw.DONT_CARE)
        
        # Clear screen once
        self.ctx.clear(0.1, 0.1, 0.1)
        
        # Handle other display modes...
        if self.fill_16_9:
            if self.display_mode in ["Full-SBS", "Half-SBS", "TAB"]:
                # Bind textures once
                self.color_tex.use(location=0)
                self.depth_tex.use(location=1)
                
                # Set common uniform values
                self.prog['u_depth_strength'].value = self.depth_strength * self.depth_ratio
            
                if self.display_mode == "Full-SBS":
                    # Full Side-by-Side mode
                    src_w, src_h = tex_w, tex_h
                    max_w, max_h = win_w / 2.0, win_h
                    render_w, render_h = self._compute_render_size(max_w, max_h, src_w, src_h)
                    center_y = win_h / 2.0
                    
                    # Left view
                    self.ctx.viewport = (
                        int(win_w / 4.0 - render_w / 2),
                        int(center_y - render_h / 2),
                        render_w, render_h
                    )
                    self.prog['u_eye_offset'].value = -self.ipd_uv / 2.0
                    self._last_eye_offset_set = -self.ipd_uv / 2.0
                    self.quad_vao.render(moderngl.TRIANGLE_STRIP)
                    
                    # Right view
                    self.ctx.viewport = (
                        int(3 * win_w / 4.0 - render_w / 2),
                        int(center_y - render_h / 2),
                        render_w, render_h
                    )
                    self.prog['u_eye_offset'].value = self.ipd_uv / 2.0
                    self._last_eye_offset_set = self.ipd_uv / 2.0
                    self.quad_vao.render(moderngl.TRIANGLE_STRIP)

                elif self.display_mode == "Half-SBS":
                    # Half Side-by-Side mode
                    src_w, src_h = tex_w / 2.0, tex_h
                    max_w, max_h = win_w / 2.0, win_h
                    render_w, render_h = self._compute_render_size(max_w, max_h, src_w, src_h)
                    center_y = win_h / 2.0
                    
                    # Left view
                    self.ctx.viewport = (
                        int(win_w / 4.0 - render_w / 2),
                        int(center_y - render_h / 2),
                        render_w, render_h
                    )
                    self.prog['u_eye_offset'].value = -self.ipd_uv / 2.0
                    self._last_eye_offset_set = -self.ipd_uv / 2.0
                    self.quad_vao.render(moderngl.TRIANGLE_STRIP)
                    
                    # Right view
                    self.ctx.viewport = (
                        int(3 * win_w / 4.0 - render_w / 2),
                        int(center_y - render_h / 2),
                        render_w, render_h
                    )
                    self.prog['u_eye_offset'].value = self.ipd_uv / 2.0
                    self._last_eye_offset_set = self.ipd_uv / 2.0
                    self.quad_vao.render(moderngl.TRIANGLE_STRIP)

                elif self.display_mode == "TAB":
                    # Top-and-Bottom mode
                    src_w, src_h = tex_w, tex_h / 2.0
                    max_w, max_h = win_w, win_h / 2.0
                    render_w, render_h = self._compute_render_size(max_w, max_h, src_w, src_h)
                    
                    # Top view
                    self.ctx.viewport = (
                        int(win_w / 2.0 - render_w / 2),
                        int(win_h / 4.0 - render_h / 2),
                        render_w, render_h
                    )
                    self.prog['u_eye_offset'].value = -self.ipd_uv / 2.0
                    self._last_eye_offset_set = -self.ipd_uv / 2.0
                    self.quad_vao.render(moderngl.TRIANGLE_STRIP)
                    
                    # Bottom view
                    self.ctx.viewport = (
                        int(win_w / 2.0 - render_w / 2),
                        int(3 * win_h / 4.0 - render_h / 2),
                        render_w, render_h
                    )
                    self.prog['u_eye_offset'].value = self.ipd_uv / 2.0
                    self._last_eye_offset_set = self.ipd_uv / 2.0
                    self.quad_vao.render(moderngl.TRIANGLE_STRIP)
        
        else:
            # Determine effective stereo frame size by display mode
            if self.display_mode == "Full-SBS":
                disp_w, disp_h = 2 * tex_w, tex_h
            elif self.display_mode == "Half-SBS":
                disp_w, disp_h = tex_w, tex_h
            elif self.display_mode == "TAB":
                disp_w, disp_h = tex_w, tex_h
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

            if self.display_mode in ["Full-SBS", "Half-SBS", "TAB"]:
                self.color_tex.use(0)
                self.depth_tex.use(1)
                self.prog['u_depth_strength'].value = self.depth_strength * self.depth_ratio
                self._last_depth_strength_set = self.depth_strength * self.depth_ratio

                if self.display_mode == "Full-SBS":
                    # Left eye
                    self.ctx.viewport = (offset_x, offset_y, view_w // 2, view_h)
                    self.prog['u_eye_offset'].value = -self.ipd_uv / 2.0
                    self._last_eye_offset_set = -self.ipd_uv / 2.0
                    self.quad_vao.render(moderngl.TRIANGLE_STRIP)

                    # Right eye
                    self.ctx.viewport = (offset_x + view_w // 2, offset_y, view_w // 2, view_h)
                    self.prog['u_eye_offset'].value = self.ipd_uv / 2.0
                    self._last_eye_offset_set = self.ipd_uv / 2.0
                    self.quad_vao.render(moderngl.TRIANGLE_STRIP)

                elif self.display_mode == "Half-SBS":
                    # Same as FULL but both squeezed into width
                    self.ctx.viewport = (offset_x, offset_y, view_w // 2, view_h)
                    self.prog['u_eye_offset'].value = -self.ipd_uv / 2.0
                    self._last_eye_offset_set = -self.ipd_uv / 2.0
                    self.quad_vao.render(moderngl.TRIANGLE_STRIP)

                    self.ctx.viewport = (offset_x + view_w // 2, offset_y, view_w // 2, view_h)
                    self.prog['u_eye_offset'].value = self.ipd_uv / 2.0
                    self._last_eye_offset_set = self.ipd_uv / 2.0
                    self.quad_vao.render(moderngl.TRIANGLE_STRIP)

                elif self.display_mode == "TAB":
                    # Top eye
                    self.ctx.viewport = (offset_x, offset_y + view_h // 2, view_w, view_h // 2)
                    self.prog['u_eye_offset'].value = -self.ipd_uv / 2.0
                    self._last_eye_offset_set = -self.ipd_uv / 2.0
                    self.quad_vao.render(moderngl.TRIANGLE_STRIP)

                    # Bottom eye
                    self.ctx.viewport = (offset_x, offset_y, view_w, view_h // 2)
                    self.prog['u_eye_offset'].value = self.ipd_uv / 2.0
                    self._last_eye_offset_set = self.ipd_uv / 2.0
                    self.quad_vao.render(moderngl.TRIANGLE_STRIP)