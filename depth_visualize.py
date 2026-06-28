# depth.py
import os, warnings
import contextlib
from utils import DEVICE_ID, MODEL_ID, CACHE_PATH, FP16, DEPTH_RESOLUTION, AA_STRENGTH, FOREGROUND_SCALE, USE_TORCH_COMPILE, USE_TENSORRT, RECOMPILE_TRT, USE_COREML, RECOMPILE_COREML, USE_OPENVINO, RECOMPILE_OPENVINO, USE_MIGRAPHX, RECOMPILE_MIGRAPHX, DISABLE_TRT_KEYWORDS, DISABLE_MIGRAPHX_KEYWORDS,  DISABLE_COREML_KEYWORDS, DISABLE_CUDNN_GFX, DISABLE_TRITON_GFX, DISABLE_OPENVINO_KEYWORDS, DEBUG, DEVICE_ID, DEVICE_INFO, DEVICE, TRT_FIX_KEYWORDS, FORCE_FP32_KEYWORDS, CAPTURE_MODE, enable_torch_compile_fallback, torch_compile_or_original, torch_compile_with_runtime_fallback
import torch
# Support for old AMD GPU with ZLUDA support (hide)
# try:
#     import zluda 
#     torch.backends.cudnn.enabled = False  # Disable cuDNN for ZLUDA compatibility
# except ImportError:
#     pass

import torch.nn.functional as F
from transformers import AutoModelForDepthEstimation
import numpy as np
from threading import Lock
import cv2

# debug constants
from PIL import Image
img  = Image.open("assets/cats.jpg").convert("RGB")
image_rgb = np.array(img).astype(np.float32)


torch.set_num_threads(1)
# Testing parameters 
DEBUG = True
AA_STRENGTH = 0
FP16 = True
FILL_16_9 = True
DEPTH_RESOLUTION = 336
FOREGROUND_SCALE = 0

USE_TORCH_COMPILE = True

USE_TENSORRT = True
RECOMPILE_TRT = True

USE_MIGRAPHX = True
RECOMPILE_MIGRAPHX = True

USE_COREML = False
RECOMPILE_COREML = True

USE_OPENVINO = False
RECOMPILE_OPENVINO = True


# MODEL_ID ="depth-anything/DA3-LARGE-1.1"
# MODEL_ID ="depth-anything/DA3-SMALL"
# MODEL_ID ="depth-anything/DA3NESTED-GIANT-LARGE"
# MODEL_ID = "depth-anything/DA3METRIC-LARGE"
# MODEL_ID = "LiheYoung/depth-anything-small-hf"
# MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
# MODEL_ID = "lc700x/depth-anything-indoor-large-hf"
# MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
# MODEL_ID = "depth-anything/Video-Depth-Anything-Small"
# MODEL_ID = "depth-anything/DA3MONO-LARGE"
# MODEL_ID = "Intel/dpt-large"
# MODEL_ID = "apple/DepthPro-hf"
# MODEL_ID = "Intel/zoedepth-nyu"
# MODEL_ID = "Intel/zoedepth-nyu-kitti"
# MODEL_ID = "lc700x/dpt-hybrid-midas-hf"
# MODEL_ID = "lc700x/dpt-large-redesign-hf"
# MODEL_ID = "lc700x/Distill-Any-Depth-Base-hf"
# MODEL_ID = "Intel/dpt-beit-base-384"
MODEL_ID = "lc700x/InfiniDepth-Small"
# MODEL_ID = "xingyang1/Distill-Any-Depth-Small-hf"

IS_CUDA = "CUDA" in DEVICE_INFO
IS_NVIDIA = "CUDA" in DEVICE_INFO and "NVIDIA" in DEVICE_INFO
IS_LEGACY_NVIDIA = False
IS_AMD_ROCM = "CUDA" in DEVICE_INFO and "AMD" in DEVICE_INFO and "ZLUDA" not in DEVICE_INFO
IS_DIRECTML = "DirectML" in DEVICE_INFO
IS_XPU = "XPU" in DEVICE_INFO
IS_MPS = "MPS" in DEVICE_INFO
IS_CPU = "CPU" in DEVICE_INFO

USE_TORCH_COMPILE = False if not IS_CUDA else USE_TORCH_COMPILE
USE_TENSORRT = False if not IS_NVIDIA else USE_TENSORRT
USE_COREML = False if not IS_MPS else USE_COREML
USE_OPENVINO = False if not IS_XPU else USE_OPENVINO
USE_MIGRAPHX = False if not IS_AMD_ROCM else USE_MIGRAPHX

print(f"[Main] Using HF Endpoint: {os.environ.get('HF_ENDPOINT')}")
print(f"{DEVICE_INFO}")
print(f"Model: {MODEL_ID.split('/')[-1]}")

if not DEBUG:
    warnings.filterwarnings('ignore') # disable for debug

warnings.filterwarnings("ignore", message=".*ONNX export mode is set to TrainingMode.EVAL.*instance_norm.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

# Try import OpenVINO runtime
if IS_XPU:
    if USE_OPENVINO:
        try:
            import openvino as ov
            OPENVINO_AVAILABLE = True
            # Disable OpenVivo
            USE_OPENVINO = False if any(x in MODEL_ID.lower()  for x in DISABLE_OPENVINO_KEYWORDS) else True
            USE_TORCH_COMPILE = False
        except Exception:
            ov = None
            OPENVINO_AVAILABLE = False
            USE_OPENVINO = False

    # Decide OpenVINO device (prefer GPU)
    OPENVINO_DEVICE = None
    if USE_OPENVINO and OPENVINO_AVAILABLE:
        try:
            core_tmp = ov.Core()
            devices_tmp = core_tmp.available_devices
            if any("GPU" in d for d in devices_tmp):
                OPENVINO_DEVICE = "GPU"
            elif any("CPU" in d for d in devices_tmp):
                OPENVINO_DEVICE = "CPU"
            else:
                OPENVINO_DEVICE = devices_tmp[0] if len(devices_tmp) > 0 else None
        except Exception:
            OPENVINO_DEVICE = None
else:
    USE_OPENVINO = False

@contextlib.contextmanager
def coreml_safe_interpolate():
    yield

if USE_COREML and IS_MPS:    
    try:
        import coremltools as ct  # optional on macOS only
    except Exception:
        ct = None
        
    USE_COREML = ct is not None
    # imports for CoreML
    if USE_COREML:
        # export-time patch to replace bicubic -> bilinear
        @contextlib.contextmanager
        def coreml_safe_interpolate():
            """
            Temporarily monkey-patch torch.nn.functional.interpolate so that any call
            using mode='bicubic' will instead use 'bilinear' during export.
            This only affects the code inside the 'with' block (used for TorchScript/CoreML export).
            """
            orig_interpolate = F.interpolate

            def patched_interpolate(
                input,
                size=None,
                scale_factor=None,
                mode='nearest',
                align_corners=None,
                recompute_scale_factor=None,
                antialias=False,
            ):
                if mode == "bicubic":
                    mode = "bilinear"

                return orig_interpolate(
                    input,
                    size=size,
                    scale_factor=scale_factor,
                    mode=mode,
                    align_corners=align_corners,
                    recompute_scale_factor=recompute_scale_factor,
                    antialias=antialias,
                )

            F.interpolate = patched_interpolate
            try:
                yield
            finally:
                F.interpolate = orig_interpolate
                
# disable FP16 on DirectML and MPS without coreml          
if IS_DIRECTML or IS_MPS or ((USE_TENSORRT or USE_COREML or USE_OPENVINO or USE_MIGRAPHX) and MODEL_ID in TRT_FIX_KEYWORDS) or (MODEL_ID == "Intel/dpt-beit-large-512" and DEPTH_RESOLUTION != 512) or (MODEL_ID in FORCE_FP32_KEYWORDS):
    FP16 = False
    print("FP16 disabled")

# Disable CoreML for models whose architecture can't be traced (implicit heads, etc.)
if any(x in MODEL_ID.lower() for x in DISABLE_COREML_KEYWORDS):
    USE_COREML = False

# Optimization for CUDA
if IS_CUDA:
    # Set cudnn benchmark for performance
    torch.backends.cudnn.benchmark = True
    os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"] ="1"
    
    if USE_TORCH_COMPILE:
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module=r"torch\._inductor\.lowering"
        )
    
    # Enable TF32 for matrix multiplications
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    # Enable math attention (safe-guard in case attributes unavailable)
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

if IS_NVIDIA:
    # Disable torch.compile for old NVIDIA gpu
    def is_legacy_nvidia():
        """Disable torch.compile for old NVIDIA gpu"""
        device = torch.device(f'cuda:{DEVICE_ID}')
        props = torch.cuda.get_device_properties(device)
        
        # check GPU compute capability
        major, minor = props.major, props.minor
        compute_capability = major + minor * 0.1
        
        # GTX 1050: compute_capability = 6.1
        if compute_capability < 7.0:  # below Volta architecture
            # print(f"[Main] Disabled torch.compile and TensorRT for {props.name}. ")
            # torch._dynamo.config.suppress_errors = True
            # os.environ['TORCHINDUCTOR_DISABLE'] = '1'
            return True
        return False
    IS_LEGACY_NVIDIA = is_legacy_nvidia()
    if IS_LEGACY_NVIDIA:
        USE_TORCH_COMPILE = False
        # USE_TENSORRT = False  # Disable TensorRT for legacy NVIDIA GPUs due to potential compatibility issues
    # Disable TRT for unsupported models
    if any(x in MODEL_ID.lower() for x in DISABLE_TRT_KEYWORDS):
        USE_TENSORRT = False

if IS_AMD_ROCM:
    import platform, re, subprocess
    
    def get_gfx_arch():
        """
        Detect the current GPU's GFX architecture code (e.g., 'gfx1200', 'gfx1030').

        - On **Linux**, it parses `rocminfo` output (looking for `Name: gfx...`).
        - On **Windows**, it parses `hipinfo` output (looking for `gcnArchName: ...`).
        - On other OSes, it tries `rocm_agent_enumerator` as a last resort.

        Returns:
            str: The GFX code, or None if not found / command unavailable.
        """
        system = platform.system()

        try:
            if system == "Linux":
                # Use rocminfo
                result = subprocess.run(['rocminfo'], capture_output=True, text=True, check=True)
                output = result.stdout

                # Most reliable: find a line like "Name: gfx1030"
                match = re.search(r'Name:\s*(gfx[0-9a-f]+)', output, re.IGNORECASE)
                if match:
                    return match.group(1)

                # Fallback: search for any "gfxXXXX" string
                fallback = re.search(r'gfx[0-9a-f]+', output)
                if fallback:
                    return fallback.group(0)

            elif system == "Windows":
                # Use hipinfo (available with ROCm on Windows)
                result = subprocess.run(['hipinfo'], capture_output=True, text=True, check=True)
                output = result.stdout
                match = re.search(r'gcnArchName:\s*(\S+)', output)
                if match:
                    return match.group(1)

            # For other OSes (macOS, etc.) or if the above fail, try rocm_agent_enumerator
            result = subprocess.run(['rocm_agent_enumerator'], capture_output=True, text=True, check=True)
            lines = result.stdout.strip().split()
            if lines:
                return lines[0]   # usually the first GPU

        except (subprocess.SubprocessError, FileNotFoundError):
            # Command not found or execution failed
            pass

        return None
    GPU_ARCH = get_gfx_arch()
    
    # Set up ROCm7 Path
    torch_dir = os.path.dirname(torch.__file__)
    site_packages = os.path.dirname(torch_dir)
    ROCM_PATH = os.path.join(site_packages, "_rocm_sdk_devel")

    os.environ['MIOPEN_ENABLE_LOGGING'] = '0'
    warnings.filterwarnings("ignore", message="bgemm_internal_cublaslt error")
    warnings.filterwarnings("ignore", message="gemm_and_bias error")
    os.environ["HIP_PLATFORM"] = "amd"
    os.environ["HIP_PATH"] = ROCM_PATH
    os.environ["HIP_CLANG_PATH"] = os.path.join(ROCM_PATH, "llvm", "bin")
    os.environ["HIP_INCLUDE_PATH"] = os.path.join(ROCM_PATH, "include")
    os.environ["HIP_LIB_PATH"] = os.path.join(ROCM_PATH, "lib")
    os.environ["HIP_DEVICE_LIB_PATH"] = os.path.join(ROCM_PATH, "lib", "llvm", "amdgcn", "bitcode")
    os.environ["PATH"] = os.pathsep.join([
        os.path.join(ROCM_PATH, "bin"),
        os.path.join(ROCM_PATH, "llvm", "bin"),
        os.environ.get("PATH", "")
    ])
    os.environ["CPATH"] = os.path.join(ROCM_PATH, "include") + os.pathsep + os.environ.get("CPATH", "")
    os.environ["LIBRARY_PATH"] = os.pathsep.join([
        os.path.join(ROCM_PATH, "lib"),
        os.path.join(ROCM_PATH, "lib64"),
        os.environ.get("LIBRARY_PATH", "")
    ])
    os.environ["PKG_CONFIG_PATH"] = os.path.join(ROCM_PATH, "lib", "pkgconfig") + os.pathsep + os.environ.get("PKG_CONFIG_PATH", "")
    if GPU_ARCH in DISABLE_CUDNN_GFX:
        torch.backends.cudnn.enabled = False  # Disable cuDNN for known problematic AMD RX6000 GPUs
        print(f"[Main] Disabled cuDNN for {DEVICE_INFO}. ")
        
    # disable trition for RX5000 series and older AMD GPUs
    is_legacy_amd = GPU_ARCH in DISABLE_TRITON_GFX   
    if is_legacy_amd:
        USE_TORCH_COMPILE = False  # Disable Triton for known problematic AMD GPUs
        print(f"[Main] Disabled torch.compile for {DEVICE_INFO}. ")
    else:
        os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "TRUE" # Enable flash attention for
        os.environ["FLASH_ATTENTION_TRITON_AMD_AUTOTUNE"] = "TRUE" # Enable flash attention autotune for AMD ROCm
    os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1" # Enable AOTriton for ROCm

    # Disable TRT for unsupported models
    if any(x in MODEL_ID.lower() for x in DISABLE_MIGRAPHX_KEYWORDS):
        USE_MIGRAPHX = False

# Model configuration
DTYPE = torch.float16 if FP16 else torch.float32

# Engine input shape, set lazily by _ensure_engine_built() on the first frame.
ENGINE_H, ENGINE_W = None, None

# Get huggingface repo and filename for a given model_id, with some sanity checks
def get_model_path(model_id, cache_dir):
    from huggingface_hub import hf_hub_download
    CKPT_NAMES = ["model.safetensors", "model.pt", "model.ckpt"]
    # Try local cache first
    for filename in CKPT_NAMES:
        try:
            return hf_hub_download(
                repo_id=model_id,
                filename=filename,
                cache_dir=cache_dir,
                local_files_only=True,
            )
        except Exception:
            pass

    # If not cached, download
    for filename in CKPT_NAMES:
        try:
            return hf_hub_download(
                repo_id=model_id,
                filename=filename,
                cache_dir=cache_dir,
            )
        except Exception:
            pass

    raise FileNotFoundError(
        f"No supported model file found in {model_id}"
    )

# Resize-alignment factor for model families that need or benefit from
# aspect-preserving patch-aligned input. Legacy HF/DPT-style models stay on the
# fixed-square path used by depth0.py so their steady-state FPS does not regress.
def get_patch_size():
    if CAPTURE_MODE == "Window":
        return None
    if "infinidepth" in MODEL_ID.lower():
        return 16
    if "da3" in MODEL_ID.lower() or "any" in MODEL_ID.lower() or "dinov2" in MODEL_ID.lower():
        return 14
    return None

if IS_CUDA:

    def process(img_uint8: torch.Tensor, target_height: int) -> torch.Tensor:
        """
        Memory-efficient version for uint8 tensors.
        Minimizes temporary memory allocations during processing.
        img_uint8: Input image tensor (HWC format)
        target_height: Desired height for the output tensor
        output: Processed tensor (CHW format)
        """
        # check if input is numpy
        if isinstance(img_uint8, np.ndarray):
            img_uint8 = torch.from_numpy(img_uint8).to(DEVICE)

        img_uint8 = img_uint8[..., [2, 1, 0]].permute(2, 0, 1).contiguous()
        _, H0, W0 = img_uint8.shape
        
        if target_height >= H0:
            return img_uint8.to(DEVICE, dtype=DTYPE)
        
        new_width = int(W0 * target_height / H0)
        new_height = (target_height // 2) * 2
        new_width = (new_width // 2) * 2
        
        # Use in-place operations where possible to reduce memory usage
        with torch.no_grad():  # Disable gradient computation for inference
            img_float = img_uint8.to(DEVICE, dtype=DTYPE)
            
            # Resize
            result = F.interpolate(
                img_float.unsqueeze(0),
                size=(new_height, new_width),
                mode='bilinear',
                align_corners=False,
                antialias=new_height < H0
            ).squeeze(0)
        return result
    

else:
    def process(img_rgb: np.ndarray | cv2.UMat, height: int) -> np.ndarray:

        """
        Resize BGR/UMat image to target height, keeping aspect ratio.
        Uses cv2.UMat for GPU acceleration when possible.
        """
        if isinstance(img_rgb, torch.Tensor):
            # Tensor capture path is already RGB CHW; keep resize on accelerator.
            if img_rgb.ndim == 3 and img_rgb.shape[0] in (3, 4):
                img_rgb = img_rgb[:3]
            elif img_rgb.ndim == 3 and img_rgb.shape[-1] >= 3:
                img_rgb = img_rgb[..., :3].permute(2, 0, 1).contiguous()
            else:
                raise ValueError(f"Unsupported tensor image shape: {tuple(img_rgb.shape)}")

            img_rgb = img_rgb.to(DEVICE)
            _, h0, w0 = img_rgb.shape
            if height >= h0:
                return img_rgb

            width = int(w0 * height / h0)
            height = (height // 2) * 2
            width = (width // 2) * 2

            with torch.no_grad(), coreml_safe_interpolate():
                return F.interpolate(
                    img_rgb.to(dtype=DTYPE).unsqueeze(0),
                    size=(height, width),
                    mode='bilinear',
                    align_corners=False,
                    antialias=False
                ).squeeze(0)
       
        # Determine if input is UMat
        is_umat = isinstance(img_rgb, cv2.UMat)

        # Get original size
        if is_umat:
            h0, w0 = img_rgb.get().shape[:2]
        else:
            h0, w0 = img_rgb.shape[:2]

        # Compute new size
        width = int(w0 * height / h0)

        # Convert BGRA to RGB if needed (WindowsCapture gives BGRA)
        if img_rgb.shape[2] == 4:
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGRA2RGB)
        elif img_rgb.shape[2] == 3:
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

        # If no resize necessary, return as-is
        if height >= h0:
            return img_rgb

        # Resize using CPU
        interpolation = cv2.INTER_AREA if height < h0 else cv2.INTER_CUBIC
        resized = cv2.resize(img_rgb, (width, height), interpolation=interpolation)

        return resized

# Folder to store compiled model / cache
MODEL_FOLDER = os.path.join(CACHE_PATH, "models--"+MODEL_ID.replace("/", "--"))
# Load depth model - either original or ONNX
DTYPE_INFO = "fp16" if FP16 else "fp32"
# Engine paths embed the shape; set lazily in _ensure_engine_built().
ONNX_PATH = None
TRT_PATH = None
COREML_PATH = None
MIGRAPHX_PATH = None
# Single character digits and letters for "FPS: XX.X"
font_dict = {
    "0": ["111","101","101","101","111"],
    "1": ["010","110","010","010","111"],
    "2": ["111","001","111","100","111"],
    "3": ["111","001","111","001","111"],
    "4": ["101","101","111","001","001"],
    "5": ["111","100","111","001","111"],
    "6": ["111","100","111","101","111"],
    "7": ["111","001","010","100","100"],
    "8": ["111","101","111","101","111"],
    "9": ["111","101","111","001","111"],
    "F": ["111","100","110","100","100"],
    "P": ["110","101","110","100","100"],
    "S": ["111","100","111","001","111"],
    ":": ["000","010","000","010","000"],
    ".": ["000","000","000","000","010"],  # for decimal point
    " ": ["000","000","000","000","000"],
}

# Model casting helper
def maybe_autocast(device):
    return (
        torch.autocast(device_type=device.type, enabled=True)
        if device.type != "privateuseone"
        else contextlib.nullcontext()
    )

# check if it is metric model
def is_metric():
    if 'metric'  in MODEL_ID.lower() or 'kitti'  in MODEL_ID.lower() or 'nyu' in MODEL_ID.lower() or 'depth-ai' in MODEL_ID.lower() or 'da3' in MODEL_ID.lower():
        return True
    else:
        return False

# GPU resize (torch F.interpolate). Longest side -> target keeping aspect, then
# each dim aligned to the patch grid, fused into ONE interpolate (one less
# DirectML kernel per frame). CUDA uses bicubic+antialias to match official DA3
# preprocessing; other backends use bilinear (bicubic+antialias is ~60x slower
# on DirectML and unimplemented for fp16).
def _resize_patch_aligned_t(tensor: torch.Tensor, target: int, patch: int) -> torch.Tensor:
    _, _, h, w = tensor.shape
    longest = max(h, w)
    scale = target / float(longest) if longest != target else 1.0
    sh = max(1, int(round(h * scale)))
    sw = max(1, int(round(w * scale)))

    def nearest_multiple(x, p):
        down = (x // p) * p
        up = down + p
        return up if abs(up - x) <= abs(x - down) else down

    new_h = max(1, nearest_multiple(sh, patch))
    new_w = max(1, nearest_multiple(sw, patch))
    if new_h == h and new_w == w:
        return tensor
    dtype = tensor.dtype if tensor.dtype.is_floating_point else torch.float32
    if IS_CUDA:
        return F.interpolate(tensor.to(dtype), size=(new_h, new_w), mode='bicubic', align_corners=False, antialias=True)
    # DirectML/CPU: bilinear cost scales with INPUT size, so a single full-res
    # downscale is ~8x slower than needed. Cheaply pre-decimate (strided slice)
    # until the longest side is ~2x the target, then bilinear to the exact size.
    stride = longest // (target * 2)
    if stride > 1:
        tensor = tensor[:, :, ::stride, ::stride]
    return F.interpolate(tensor.to(dtype), size=(new_h, new_w), mode='bilinear', align_corners=False)

# Post-processing functions
def apply_foreground_scale(depth: torch.Tensor, scale: float, mid: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    """
    Scale depth contrast so that:
      - depth in [0,1], where 0 = background (far), 1 = foreground (near)
      - scale > 0 : increase separation (foreground closer -> values move toward 1, background farther -> values move toward 0)
      - scale < 0 : reduce separation (flatten)
      - scale = 0 : identity

    Args:
        depth: torch.Tensor shape (..., 1) or (...), values in [0,1]
        scale: float, must be > -1.0 (we avoid scale == -1 which would divide by zero)
        mid: midpoint for separation (default 0.5)
        eps: small eps to avoid numerical issues

    Returns:
        Tensor same shape as depth, clamped to [0,1].
    """
    if not (-1.0 + 1e-12 < scale):  # avoid scale <= -1
        raise ValueError("scale must be greater than -1.0")

    depth.clamp_(0.0, 1.0)
    if abs(scale) < eps:
        return depth  # identity for scale ~ 0

    exponent = 1.0 / (1.0 + scale)  # >1 if scale<0 (flatten), <1 if scale>0 (exaggerate)
    dist = depth - mid
    out = mid + torch.sign(dist) * torch.pow(torch.abs(dist), exponent)
    return out.clamp(0.0, 1.0)
       
def anti_alias(depth: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    """
    Apply anti-aliasing to reduce jagged edges in depth maps.
    
    Args:
        depth (torch.Tensor): Normalized depth map tensor [H,W] or [B,1,H,W] with values in [0,1].
        strength (float): Blur strength; higher = smoother edges. Recommended range [0.5, 2.0].
    
    Returns:
        torch.Tensor: Smoothed depth map with same shape.
    """
    if depth.dim() == 2:
        depth = depth.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    elif depth.dim() == 3:
        depth = depth.unsqueeze(1)  # [B,1,H,W]

    # Kernel size scales with strength
    k = int(3 * strength) | 1  # force odd number
    if k < 3:
        return depth.squeeze()

    # Gaussian blur kernel
    sigma = 0.5 * strength
    coords = torch.arange(k, device=depth.device, dtype=depth.dtype) - k // 2
    gauss = torch.exp(-(coords**2) / (2 * sigma**2))
    gauss /= gauss.sum()

    # Separable convolution (X then Y)
    depth = F.conv2d(depth, gauss.view(1,1,1,-1), padding=(0, k//2), groups=1)
    depth = F.conv2d(depth, gauss.view(1,1,-1,1), padding=(k//2, 0), groups=1)

    return depth.squeeze()

def chw_tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    hwc_tensor = tensor.permute(1, 2, 0).contiguous().detach().cpu().float()
    # torch.compile can return tensor subclasses on recent PyTorch builds.
    # NumPy export only supports plain Tensor instances, so unwrap last.
    if type(hwc_tensor) is not torch.Tensor and hasattr(hwc_tensor, "as_subclass"):
        hwc_tensor = hwc_tensor.as_subclass(torch.Tensor)
    return hwc_tensor.numpy()

def apply_gamma(depth, gamma=1.2):
    return torch.pow(depth, gamma)

def normalize_tensor(depth: torch.tensor):
    if is_metric():
        depth.clamp_(min=5e-3).reciprocal_()
    
    return (depth - depth.min()) / (depth.max() - depth.min() + 1e-6) 

def _percentile_bounds_no_lerp(values: torch.Tensor, percentile: float):
    """Backend-friendly percentile bounds without torch.quantile's lerp kernel."""
    vv = values.flatten()
    n = vv.numel()
    lo_q = max(0.0, min(1.0, float(percentile) / 100.0))
    tail_count = min(n, max(1, int(round(lo_q * (n - 1))) + 1))
    if tail_count == n:
        return vv.min(), vv.max()
    lo_tail = torch.topk(vv, tail_count, largest=False, sorted=False).values
    hi_tail = torch.topk(vv, tail_count, largest=True, sorted=False).values
    return lo_tail.max(), hi_tail.min()

def _percentile_bounds_sort(values: torch.Tensor, percentile: float):
    """Fallback percentile bounds that still avoids torch.quantile."""
    vv = torch.sort(values.flatten()).values
    n = vv.numel()
    lo_q = max(0.0, min(1.0, float(percentile) / 100.0))
    hi_q = 1.0 - lo_q
    lo_idx = min(n - 1, max(0, int(round(lo_q * (n - 1)))))
    hi_idx = min(n - 1, max(0, int(round(hi_q * (n - 1)))))
    return vv[lo_idx], vv[hi_idx]

def post_process_depth(depth):
    # normalize() mirrors the official DA3 visualize_depth: it does the 1/depth
    # inversion (gated on is_metric()) + 2nd/98th percentile clip + min-max, all
    # on GPU. So no separate reciprocal_() here — that would double-invert.
    depth = normalize(depth).squeeze()
    depth = apply_gamma(depth)
    depth = apply_foreground_scale(depth, scale=FOREGROUND_SCALE)
    depth = anti_alias(depth, strength=AA_STRENGTH)
    return depth

def normalize(depth, percentile=2.0, subsample_cap=6_144):
    """GPU-tensor version of the official DA3 visualize_depth normalization.

    Mirrors models/depth_anything_3/utils/visualize.py: invert (1/depth) on the
    valid mask, clip to the 2nd/98th percentile, min-max normalize. The 1/depth
    inversion is gated on is_metric() so non-metric disparity models (already
    near=high) keep correct near/far orientation. Returns [H,W] in [0,1] oriented
    near~1, far~0 (i.e. BEFORE the official final `1-depth`)."""
    d = depth.detach().float().squeeze()
    if is_metric():
        valid = d > 0
        inv = torch.where(valid, 1.0 / d.clamp(min=1e-12), d)
        v = inv[valid]
    else:
        inv = d
        v = inv.flatten()
    if v.numel() <= 10:
        dmin = torch.zeros((), device=d.device)
        dmax = torch.zeros((), device=d.device)
    else:
        vv = v
        if vv.numel() > subsample_cap:
            step = (vv.numel() + subsample_cap - 1) // subsample_cap
            vv = vv[::step]
        try:
            dmin, dmax = _percentile_bounds_no_lerp(vv, percentile)
        except (RuntimeError, ValueError):
            dmin, dmax = _percentile_bounds_sort(vv, percentile)
    denom = (dmax - dmin).clamp_min(1e-6)
    norm = ((inv - dmin) / denom).clamp(0.0, 1.0)  # near~1, far~0
    return norm

# Load Video Depth Anything Model
def get_video_depth_anything_model(model_id=MODEL_ID):
    """ Load Video Depth Anything model from HuggingFace hub. """
    from huggingface_hub import hf_hub_download
    from models.video_depth_anything.vda2_s import VideoDepthAnything
    # Preparation for video depth anything models
    encoder_dict = {'depth-anything/Video-Depth-Anything-Small': 'vits',
                    'depth-anything/Video-Depth-Anything-Base': 'vitb',
                    'depth-anything/Video-Depth-Anything-Large': 'vitl',
                    'depth-anything/Metric-Video-Depth-Anything-Small': 'vits',
                    'depth-anything/Metric-Video-Depth-Anything-Base': 'vitb',
                    'depth-anything/Metric-Video-Depth-Anything-Large': 'vitl'}

    encoder = encoder_dict.get(model_id, 'vits')

    if 'depth-anything/video-depth-anything' in model_id.lower():
        checkpoint_name = f'video_depth_anything_{encoder}.pth'
    elif 'depth-anything/metric-video-depth-anything' in model_id.lower():
        checkpoint_name = f'metric_video_depth_anything_{encoder}.pth'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    # Load depth model without network warning when local cache exists
    try:
        checkpoint_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name, cache_dir=CACHE_PATH, local_files_only=True)
    except:
        checkpoint_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name, cache_dir=CACHE_PATH)

    model = VideoDepthAnything(**model_configs[encoder])
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True), strict=True)
    return model.to(DEVICE, dtype=DTYPE)

# Load InfiniDepth Model
def get_infinidepth_model(model_id=MODEL_ID, dtype=DTYPE):
    """ Load InfiniDepth model from HuggingFace hub. """
    # Load depth model without network warning when local cache exists
    model_path = get_model_path(model_id, cache_dir=CACHE_PATH)
    # Preparation for video depth anything models
    encoder_dict = {'lc700x/InfiniDepth-SmallPlus': 'vits16plus',
                    'lc700x/InfiniDepth-Small': 'vits16',
                    'lc700x/InfiniDepth-Base': 'vitb16',
                    'lc700x/InfiniDepth-Large': 'vitl16'}

    from models.InfiniDepth.api import InfiniDepthModel
    model = InfiniDepthModel(model_path=model_path, encoder=encoder_dict.get(model_id, 'vitl16'))
    # Honor the configured dtype. InfiniDepthModel keeps inputs consistent with
    # the loaded parameter dtype so FP16 exports do not hit input/bias mismatches.
    return model.to(DEVICE, dtype=DTYPE)

# Load Depth-Anything-V3 Model
def get_da3_model(model_id=MODEL_ID, dtype=DTYPE):
    from models.depth_anything_3.api_n import DepthAnything3
    # Load depth model without network warning when local cache exists
    try:
        model = DepthAnything3.from_pretrained(model_id, cache_dir=CACHE_PATH, dtype=dtype, local_files_only=True)
    except:
        model = DepthAnything3.from_pretrained(model_id, cache_dir=CACHE_PATH, dtype=dtype)
    return model.to(DEVICE)

# MIGraphX Optimization and Engine (ROCm7)
def _ensure_migraphx_available():
    """Check if migraphx is available."""
    try:
        import migraphx  # noqa: F401
        return True
    except ImportError:
        return False

MIGRAPHX_AVAILABLE = _ensure_migraphx_available()

def optimize_with_migraphx(onnx_path, migraphx_path):
    """
    Parse ONNX and compile with MIGraphX for AMD GPU.
    Saves compiled graph to disk for persistent caching.

    Returns path to compiled graph, or None on failure.
    """
    if not MIGRAPHX_AVAILABLE:
        raise ImportError("migraphx is not installed")

    if os.path.exists(migraphx_path) and not RECOMPILE_MIGRAPHX:
        print(f"[MIGraphX] Using cached graph: {migraphx_path}")
        return migraphx_path

    try:
        import migraphx as mx

        print("[MIGraphX] Compiling ONNX model, this may take a while...")
        prog = mx.parse_onnx(onnx_path)
        target = mx.get_target("gpu")
        force_migraphx_fp32 = (
            MODEL_ID in FORCE_FP32_KEYWORDS
            or (MODEL_ID == "Intel/dpt-beit-large-512" and DEPTH_RESOLUTION != 512)
        )
        if not force_migraphx_fp32:
            try:
                mx.autocast_fp8(prog)
                print("[MIGraphX] Quantized to FP8")
            except Exception:
                print("[MIGraphX] FP8 not available, falling back to FP16")
                mx.quantize_fp16(prog)
        prog.compile(target, offload_copy=False)
        mx.save(prog, migraphx_path)

        print(f"[MIGraphX] Graph saved to {migraphx_path}")
        return migraphx_path

    except Exception as e:
        print(f"[Error] MIGraphX optimization failed: {e}")
        return None


class MIGraphXEngine:
    """
    Wrapper around a compiled MIGraphX program (offload_copy=False).
    Zero-copy GPU path: argument_from_pointer for both input and output.
    offload_copy=False requires caller to supply GPU buffers for all params,
    including output params named '#output_N'.
    """
    def __init__(self, graph_path, device, dtype):
        import migraphx as mx
        self.mx = mx
        self.device = device
        self.dtype = dtype
        self.prog = mx.load(graph_path)
        self._logged_zero_copy = False
        param_shapes = self.prog.get_parameter_shapes()
        self.input_name = None
        self._out_params = {}
        for name, shape in param_shapes.items():
            if '#output_' in name:
                self._out_params[name] = shape
            else:
                self.input_name = name
                self._in_shape = shape
        if self.input_name is None:
            raise RuntimeError("MIGraphX graph has no input parameter")
        if not self._out_params:
            raise RuntimeError(
                "MIGraphX graph has no output parameter buffers. "
                "Recompile it with offload_copy=False or delete the cached .mgx file."
            )
        self._out_name = list(self._out_params.keys())[0]
        self._out_lens = list(self._out_params[self._out_name].lens())
        # Map MIGraphX type enum to torch dtype (half_type=1, float_type=2)
        _MX_TO_TORCH = {1: torch.float16, 2: torch.float32, 3: torch.float64}
        self._mgx_in_dtype = _MX_TO_TORCH.get(int(self._in_shape.type()), torch.float32)
        self._mgx_out_dtype = _MX_TO_TORCH.get(int(self._out_params[self._out_name].type()), torch.float32)

    def __call__(self, tensor: torch.Tensor):
        if tensor.dtype != self._mgx_in_dtype or not tensor.is_contiguous():
            tensor = tensor.contiguous().to(dtype=self._mgx_in_dtype)
        in_arg = self.mx.argument_from_pointer(self._in_shape, tensor.data_ptr())
        out = torch.empty(tuple(self._out_lens), dtype=self._mgx_out_dtype, device=self.device)
        out_arg = self.mx.argument_from_pointer(self._out_params[self._out_name], out.data_ptr())
        # Run on PyTorch's current stream so input writes, inference, and output
        # reads stay stream-ordered. Avoids blocking syncs and the FP16
        # cross-stream race in one shot.
        stream = torch.cuda.current_stream(self.device)
        self.prog.run_async({self.input_name: in_arg, self._out_name: out_arg},
                            stream.cuda_stream, "ihipStream_t")
        if not self._logged_zero_copy:
            if DEBUG:
                print(f"[MIGraphX] Zero-copy GPU path active | input={tuple(tensor.shape)} {tensor.dtype} -> output={tuple(out.shape)} {out.dtype}")
            self._logged_zero_copy = True
        return out


# TensorRT Optimization
def optimize_with_tensorrt(onnx_path, trt_path):
    """
    Convert ONNX model to TensorRT engine with fixed square input size.
    FP8 enabled, no pycuda required.
    """
    try:
        if os.path.exists(trt_path) and not RECOMPILE_TRT:
            print(f"Loaded existing TensorRT engine: {trt_path}")
            return trt_path

        import tensorrt as trt

        logger = trt.Logger(trt.Logger.ERROR)
        builder = trt.Builder(logger)

        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print("[ONNX ERROR]", parser.get_error(i))
                return None

        config = builder.create_builder_config()

        if MODEL_ID in FORCE_FP32_KEYWORDS or (MODEL_ID == "Intel/dpt-beit-large-512" and DEPTH_RESOLUTION != 512):
            # Transformer-based model: use TF32 only. 
            config.set_flag(trt.BuilderFlag.TF32)
        else:
            # FP4, FP8, FP16 ENABLED
            if not is_legacy_nvidia:
                config.set_flag(trt.BuilderFlag.FP4)
                config.set_flag(trt.BuilderFlag.FP8)
            config.set_flag(trt.BuilderFlag.FP16)
            config.set_flag(trt.BuilderFlag.TF32)
        # Optional but recommended
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
        config.set_flag(trt.BuilderFlag.REJECT_EMPTY_ALGORITHMS)

        # Workspace
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, 4 << 30
        )

        # Fixed input profile
        profile = builder.create_optimization_profile()
        input_name = network.get_input(0).name
        fixed_size = (1, 3, ENGINE_H, ENGINE_W)
        profile.set_shape(input_name, fixed_size, fixed_size, fixed_size)
        config.add_optimization_profile(profile)

        serialized_engine = builder.build_serialized_network(network, config)
        
        if serialized_engine is None:
            print("[Error] TensorRT FP8 build failed")
            return None

        with open(trt_path, "wb") as f:
            f.write(serialized_engine)

        print(f"[Main] TensorRT engine saved to {trt_path}")
        return trt_path

    except Exception as e:
        print(f"[Error] TensorRT optimization failed: {e}")
        return None
 
# Export to ONNX
class ModelForONNX(torch.nn.Module):
    """Return a single depth tensor for accelerator exports."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        if hasattr(self.model, "predict_depth"):
            return self.model.predict_depth(x)

        if (
            "video-depth-anything" in MODEL_ID.lower()
            and hasattr(self.model, "forward_features")
            and hasattr(self.model, "forward_depth")
        ):
            cur_input = x.unsqueeze(0)
            features = self.model.forward_features(cur_input)
            depth, _ = self.model.forward_depth(features, cur_input.shape)
            return depth

        try:
            out = self.model(pixel_values=x)
        except TypeError:
            out = self.model(x)

        if hasattr(out, "predicted_depth"):
            return out.predicted_depth
        if isinstance(out, dict) and "predicted_depth" in out:
            return out["predicted_depth"]
        if isinstance(out, torch.Tensor):
            return out
        if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
            return out[0]
        raise RuntimeError("Unsupported model output type for ONNX export")

@contextlib.contextmanager
def _onnx_export_safe_attention():
    if "video-depth-anything" not in MODEL_ID.lower():
        yield
        return

    toggled = []
    for module_name in (
        "models.video_depth_anything.dinov2_layers.attention",
        "models.video_depth_anything.motion_module.attention",
        "models.video_depth_anything.motion_module.motion_module",
    ):
        try:
            module = __import__(module_name, fromlist=("XFORMERS_AVAILABLE",))
        except Exception:
            continue
        if hasattr(module, "XFORMERS_AVAILABLE"):
            toggled.append((module, module.XFORMERS_AVAILABLE))
            module.XFORMERS_AVAILABLE = False

    try:
        yield
    finally:
        for module, original in toggled:
            module.XFORMERS_AVAILABLE = original

def export_to_onnx(model, output_path="depth_model.onnx", device=DEVICE, dtype=DTYPE):
    """
    Export the depth estimation model to ONNX format with fixed square input.
    """
    model_param_dtype = next(
        (p.dtype for p in model.parameters() if p.is_floating_point()),
        dtype,
    )

    # Use fixed square input size
    dummy_input = torch.randn(1, 3, ENGINE_H, ENGINE_W, device=device, dtype=model_param_dtype)
    
    input_names = ["pixel_values"]
    output_names = ["predicted_depth"]
    
    model.eval()
    export_model = ModelForONNX(model).eval()
    with torch.no_grad(), _onnx_export_safe_attention():
        torch.onnx.export(
            export_model,
            dummy_input,
            output_path,
            input_names=input_names,
            output_names=output_names,
            do_constant_folding=True,
            export_params=True,
            verbose=False,
            training=torch.onnx.TrainingMode.EVAL,
            dynamo=False,
        )
    
    if USE_TENSORRT:
        engine_name = "TensorRT"
    elif USE_OPENVINO:
        engine_name = "OpenVINO"
    elif USE_MIGRAPHX:
        engine_name = "MIGraphX"
    else:
        engine_name = "unknown"
    print(f"ONNX model generated, {engine_name} engine compiling may take a while...")

class ModelForCoreML(torch.nn.Module):
    """
    A thin wrapper that calls the original model and returns a single tensor
    (the depth tensor). This avoids ModelOutput/dict construction during tracing.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Some models expose predict_depth; prefer that when present
        if hasattr(self.model, "predict_depth"):
            out = self.model.predict_depth(x)
            if isinstance(out, torch.Tensor):
                return out
            # if predict_depth returns something else, try common fields:
            if isinstance(out, dict) and "predicted_depth" in out:
                return out["predicted_depth"]
            # fallback
            raise RuntimeError("Unsupported predict_depth return type for CoreML export")

        # Standard HF models: often accept pixel_values kwarg, but tracing a module
        # should call model directly; try both signatures.
        try:
            out = self.model(pixel_values=x)
        except TypeError:
            out = self.model(x)

        # If the model returns an object with attribute `predicted_depth`, extract it
        if hasattr(out, "predicted_depth"):
            return out.predicted_depth
        # If it's a dict (ModelOutput), attempt to pick the predicted_depth key
        if isinstance(out, dict):
            if "predicted_depth" in out:
                return out["predicted_depth"]
            # fallback: return the first tensor value found
            for v in out.values():
                if isinstance(v, torch.Tensor):
                    return v
        # If it's a tuple, assume first element is the tensor
        if isinstance(out, (tuple, list)):
            if len(out) > 0 and isinstance(out[0], torch.Tensor):
                return out[0]

        # If we reach here, we don't know how to handle output
        raise RuntimeError("Unsupported model output type for CoreML export")

def export_to_coreml(model, output_path, height, width):
    """
    Export model to CoreML via TorchScript (CPU-only, FP32).
    Uses ModelForCoreML wrapper to ensure traced graph returns a tensor (no dicts),
    and uses coreml_safe_interpolate() to replace bicubic->bilinear during tracing.
    """
    if ct is None:
        raise ImportError("coremltools must be installed on macOS to convert to CoreML")

    # Wrap to ensure a single-tensor return (no dict constructs)
    wrapped = ModelForCoreML(model).float().eval()

    # Dummy input for tracing (CPU, FP32). Height/width may differ when the
    # model uses aspect-preserving resize (InfiniDepth) instead of square.
    dummy = torch.randn(1, 3, height, width, device=DEVICE, dtype=torch.float32)

    try:
        with torch.no_grad():
            # Apply the bicubic->bilinear patch only during tracing
            with coreml_safe_interpolate():
                traced = torch.jit.trace(wrapped, dummy, strict=False)
                traced = torch.jit.freeze(traced)
    except Exception as e:
        raise RuntimeError(f"TorchScript export failed: {e}")

    # Convert to CoreML
    print("TorchScript conversion finished, CoreML compiling may take a while...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="pixel_values",
                shape=dummy.shape,
                dtype=np.float32
            )
        ],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS13
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mlmodel.save(output_path)
    print(f"[CoreML] Model saved to {output_path}")
    return output_path

class CoreMLEngine:
    def __init__(self, model_path):
        import coremltools as ct
        self.model = ct.models.MLModel(
            model_path,
            compute_units=ct.ComputeUnit.ALL
        )

    def __call__(self, tensor):
        # tensor: [1,3,H,W] torch tensor
        np_input = tensor.detach().cpu().numpy()
        out = self.model.predict({"pixel_values": np_input})

        # CoreML output names are not stable — grab the first output
        if isinstance(out, dict):
            value = next(iter(out.values()))
        else:
            value = out

        return torch.from_numpy(value)

# OpenVINO Optimization and Engine (persistent cache support)
def _ensure_openvino_cache_dir():
    """
    Return path to OpenVINO cache directory and ensure it exists.
    """
    cache_dir = os.path.join(MODEL_FOLDER, f"openvino_cache_{DEPTH_RESOLUTION}" + ("_fp16" if FP16 else "_fp32"))
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def optimize_with_openvino(onnx_path: str, device: str = None):
    """
    Load ONNX and compile with OpenVINO runtime for the selected device.
    Uses OpenVINO runtime cache directory so compiled blobs persist to disk.

    Returns compiled model (ov.CompiledModel).
    """
    if not OPENVINO_AVAILABLE:
        raise ImportError("OpenVINO runtime (openvino) is not installed")

    core = ov.Core()
    device_name = device or OPENVINO_DEVICE or "CPU"

    # Prepare cache dir for persistent compiled blobs
    cache_dir = _ensure_openvino_cache_dir()

    # If user asked to force recompile, clear cache before compiling.
    if RECOMPILE_OPENVINO:
        try:
            import shutil
            # remove content but keep the directory itself
            for entry in os.listdir(cache_dir):
                path = os.path.join(cache_dir, entry)
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    try:
                        os.remove(path)
                    except Exception:
                        pass
            print(f"[OpenVINO] Cleared cache at {cache_dir} (RECOMPILE_OPENVINO=True)")
        except Exception as e:
            print(f"[OpenVINO] Failed to clear cache: {e}")

    # Tell OpenVINO runtime to use the cache directory (persist compiled blobs)
    try:
        core.set_property({"CACHE_DIR": cache_dir})
    except Exception:
        # Older/newer OV versions might require different API; ignore if not supported
        pass

    # Read ONNX model and compile
    model = core.read_model(onnx_path)

    # compile_model will produce cache artifacts under CACHE_DIR
    compiled = core.compile_model(model, device_name)
    print(f"[OpenVINO] Compiled model for device {device_name} (cache: {cache_dir})")
    return compiled

class OpenVINOEngine:
    """
    Wrapper around an OpenVINO CompiledModel.
    Inputs: torch tensor (NCHW), returns torch tensor on target DEVICE.
    """
    def __init__(self, compiled_model, device_name: str = None, out_dtype=np.float32):
        # compiled_model: ov.CompiledModel object
        self.compiled = compiled_model
        # get input / output objects
        self.input = self.compiled.input(0)
        self.input_name = self.input.get_any_name()
        self.output = self.compiled.output(0)
        self.output_name = self.output.get_any_name()
        self.device_name = device_name or "OpenVINO"
        self.out_dtype = out_dtype

    def __call__(self, tensor: torch.Tensor):
        """
        Accepts a torch tensor [1,3,H,W] dtype float32/float16,
        runs inference via OpenVINO, returns torch tensor on DEVICE/dtype MODEL_DTYPE.
        """
        # Ensure numpy input on CPU (OpenVINO runtime runs on host)
        arr = tensor.detach().cpu().numpy()

        # Enforce common float dtypes
        if arr.dtype not in (np.float32, np.float16):
            arr = arr.astype(np.float32)

        # Some compiled models accept a single numpy input or a dict keyed by input name
        try:
            result = self.compiled({self.input_name: arr})
        except TypeError:
            # fallback to positional call
            result = self.compiled(arr)

        # Extract output
        if isinstance(result, dict):
            out_np = result.get(self.output_name, next(iter(result.values())))
        elif isinstance(result, (list, tuple)):
            out_np = result[0]
        else:
            try:
                out_np = next(iter(result.values()))
            except Exception:
                raise RuntimeError("Unexpected OpenVINO compiled model output type")

        # Convert to torch and move to your DEVICE with MODEL_DTYPE
        out_torch = torch.from_numpy(out_np.copy())
        try:
            out_torch = out_torch.to(device=DEVICE, dtype=MODEL_DTYPE)
        except Exception:
            # If moving to device fails, return CPU tensor
            pass

        return out_torch

# TensorRT Engine Wrapper Class (Without PyCUDA)
class TensorRTEngine:
    def __init__(self, engine_path, device, dtype):
        """
        Initialize TensorRT engine using binding names instead of deprecated methods.
        """
        self.device = device
        self.dtype = dtype
        
        try:
            import tensorrt as trt
            
            # Load TensorRT engine
            with open(engine_path, "rb") as f:
                engine_data = f.read()
            
            logger = trt.Logger(trt.Logger.ERROR)
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(engine_data)
            self.context = self.engine.create_execution_context()
            
            # Get binding information using names instead of deprecated methods
            self.input_binding_indices = []
            self.output_binding_indices = []
            
            for binding in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(binding)
                
                # Use name pattern matching to identify inputs/outputs
                if "input" in name.lower() or "pixel_values" in name.lower():
                    self.input_binding_indices.append(binding)
                else:
                    self.output_binding_indices.append(binding)
            
            # Pre-allocate output tensors
            self.output_shapes = {}
            for binding in self.output_binding_indices:
                name = self.engine.get_tensor_name(binding)
                self.output_shapes[name] = self.engine.get_tensor_shape(name)
            
        except ImportError:
            raise ImportError("TensorRT not available")

    def __call__(self, tensor):
        """Execute inference with TensorRT using the TRT 10 native API."""
        # Input must be contiguous on the correct device
        tensor = tensor.contiguous().to(device=self.device, dtype=self.dtype)

        # Set input shape and address
        input_name = self.engine.get_tensor_name(0)
        self.context.set_input_shape(input_name, tuple(tensor.shape))
        self.context.set_tensor_address(input_name, tensor.data_ptr())

        # Allocate output tensors and set their addresses
        outputs = {}
        for binding in self.output_binding_indices:
            name = self.engine.get_tensor_name(binding)
            dims = self.context.get_tensor_shape(name)
            out = torch.empty(tuple(dims), device=self.device, dtype=self.dtype)
            outputs[name] = out
            self.context.set_tensor_address(name, out.data_ptr())

        # Execute on the current CUDA stream and wait for completion
        stream = torch.cuda.current_stream(self.device)
        self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        stream.synchronize()

        return outputs['predicted_depth']

    def close(self):
        for attr in ("context", "engine"):
            try:
                obj = getattr(self, attr, None)
                if obj is not None:
                    del obj
            except Exception:
                pass
            try:
                setattr(self, attr, None)
            except Exception:
                pass

# Model Wrapper Class
class DepthModelWrapper:
    def __init__(self, model_path, device, device_info, dtype, size=None,
                 onnx_path=None, trt_path=None, migraphx_path=None):
        """
        Wrapper class that handles both PyTorch and TensorRT/MIGraphX backends.
        """
        self.device = device
        self.device_info = device_info
        self.dtype = dtype
        self.model_path = model_path
        self.onnx_path = onnx_path
        self.trt_path = trt_path
        self.migraphx_path = migraphx_path
        self.size = size
        self.use_torch_compile = USE_TORCH_COMPILE

        # Determine backend based on device
        self.is_nvidia = IS_NVIDIA
        self.is_rocm = IS_AMD_ROCM
        self.is_mps = IS_MPS
        self.is_xpu = IS_XPU
        self.is_cpu = IS_CPU

        # Load PyTorch now; defer the accelerated backend until the first frame
        # gives us the fixed engine shape (see build_accelerated_backend).
        self._built = False
        self._pending_backend = None

        if self.is_mps and USE_COREML:
            self._pending_backend = "CoreML"
        elif USE_OPENVINO and OPENVINO_AVAILABLE and OPENVINO_DEVICE is not None:
            self._pending_backend = "OpenVINO"
        elif self.is_rocm and USE_MIGRAPHX and MIGRAPHX_AVAILABLE:
            self._pending_backend = "MIGraphX"
        elif self.is_nvidia and USE_TENSORRT:
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            self._pending_backend = "TensorRT"
        else:
            # Ignore specific warning message
            warnings.filterwarnings("ignore", message="User provided device_type of 'cuda', but CUDA is not available")

        # Provisional PyTorch backend (also the fallback if engine build fails).
        self.backend = "PyTorch"
        self.model = self._load_pytorch_model()
        # Accelerated backends announce themselves from build_accelerated_backend().
        if self._pending_backend is None:
            print(f"Using backend: {self.backend}")

    def build_accelerated_backend(self, engine_h, engine_w, onnx_path, trt_path, coreml_path, migraphx_path=None):
        """Compile the deferred accelerated backend for the known engine shape;
        falls back to PyTorch on failure. Runs at most once."""
        if self._built or self._pending_backend is None:
            self._built = True
            return

        self.onnx_path = onnx_path
        self.trt_path = trt_path
        self.migraphx_path = migraphx_path
        try:
            if self._pending_backend == "TensorRT":
                engine = self._load_tensorrt_engine()
                if engine is not None:
                    self.model = engine
                    self.backend = "TensorRT"
                else:
                    print("[Error] TensorRT failed, keeping PyTorch")
            elif self._pending_backend == "MIGraphX":
                engine = self._load_migraphx_engine()
                if engine is not None:
                    self.model = engine
                    self.backend = "MIGraphX"
                else:
                    print("[Error] MIGraphX failed, keeping PyTorch")
            elif self._pending_backend == "OpenVINO":
                self.model = self._load_openvino_engine()
                self.backend = "OpenVINO"
            elif self._pending_backend == "CoreML":
                global COREML_PATH
                COREML_PATH = coreml_path
                self.model = self._load_coreml_model(engine_h, engine_w)
                self.backend = "CoreML"
        except Exception as e:
            print(f"[Error] {self._pending_backend} initialization failed: {str(e)}, falling back to PyTorch")
            self.backend = "PyTorch"

        print(f"Using backend: {self.backend}")
        self._built = True

    def _load_pytorch_model(self, enable_trt=USE_TENSORRT, enable_migraphx=USE_MIGRAPHX):
        """Load the original PyTorch model."""
        global DTYPE
        DTYPE = self.dtype

        # Load model
        if 'video-depth-anything' in MODEL_ID.lower():
            model = get_video_depth_anything_model(MODEL_ID)
        elif 'da3' in MODEL_ID.lower():
            model = get_da3_model(MODEL_ID, dtype=self.dtype)
        elif 'infinidepth' in MODEL_ID.lower():
            model = get_infinidepth_model(MODEL_ID, dtype=self.dtype)
        else:
            try:
                # Load depth model without network warning when local cache exists
                try:
                    model = AutoModelForDepthEstimation.from_pretrained(
                        MODEL_ID,
                        dtype=self.dtype,
                        cache_dir=CACHE_PATH,
                        weights_only=True,
                        local_files_only=True
                    ).to(DEVICE)
                except Exception:
                    model = AutoModelForDepthEstimation.from_pretrained(
                        MODEL_ID,
                        dtype=self.dtype,
                        cache_dir=CACHE_PATH,
                        weights_only=True
                    ).to(DEVICE)
            except Exception as e:
                print(f"[Error]: Failed to load model, please check your local model file and network connection. Details: {e}")

        if self.dtype == torch.float16:
            model.half()

        model = model.eval()
        if self.is_nvidia and self.use_torch_compile and not enable_trt:
            enable_torch_compile_fallback(torch)
            compiled_model = torch_compile_or_original(torch, model, "depth model")
            if compiled_model is not model:
                print("Processing torch.compile with Triton, it may take a while...")
            model = compiled_model
        elif self.is_rocm and self.use_torch_compile and not enable_migraphx:
            enable_torch_compile_fallback(torch)
            compiled_model = torch_compile_or_original(torch, model, "depth model")
            if compiled_model is not model:
                print("Processing torch.compile with Triton, it may take a while...")
            model = compiled_model

        return model
    
    def _load_coreml_model(self, engine_h, engine_w):
        pytorch_model = self._load_pytorch_model()

        if not os.path.exists(COREML_PATH) or RECOMPILE_COREML:
            export_to_coreml(
                pytorch_model,
                COREML_PATH,
                engine_h,
                engine_w,
            )

        return CoreMLEngine(COREML_PATH)

    def _load_openvino_engine(self):
        """
        Ensure ONNX exists, then compile & return an OpenVINOEngine.
        """
        # export to ONNX if not exists
        if RECOMPILE_OPENVINO or not os.path.exists(self.onnx_path):
            # need a PyTorch model to export
            pytorch_model = self._load_pytorch_model()
            export_to_onnx(pytorch_model, self.onnx_path, self.device, self.dtype)
        
        device_name = OPENVINO_DEVICE or "CPU"
        compiled = optimize_with_openvino(self.onnx_path, device=device_name)
        return OpenVINOEngine(compiled, device_name=device_name)
    
    def _load_tensorrt_engine(self):
        """Load or create TensorRT engine."""
        # First, load PyTorch model to export ONNX
        pytorch_model = self._load_pytorch_model()
        
        # Export to ONNX if not exists
        if RECOMPILE_TRT or not os.path.exists(self.onnx_path):
            export_to_onnx(pytorch_model, self.onnx_path, self.device, self.dtype)
        
        # Build or load TensorRT engine
        trt_engine_path = optimize_with_tensorrt(self.onnx_path, self.trt_path)
        if trt_engine_path is None:
            return None
        try:
            return TensorRTEngine(trt_engine_path, self.device, self.dtype)
        except Exception as e:
            print(f"[Error] TensorRT engine loading failed: {str(e)}")
            return None

    def _load_migraphx_engine(self):
        """Load or create MIGraphX engine for AMD ROCm7."""
        # Export to ONNX if not exists
        if RECOMPILE_MIGRAPHX or not os.path.exists(self.onnx_path):
            # Only load PyTorch for export. Cached ONNX/MGX runs avoid the extra
            # model load and stay on the compiled zero-copy path sooner.
            pytorch_model = self._load_pytorch_model()
            export_to_onnx(pytorch_model, self.onnx_path, self.device, self.dtype)

        # Build or load MIGraphX graph
        migraphx_graph_path = optimize_with_migraphx(self.onnx_path, self.migraphx_path)
        if migraphx_graph_path is None:
            return None
        try:
            return MIGraphXEngine(migraphx_graph_path, self.device, self.dtype)
        except Exception as e:
            print(f"[Error] MIGraphX engine loading failed: {str(e)}")
            return None

    def __call__(self, tensor):
        """Run inference using the active backend."""
        # Fast path for OpenVINO / MIGraphX backends (they handle their own device/dtype)
        if getattr(self, "backend", None) in ("OpenVINO", "MIGraphX"):
            return self.model(tensor)

        """Run inference using the active backend."""
        with torch.no_grad():
            with maybe_autocast(self.device):
                if self.backend == "PyTorch":
                    if "video-depth-anything" in MODEL_ID.lower():
                        return self.model(pixel_values=tensor, fp32=not FP16)
                    elif "da3" in MODEL_ID.lower() or "infinidepth" in MODEL_ID.lower():
                        return self.model.predict_depth(tensor, fp32=not FP16)
                    else:
                        return self.model(pixel_values=tensor).predicted_depth
                else:
                    # TensorRT, CoreML
                    return self.model(tensor)

# Initialize model wrapper
model_wraper = DepthModelWrapper(
    model_path=MODEL_ID,
    device=DEVICE,
    device_info=DEVICE_INFO,
    dtype=DTYPE
)

MODEL_DTYPE = next(model_wraper.model.parameters()).dtype if hasattr(model_wraper.model, 'parameters') else DTYPE
if "depthpro"  in MODEL_ID.lower() or "zoedepth" in MODEL_ID.lower() or "dpt" in MODEL_ID.lower():
    MEAN = torch.tensor([0.5,0.5,0.5], dtype=MODEL_DTYPE, device=DEVICE).view(1,3,1,1)
    STD = torch.tensor([0.5,0.5,0.5], dtype=MODEL_DTYPE, device=DEVICE).view(1,3,1,1)
else:    
    MEAN = torch.tensor([0.485,0.456,0.406], dtype=MODEL_DTYPE, device=DEVICE).view(1,3,1,1)
    STD = torch.tensor([0.229,0.224,0.225], dtype=MODEL_DTYPE, device=DEVICE).view(1,3,1,1)
    
if USE_TORCH_COMPILE and IS_CUDA:
    try:
        # Compile the model as before, but SKIP compiling lightweight post-processing functions，avoid FX re-tracing conflicts. These are fast without it.  
        enable_torch_compile_fallback(torch)
        post_process_depth = torch_compile_with_runtime_fallback(torch, post_process_depth, "post_process_depth")
        # Assign to a global or module-level var if needed for access
        globals()['post_process_depth'] = post_process_depth  # Or use a class/module attribute
        
    except Exception as e:
        print(f"[Warning] torch.compile failed: {str(e)}, running without it.")

# Initialize with dummy input for warmup
def warmup_model(model_wraper, engine_h, engine_w, steps: int = 3):
    """Warmup with the fixed model-input shape (engine_h, engine_w)."""
    with maybe_autocast(DEVICE):
        for i in range(steps):
            dummy = torch.randn(1, 3, engine_h, engine_w,
                               device=DEVICE)
            model_wraper(dummy)
    return True

lock = Lock()

# Build the accelerated engine + warmup on the first frame, once the shape is known.
_engine_ready = False
def _ensure_engine_built(engine_h, engine_w):
    global _engine_ready, ENGINE_H, ENGINE_W, ONNX_PATH, TRT_PATH, COREML_PATH, MIGRAPHX_PATH
    if _engine_ready:
        return
    with lock:
        if _engine_ready:
            return
        ENGINE_H, ENGINE_W = engine_h, engine_w
        if engine_h == DEPTH_RESOLUTION and engine_w == DEPTH_RESOLUTION:
            ONNX_PATH       = os.path.join(MODEL_FOLDER, f"model_{DTYPE_INFO}_{DEPTH_RESOLUTION}.onnx")
            TRT_PATH        = os.path.join(MODEL_FOLDER, f"model_{DTYPE_INFO}_{DEPTH_RESOLUTION}.trt")
            COREML_PATH     = os.path.join(MODEL_FOLDER, f"model_{DTYPE_INFO}_{DEPTH_RESOLUTION}.mlpackage")
            MIGRAPHX_PATH   = os.path.join(MODEL_FOLDER, f"model_{DTYPE_INFO}_{DEPTH_RESOLUTION}_gpu.mgx")
        else:
            ONNX_PATH       = os.path.join(MODEL_FOLDER, f"model_{DTYPE_INFO}_{engine_h}x{engine_w}.onnx")
            TRT_PATH        = os.path.join(MODEL_FOLDER, f"model_{DTYPE_INFO}_{engine_h}x{engine_w}.trt")
            COREML_PATH     = os.path.join(MODEL_FOLDER, f"model_{DTYPE_INFO}_{engine_h}x{engine_w}.mlpackage")
            MIGRAPHX_PATH   = os.path.join(MODEL_FOLDER, f"model_{DTYPE_INFO}_{engine_h}x{engine_w}_gpu.mgx")
        model_wraper.build_accelerated_backend(engine_h, engine_w, ONNX_PATH, TRT_PATH, COREML_PATH, MIGRAPHX_PATH)
        warmup_model(model_wraper, engine_h, engine_w, steps=3)
        _engine_ready = True

# Temporal depth stabilizer (EMA)
class DepthStabilizer:
    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.prev = None
        self.enabled = True
        self.lock = Lock()

    def __call__(self, depth: torch.Tensor):
        if not self.enabled:
            return depth
        with self.lock:
            if self.prev is None or self.prev.shape != depth.shape or self.prev.device != depth.device:
                self.prev = depth.detach().clone()
                return depth
            out = self.alpha * self.prev + (1.0 - self.alpha) * depth
            self.prev = out.detach().clone()
            return out

depth_stabilizer = DepthStabilizer(alpha=0.9)  # increase alpha for more stability
if USE_TORCH_COMPILE and IS_CUDA:
    enable_torch_compile_fallback(torch)
    depth_stabilizer = torch_compile_with_runtime_fallback(
        torch, depth_stabilizer.__call__, "depth_stabilizer"
    )

# Modified predict_depth function with improved TRT integration
def predict_depth(image_rgb, return_tuple=False, use_temporal_smooth: bool = True, dtype=DTYPE):
    """
    Returns depth in [0,1] using fixed square input.
    """
    
    # Use fixed square size
    target_size = DEPTH_RESOLUTION
    patch = get_patch_size()

    # Check image_rgb is a tensor or numpy array
    if isinstance(image_rgb, torch.Tensor):
        rgb_tensor = image_rgb.to(device=DEVICE)
        h, w = rgb_tensor.shape[1:]
    else:
        # Convert NumPy -> Torch tensor and move to device early
        h, w = image_rgb.shape[:2]
        rgb_tensor = torch.from_numpy(image_rgb).to(device=DEVICE, non_blocking=True).permute(2, 0, 1).contiguous()  # [C, H, W]

    tensor = rgb_tensor.unsqueeze(0)  # [1, C, H, W], still uint8

    if patch is not None:
        # Aspect-preserve longest-side resize + patch alignment in one interpolate.
        # _resize_patch_aligned_t casts to float, so downstream math runs on the
        # small model-input tensor, not the full-res frame.
        tensor = _resize_patch_aligned_t(tensor, target_size, patch)
        tensor = tensor.to(dtype=MODEL_DTYPE) / 255.0
    else:
        # Fixed square resize for models hardcoded to a square input (DepthPro).
        if (h, w) != (target_size, target_size):
            with coreml_safe_interpolate():
                tensor = F.interpolate(
                    tensor.to(dtype=MODEL_DTYPE),
                    size=(target_size, target_size),
                    mode='bilinear',
                    align_corners=False
                ) / 255.0
        else:
            tensor = tensor.to(dtype=MODEL_DTYPE) / 255.0

    # InfiniDepth normalizes internally.
    if "infinidepth" not in MODEL_ID.lower():
        tensor = (tensor - MEAN) / STD
    tensor = tensor.contiguous()

    # Tensor now has the final model-input shape; build engine + warmup once.
    if not _engine_ready:
        _, _, eh, ew = tensor.shape
        _ensure_engine_built(eh, ew)

    # Model inference with appropriate autocast context
    depth = model_wraper(tensor)
            
    # POST-PROCESSING
    with torch.no_grad():
        depth = post_process_depth(depth)

        # Optional temporal stabilization (EMA)
        if use_temporal_smooth:
            depth = depth_stabilizer(depth)

    # Resize depth back to original input resolution (on GPU)# Resize depth back to original input resolution (on GPU)
    with coreml_safe_interpolate():
        depth = F.interpolate(
            depth.unsqueeze(0).unsqueeze(0),  # [1, 1, target_size, target_size]
            size=(h, w),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).squeeze(0)

    # Return
    if return_tuple:
        return depth, rgb_tensor
    else:
        return depth
# Global cache (module-level)
_FONT_CACHE = {}

def build_font(device="cpu", dtype=torch.float32):
    """
    Build (once) and cache FPS font tensors per (device, dtype).
    Safe for CUDA / DirectML / MPS / CPU.
    """
    key = (str(device), dtype)

    if key in _FONT_CACHE:
        return _FONT_CACHE[key]

    # Build once
    chars = sorted(font_dict.keys())

    font_tensor = torch.stack([
        torch.tensor(
            [[1.0 if c == "1" else 0.0 for c in row] for row in font_dict[ch]],
            device=device,
            dtype=dtype,
        )
        for ch in chars
    ])  # [num_chars, 5, 3]

    _FONT_CACHE[key] = (chars, font_tensor)
    return chars, font_tensor

# module-level cache
_FPS_MASK_CACHE = {
    "mask": None,
    "frame": 0,
    "interval": 10,  # update every N frames
}

def overlay_fps(rgb: torch.Tensor, fps: float):
    device, dtype = rgb.device, rgb.dtype
    H, W = rgb.shape[1:]

    cache = _FPS_MASK_CACHE
    cache["frame"] += 1

    # Rebuild only every N frames
    if cache["mask"] is None or cache["frame"] % cache["interval"] == 0:
        chars, font_tensor = build_font(device, dtype)
        txt = f"FPS: {fps:.1f}"

        idxs = torch.tensor(
            [chars.index(ch) if ch in chars else chars.index(" ") for ch in txt],
            device=device
        )

        scale = max(1, min(8, H // 60))
        char_h, char_w = 5 * scale, 3 * scale
        spacing = scale
        margin_x, margin_y = 2 * scale, 2 * scale

        glyphs = font_tensor[idxs]
        glyphs = glyphs.repeat_interleave(scale, 1).repeat_interleave(scale, 2)

        mask = torch.zeros((H, W), device=device, dtype=dtype)

        for i, glyph in enumerate(glyphs):
            x0 = margin_x + i * (char_w + spacing)
            y0 = margin_y
            x1 = min(W, x0 + char_w)
            y1 = min(H, y0 + char_h)
            if x0 < W and y0 < H:
                mask[y0:y1, x0:x1] = torch.maximum(
                    mask[y0:y1, x0:x1],
                    glyph[:y1 - y0, :x1 - x0]
                )

        cache["mask"] = mask

    alpha = cache["mask"].unsqueeze(0)
    color = torch.tensor([0.0, 255.0, 0.0], device=device, dtype=dtype).view(3,1,1)
    return rgb * (1 - alpha) + color * alpha

# Aspect pad helper
def pad_to_aspect_tensor(tensor, target_ratio=(16, 9)):
    _, h, w = tensor.shape
    t_w, t_h = target_ratio
    r_img, r_t = w / h, t_w / t_h
    if abs(r_img - r_t) < 1e-3:
        return tensor
    if r_img > r_t:  # too wide -> pad height
        new_h = int(round(w / r_t))
        pad_top = (new_h - h) // 2
        return F.pad(tensor, (0, 0, pad_top, new_h - h - pad_top))
    else:  # too tall -> pad width
        new_w = int(round(h * r_t))
        pad_left = (new_w - w) // 2
        return F.pad(tensor, (pad_left, new_w - w - pad_left, 0, 0))

# generate left and right eye view for streamer 
def make_sbs_core(rgb: torch.Tensor,
                  depth: torch.Tensor,
                  ipd_uv=0.064,
                  depth_ratio=2.0,
                  display_mode="Half-SBS",
                  fill_16_9=False,
                  convergence=0.0,
                  device=DEVICE) -> torch.Tensor:
    """
    Core tensor operations for side-by-side stereo.
    Keeps CUDA fast path (grid_sample) and fallback path (gather).
    Compatible with torch.compile.
    Inputs:
        rgb: [C,H,W] float tensor
        depth: [H,W] float tensor
    Returns:
        SBS image [C,H,W] float tensor (0-255 range)
    """
    C, H, W = rgb.shape
    img = rgb.unsqueeze(0).clamp(0, 255)  # [1,C,H,W]
    depth_strength = 0.05
    depth = depth - convergence
    inv = - depth * depth_ratio
    max_px = ipd_uv * W
    shifts = inv * max_px * depth_strength
    
    with maybe_autocast(device):
        # CUDA fast path: grid_sample
        if not IS_DIRECTML:
            xs = torch.linspace(-1.0, 1.0, W, device=device).view(1, 1, W).expand(1, H, W)
            ys = torch.linspace(-1.0, 1.0, H, device=device).view(1, H, 1).expand(1, H, W)
            shift_norm = shifts * (2.0 / (W - 1))
            grid_left = torch.stack([xs + shift_norm, ys], dim=-1)
            grid_right = torch.stack([xs - shift_norm, ys], dim=-1)
            left = F.grid_sample(img, grid_left, mode="bilinear",
                                 padding_mode="reflection", align_corners=True)[0]
            right = F.grid_sample(img, grid_right, mode="bilinear",
                                  padding_mode="reflection", align_corners=True)[0]
        # Fallback path: vectorized gather (DirectML)
        else:
            base = torch.arange(W, device=device, dtype=torch.int64).view(1, -1).expand(H, -1)
            shifts = shifts.to(dtype=torch.float32)
            coords_left = (base.to(dtype=torch.float32) + shifts).clamp(0, W - 1).long()
            coords_right = (base.to(dtype=torch.float32) - shifts).clamp(0, W - 1).long()
            # Left eye
            gather_idx_left = coords_left.unsqueeze(0).expand(C, H, W).unsqueeze(0)
            left = torch.gather(img.expand(1, C, H, W), 3, gather_idx_left)[0]
            # Right eye
            gather_idx_right = coords_right.unsqueeze(0).expand(C, H, W).unsqueeze(0)
            right = torch.gather(img.expand(1, C, H, W), 3, gather_idx_right)[0]
    
    # Aspect pad & arrange SBS/TAB
    if fill_16_9:
        left = pad_to_aspect_tensor(left)
        right = pad_to_aspect_tensor(right)
    if display_mode in ["Half-TAB", "Full-TAB"]:
        out = torch.cat([left, right], dim=1)
    else:
        out = torch.cat([left, right], dim=2)
    if display_mode not in ["Full-SBS", "Full-TAB"]:
        out = F.interpolate(out.unsqueeze(0), size=left.shape[1:], mode="area")[0]
    return out.clamp(0, 255)

def make_sbs(rgb_c, depth, ipd_uv=0.064, depth_ratio=2.0, convergence=0.0, fill_16_9=False, display_mode="Half-SBS", fps=None):
    """
    Full function: adds optional FPS overlay and converts output to numpy uint8.
    Calls `make_sbs_core` for tensor computations (torch.compile compatible).
    """
    if USE_COREML:
        device = DEVICE
        if isinstance(depth, np.ndarray):
            depth = torch.from_numpy(depth).to(device=DEVICE, dtype=DTYPE)
        else:
            depth = depth.to(device=device)
    
    # Handle input type conversion
    if isinstance(rgb_c, np.ndarray):
        # Convert numpy array to tensor with matching device/dtype
        rgb = torch.from_numpy(rgb_c).to(device=depth.device, dtype=depth.dtype)
        # Convert from HWC to CHW format
        if rgb.ndim == 3 and rgb.shape[2] == 3:
            rgb = rgb.permute(2, 0, 1).contiguous()
    else:
        # Ensure tensor is on correct device and dtype
        rgb = rgb_c.to(device=depth.device, dtype=depth.dtype)

    # Optional FPS overlay can stay in Python side (avoids torch.compile recompiles)
    if fps is not None:
        rgb = overlay_fps(rgb, fps)  # your existing overlay function

    sbs_tensor = make_sbs_core(
        rgb=rgb, 
        depth=depth, 
        ipd_uv=ipd_uv, 
        depth_ratio=depth_ratio, 
        convergence=convergence, 
        fill_16_9=fill_16_9,
        display_mode=display_mode
    )

    return chw_tensor_to_numpy(sbs_tensor)

if USE_TORCH_COMPILE and IS_CUDA:
    enable_torch_compile_fallback(torch)
    make_sbs_core = torch_compile_with_runtime_fallback(torch, make_sbs_core, "make_sbs_core")


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt

    base_name = os.path.splitext(os.path.basename("assets/cats.jpg"))[0]
    model_short = MODEL_ID.split("/")[-1]
    tag = f"{model_short}_{DEPTH_RESOLUTION}"
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)

    import time
    # Trigger engine build on first call
    depth = predict_depth(image_rgb, dtype=DTYPE)

    # FPS benchmark: 5 warmup + 30 timed iterations
    BENCH_N = 30
    for _ in range(5):
        predict_depth(image_rgb, dtype=DTYPE)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    for _ in range(BENCH_N):
        predict_depth(image_rgb, dtype=DTYPE)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - t0
    print(f"[Benchmark] backend={model_wraper.backend} | {BENCH_N} iters | {elapsed:.2f}s | {BENCH_N/elapsed:.1f} FPS")

    sbs = make_sbs(image_rgb, depth, ipd_uv=0.064, depth_ratio=2.0,
                   convergence=-0.5, display_mode="Half-SBS", fps=None, fill_16_9=False)

    # Inspection only: apply the official `1-depth` orientation + Spectral cmap.
    depth_vis = (1.0 - depth).detach().float().cpu().numpy()
    depth_path = os.path.join(out_dir, f"{base_name}_{tag}_depth.png")
    plt.imshow(depth_vis, cmap='Spectral')
    plt.colorbar()
    plt.show()
    plt.imsave(depth_path, depth_vis, cmap='Spectral')
    print(f"Saved: {depth_path}")
    plt.close()

    # Show and save SBS
    sbs_display = np.clip(sbs, 0, 255).astype(np.uint8)
    plt.imshow(sbs_display)
    plt.axis('off')
    plt.show()
    sbs_path = os.path.join(out_dir, f"{base_name}_{tag}_sbs.png")
    plt.imsave(sbs_path, sbs_display)
    print(f"Saved: {sbs_path}")
    plt.close()
