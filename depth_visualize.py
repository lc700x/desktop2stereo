# depth.py
import torch, cv2
torch.set_num_threads(1)
import torch.nn.functional as F
from transformers import AutoModelForDepthEstimation
import numpy as np
from threading import Lock
from PIL import Image
img  = Image.open("assets/test.png").convert("RGB")
image_rgb = np.array(img)
AA_STRENTH = 4
FP16 = True
DTYPE = torch.float16 if FP16 else torch.float32
CACHE_PATH = "models"
DEVICE_ID = 0
MODEL_ID = "depth-anything/Video-Depth-Anything-Small"
# MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_RESOLUTION = 518
FOREGROUND_SCALE = -1

def get_device(index=0):
    try:
        try:
            import torch_directml
            if torch_directml.is_available():
                return torch_directml.device(index), f"Using DirectML device: {torch_directml.device_name(index)}"
        except ImportError:
            pass
        if torch.backends.mps.is_available() and index==0:
            return torch.device("mps"), "Using Apple Silicon (MPS) device"
        if torch.cuda.is_available():
            return torch.device("cuda"), f"Using CUDA device: {torch.cuda.get_device_name(index)}"
        else:
            return torch.device("cpu"), "Using CPU device"
    except:
        return torch.device("cpu"), "Using CPU device"
    
DEVICE, DEVICE_INFO = get_device(DEVICE_ID)
print(DEVICE_INFO)
print(f"Model: {MODEL_ID}")

# Optimization for CUDA
if "CUDA" in DEVICE_INFO:
    torch.backends.cudnn.benchmark = True
    if not FP16:
        # Enable TF32 for matrix multiplications
        torch.backends.cuda.matmul.allow_tf32 = True
        # Enable TF32 for cuDNN (convolution operations)
        torch.backends.cudnn.allow_tf32 = True
        # Enable TF32 matrix multiplication for better performance
        torch.set_float32_matmul_precision('high')
    else:
        torch.set_autocast_enabled(True)
    # Enable math attention
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

# Model configuration
DTYPE = torch.float16 if FP16 else torch.float32
# Folder to store compiled model / cache 
DTYPE_INFO = "fp16" if FP16 else "fp32"


# Post-processing functions
def apply_stretch(x: torch.Tensor, low: float = 2.0, high: float = 98.0) -> torch.Tensor:
    """
    Percentile-based clipping + normalization.
    Fully DirectML compatible (no torch.clamp).
    """
    x = x.to(DTYPE)
    if x.numel() < 2:
        return x

    # Downsample to reduce cost
    x_sampled = x[::8, ::8] if x.dim() == 2 else x.flatten()[::8]
    flat = x_sampled.flatten()

    if flat.numel() < 2:
        return x

    # Sort values
    vals, _ = torch.sort(flat)

    # Compute percentile indices
    k_low = int((low / 100.0) * (vals.numel() - 1))
    k_high = int((high / 100.0) * (vals.numel() - 1))
    k_low = max(0, min(k_low, vals.numel() - 1))
    k_high = max(0, min(k_high, vals.numel() - 1))

    lo = vals[k_low]
    hi = vals[k_high]

    # Avoid divide-by-zero
    scale = hi - lo
    if scale <= 0:
        return torch.zeros_like(x)

    # Manual clamp: DirectML supports min/max
    x = torch.maximum(x, lo)
    x = torch.minimum(x, hi)

    return (x - lo) / (scale + 1e-6)

def apply_gamma(depth: torch.Tensor, gamma: float = 0.8) -> torch.Tensor:
    """
    Apply gamma correction to exaggerate depth differences.
    Here 1=near, 0=far.
    gamma < 1 -> expand far (background)
    gamma > 1 -> expand near (foreground)
    """
    depth = torch.clamp(depth, 0.0, 1.0)
    return depth.pow(gamma)

def apply_sigmoid(depth: torch.Tensor, k: float = 10.0, midpoint: float = 0.5) -> torch.Tensor:
    """
    Apply sigmoid mapping to emphasize mid-range depth.
    Larger k makes it steeper.
    1=near, 0=far convention.
    """
    depth = torch.clamp(depth, 0.0, 1.0)
    return 1.0 / (1.0 + torch.exp(-k * (depth - midpoint)))

def apply_foreground_scale(depth: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Apply foreground/background scaling to a depth map.

    Args:
        depth (torch.Tensor): depth map of shape (H, W, 1), values normalized in [0, 1].
                              0 = near (foreground), 1 = far (background).
        scale (float): scaling factor
                       > 0: foreground closer, background further
                       < 0: foreground flatter, background closer
                       = 0: no change

    Returns:
        torch.Tensor: scaled depth map of same shape as input
    """
    if scale > 0:
        # Exaggerate separation: foreground closer, background further
        return torch.pow(depth, 1.0 + scale)
    elif scale < 0:
        # Compress foreground, pull background closer
        return 1.0 - torch.pow(1.0 - depth, 1.0 + abs(scale))
    else:
        return depth
    
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

def process_tensor(img_rgb: np.ndarray, height) -> torch.Tensor:
    """
    Convert BGR/UMat numpy image to normalized GPU tensor in model dtype.
    Keeps transfers efficient and uses non_blocking for pinned tensors where possible.
    """
    if isinstance(img_rgb, cv2.UMat):
        img_rgb = img_rgb.get()

    h0, w0 = img_rgb.shape[:2]
    if height < h0:
        width = int(img_rgb.shape[1] / h0 * height)
        img_rgb = cv2.resize(img_rgb, (width, height), interpolation=cv2.INTER_AREA)

    # Ensure contiguous numpy array (uint8)
    np_img = np.ascontiguousarray(img_rgb)
    # convert to torch tensor on CPU (uint8) then float on device to avoid double copy
    t_cpu = torch.from_numpy(np_img)  # shape H,W,C dtype=uint8
    # move to device and convert in one step
    t = t_cpu.permute(2, 0, 1).contiguous().unsqueeze(0).to(device=DEVICE, dtype=MODEL_DTYPE)
    t = t / 255.0
    return t

def process(img_rgb: np.ndarray, height) -> np.ndarray:
    """
    Resize BGR/UMat numpy image to target height, keeping aspect ratio.
    """
    if isinstance(img_rgb, cv2.UMat):
        img_rgb = img_rgb.get()
    h0 = img_rgb.shape[0]
    if height < h0:
        width = int(img_rgb.shape[1] / h0 * height)
        img_rgb = cv2.resize(img_rgb, (width, height), interpolation=cv2.INTER_AREA)
    return img_rgb

def normalize_tensor(tensor):
    """ Normalize tensor to [0,1] """
    return (tensor - tensor.min())/(tensor.max() - tensor.min()+1e-6)

def apply_piecewise(
    depth: torch.Tensor,
    split: float = 0.5,
    near_gamma: float = 2.0,
    far_gamma: float = 0.8
    ) -> torch.Tensor:
    """
    Efficient piecewise gamma remap for depth maps.
    Assumes 1=near, 0=far.
    near_gamma -> [split, 1]
    far_gamma  -> [0, split]
    """
    depth = depth.clamp(0.0, 1.0)

    # Near branch
    near_val = (((depth - split).clamp(min=0) / (1 - split + 1e-6))
                .pow(near_gamma) * (1 - split)) + split

    # Far branch
    far_val = (((depth).clamp(max=split) / (split + 1e-6))
               .pow(far_gamma) * split)

    # Select branch without indexing
    out = torch.where(depth >= split, near_val, far_val)
    return out

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

    if 'depth-anything/Video-Depth-Anything' in model_id:
        checkpoint_name = f'video_depth_anything_{encoder}.pth'
    elif 'depth-anything/Metric-Video-Depth-Anything' in model_id:
        checkpoint_name = f'metric_video_depth_anything_{encoder}.pth'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    checkpoint_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name, cache_dir=CACHE_PATH)

    model = VideoDepthAnything(**model_configs[encoder])
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True), strict=True)
    return model.to(DEVICE).eval()

# Model Wrapper Class
class DepthModelWrapper:
    def __init__(self, model_path, device, device_info, dtype, size=None):
        """
        Wrapper class that handles both PyTorch and TensorRT backends.
        """
        self.device = device
        self.device_info = device_info
        self.dtype = dtype
        self.model_path = model_path
        self.size = size
        
        # Determine backend based on device
        self.is_cuda = "CUDA" in device_info
        
        # Use PyTorch backend for DirectML/MPS/CPU
        self.backend = "PyTorch"
        self.model = self._load_pytorch_model()
    
        if hasattr(self.model, 'to'):
            self.model.to(DTYPE)
        
        print(f"Using backend: {self.backend}")
    
    def _load_pytorch_model(self):
        """Load the original PyTorch model."""
        # Load model
        if 'Video-Depth-Anything' in MODEL_ID:
            model = get_video_depth_anything_model(MODEL_ID)

        else:
            # Load depth model
            model = AutoModelForDepthEstimation.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16 if FP16 else torch.float32,
                cache_dir=CACHE_PATH,
                weights_only=True
            ).to(DEVICE).eval()
        
        if FP16:
            model.half()
        return model
    
    def __call__(self, tensor):
        """Run inference using the active backend."""
        if "CUDA" in DEVICE_INFO:
            with torch.amp.autocast('cuda'):
                if self.backend == "PyTorch":
                    if "Video-Depth-Anything" in MODEL_ID:
                        return self.model(pixel_values=tensor)
                    return self.model(pixel_values=tensor).predicted_depth
                else:
                    return self.model(tensor)
        else:
            with torch.no_grad():
                if self.backend == "PyTorch":
                    if "Video-Depth-Anything" in MODEL_ID:
                        return self.model(pixel_values=tensor)
                    return self.model(pixel_values=tensor).predicted_depth
                else:
                    return self.model(tensor)

# Initialize model wrapper
model_wraper = DepthModelWrapper(
    model_path=MODEL_ID,
    device=DEVICE,
    device_info=DEVICE_INFO,
    dtype=DTYPE
)

MODEL_DTYPE = next(model_wraper.model.parameters()).dtype if hasattr(model_wraper.model, 'parameters') else DTYPE
MEAN = torch.tensor([0.485,0.456,0.406], device=DEVICE).view(1,3,1,1)
STD = torch.tensor([0.229,0.224,0.225], device=DEVICE).view(1,3,1,1)


# # Initialize with dummy input for warmup
# def warmup_model(model_wraper, steps: int = 3):
#     if "CUDA" in DEVICE_INFO:
#         with torch.amp.autocast('cuda'):
#             for i in range(steps):
#                 dummy = torch.randn(1, 3, DEPTH_RESOLUTION, DEPTH_RESOLUTION,
#                                     device=DEVICE, dtype=MODEL_DTYPE)
#                 model_wraper(dummy)
#     else:
#         with torch.no_grad():
#             for i in range(steps):
#                 dummy = torch.randn(1, 3, DEPTH_RESOLUTION, DEPTH_RESOLUTION,
#                                     device=DEVICE, dtype=MODEL_DTYPE)
#                 model_wraper(dummy)
#         # print(f"Warmup complete with {steps} iterations.")

# warmup_model(model_wraper, steps=3)

lock = Lock()

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

def get_target_size(h, w, depth_resolution):
    scale = depth_resolution / min(h, w)
    target_h, target_w = int(h * scale), int(w * scale)
    target_h = (target_h // 14) * 14
    target_w = (target_w // 14) * 14
    return target_w, target_h

# Modified predict_depth function with improved TRT integration
def predict_depth(image_rgb: np.ndarray, return_tuple=False, use_temporal_smooth: bool = True):
    """
    Returns depth in [0,1], 1=near, 0=far. Optionally returns (depth, rgb_c).
    """
    # Get input dimensions
    h, w = image_rgb.shape[:2]
    
    # Compute target size: shortest edge to DEPTH_RESOLUTION, preserve aspect ratio
    scale = DEPTH_RESOLUTION / min(h, w)
    target_h, target_w = int(h * scale), int(w * scale)
    # Ensure dimensions are divisible by 14 (ViT patch size)
    if "anything" in MODEL_ID:
        target_h = (target_h // 14) * 14
        target_w = (target_w // 14) * 14
    # fix for Video-Depth-Anything
    if 'Video-Depth-Anything' in MODEL_ID:
        calc_w = max(target_w,target_h)
        target_size = (calc_w, calc_w)
    else:
        target_size = (target_w, target_h)
    
    if return_tuple:
        # Convert to tensor and prepare rgb_c
        tensor = torch.from_numpy(image_rgb).to(device=DEVICE, dtype=MODEL_DTYPE, non_blocking=True)
        rgb_c = tensor.permute(2, 0, 1).contiguous()  # [C,H,W]
        tensor = rgb_c.unsqueeze(0) / 255.0
        # Resize using bilinear interpolation
        tensor = F.interpolate(tensor, size=target_size, mode='bilinear', align_corners=False)
    else:
        # Resize on CPU with bilinear interpolation
        if (h, w) != target_size:
            input_rgb = cv2.resize(image_rgb, target_size, interpolation=cv2.INTER_CUBIC)
        else:
            input_rgb = image_rgb
        # Convert to tensor
        tensor = torch.from_numpy(input_rgb).permute(2, 0, 1).contiguous().unsqueeze(0).to(DEVICE, dtype=DTYPE) / 255.0
    
    tensor = ((tensor - MEAN) / STD).contiguous()
    tensor = tensor.to(dtype=MODEL_DTYPE)
        
    # Use model wrapper instead of direct model call
    if 'Video-Depth-Anything' in MODEL_ID:
        depth = model_wraper(tensor)
    else:
        if "CUDA" in DEVICE_INFO:
            with torch.amp.autocast('cuda'):
                depth = model_wraper(tensor)
        else:
            with torch.no_grad():
                depth = model_wraper(tensor)
    
    # Robust normalize and Post depth processing
    depth = apply_stretch(depth, 5, 95)
    
    # invert for metric models
    if 'Metric' in MODEL_ID:
        depth = 1.0 - depth
        
    # post processing
    # depth = apply_sigmoid(depth, k=1, midpoint=0.618)
    depth = apply_piecewise(depth, split=0.5, near_gamma=1.2, far_gamma=0.6)
    depth = apply_foreground_scale(depth, scale=FOREGROUND_SCALE)
    depth = normalize_tensor(depth)
    # Mild AA to reduce jaggies
    depth = anti_alias(depth, strength=AA_STRENTH)

    # Optional temporal stabilization (EMA)
    if use_temporal_smooth:
        depth = depth_stabilizer(depth)
        
     # Interpolate output to original size
    depth = F.interpolate(depth.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=True)
    depth = depth.squeeze(0)
    if return_tuple:
        return depth, rgb_c
    else:
        return depth  
 
if __name__ == "__main__":
    depth = predict_depth(image_rgb).squeeze(0)

    import matplotlib.pyplot as plt
    plt.imshow(depth.cpu().detach().numpy(), cmap='inferno')
    plt.colorbar()
    plt.show()
    plt.close()
    plt.imshow(depth.cpu().detach().numpy(), cmap='inferno')
    plt.colorbar()
    plt.savefig("test.png", dpi=300)