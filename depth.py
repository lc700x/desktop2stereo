# depth.py
import torch
torch.set_num_threads(1)
import torch.nn.functional as F
from transformers import AutoModelForDepthEstimation
import numpy as np
from threading import Lock
import cv2
from utils import DEVICE_ID, MODEL_ID, CACHE_PATH, FP16, DEPTH_RESOLUTION, AA_STRENTH, FOREGROUND_SCALE

# Model configuration
DTYPE = torch.float16 if FP16 else torch.float32

# Initialize DirectML Device
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
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

print(f"{DEVICE_INFO}")
print(f"Model: {MODEL_ID}")

# Load depth model
model = AutoModelForDepthEstimation.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if FP16 else torch.float32,
    cache_dir=CACHE_PATH,
    weights_only=True
).to(DEVICE).eval()

if 'CUDA' in DEVICE_INFO:
    model = torch.compile(model)  # Torch 2.0+ compile for speed
if FP16:
    model.half()

MODEL_DTYPE = next(model.parameters()).dtype
MEAN = torch.tensor([0.485,0.456,0.406], device=DEVICE).view(1,3,1,1)
STD = torch.tensor([0.229,0.224,0.225], device=DEVICE).view(1,3,1,1)
with torch.no_grad():
    dummy = torch.zeros(1,3,DEPTH_RESOLUTION,DEPTH_RESOLUTION, device=DEVICE, dtype=MODEL_DTYPE)
    model(pixel_values=dummy)

lock = Lock()

def apply_stretch(x: torch.Tensor, low: float = 2.0, high: float = 98.0) -> torch.Tensor:
    """
    Percentile-based clipping + normalization.
    Fully DirectML compatible (no torch.clamp).
    """
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
    if not torch.is_floating_point(depth):
        depth = depth.float()

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
    if height < img_rgb.shape[0]:
        width = int(img_rgb.shape[1]/img_rgb.shape[0]*height)
        img_rgb = cv2.resize(img_rgb,(width,height), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(img_rgb).to(DEVICE, dtype=DTYPE)

def process(img_rgb: np.ndarray, height) -> np.ndarray:
    if height < img_rgb.shape[0]:
        width = int(img_rgb.shape[1]/img_rgb.shape[0]*height)
        img_rgb = cv2.resize(img_rgb,(width,height), interpolation=cv2.INTER_AREA)
    return img_rgb

def normalize_tensor(tensor):
    return (tensor - tensor.min())/(tensor.max() - tensor.min()+1e-6)

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
                self.prev = depth.detach()
                return depth
            out = self.alpha * self.prev + (1.0 - self.alpha) * depth
            self.prev = out.detach()
            return out

depth_stabilizer = DepthStabilizer(alpha=0.9)  # increase alpha for more stability

# Piecewise gamma remap
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

def predict_depth(image_rgb: np.ndarray, return_tuple=False, use_temporal_smooth: bool = True):
    """
    Returns depth in [0,1], 1=near, 0=far. Optionally returns (depth, rgb_c).
    """
    h, w = image_rgb.shape[:2]
    if return_tuple:
        tensor = torch.from_numpy(image_rgb).to(DEVICE, dtype=DTYPE)
        rgb_c = tensor.permute(2,0,1).contiguous()  # [C,H,W]
        tensor = rgb_c.unsqueeze(0) / 255.0
        tensor = F.interpolate(tensor, (DEPTH_RESOLUTION, DEPTH_RESOLUTION), mode='bilinear', align_corners=False)
    else:
        # Resize input on CPU to model resolution for efficiency (avoids large GPU transfers and interpolate)
        target_size = (DEPTH_RESOLUTION, DEPTH_RESOLUTION)
        if (h, w) != target_size:
            interpolation = cv2.INTER_AREA if max(h, w) > DEPTH_RESOLUTION else cv2.INTER_LINEAR
            input_rgb = cv2.resize(image_rgb, target_size, interpolation=interpolation)
        
        tensor = torch.from_numpy(input_rgb).permute(2,0,1).contiguous().unsqueeze(0).to(DEVICE, dtype=DTYPE) / 255.0
    tensor = ((tensor - MEAN) / STD).contiguous()
    
    with torch.no_grad():  # Slightly faster than no_grad
        tensor = tensor.to(dtype=MODEL_DTYPE)
        depth = model(pixel_values=tensor).predicted_depth

    # Interpolate output to original size
    depth = F.interpolate(depth.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False)[0,0]
    
    # Robust normalize and Post depth processing
    depth = apply_stretch(depth, 5, 95)
    depth = apply_sigmoid(depth, k=4, midpoint=0.618)
    depth = apply_piecewise(depth, split=0.618, near_gamma=1.2, far_gamma=0.6)
    depth = apply_foreground_scale(depth, scale=FOREGROUND_SCALE)
    depth = normalize_tensor(depth)
    # Mild AA to reduce jaggies
    depth = anti_alias(depth, strength=AA_STRENTH)

    # Optional temporal stabilization (EMA)
    if use_temporal_smooth:
        depth = depth_stabilizer(depth)
    if return_tuple:
        return depth, rgb_c
    else:
        return depth
    
# generate left and right eye view    
def make_sbs(rgb_c, depth, ipd_uv=0.064, depth_ratio=1.0, display_mode="Half-SBS"):
    C, H, W = rgb_c.shape
    device = depth.device
    rgb_c = rgb_c.to(device, dtype=DTYPE)
    depth_strength = 0.05

    # Precompute pixel coordinates (cache this tensor outside if called repeatedly!)
    xs = torch.arange(W, dtype=torch.float32, device=device).view(1, -1)  # shape [1,W]

    # Depth inversion & shifts (keep as float for sub-pixel accuracy)
    inv = 1.0 - depth * depth_ratio
    max_px = ipd_uv * W
    shifts_half = inv * max_px * depth_strength  # [H,W], float

    # Build shifted indices with broadcasting
    idx_left = xs + shifts_half
    idx_right = xs - shifts_half

    def sample_bilinear(img, idx):
        # Clamp indices to image bounds for replicate padding
        idx_clamped = idx.clamp(0, W - 1)
        floor_idx = torch.floor(idx_clamped).long()
        ceil_idx = (floor_idx + 1).clamp(0, W - 1)
        frac = idx_clamped - floor_idx.float()

        # Gather values (expand indices to [C, H, W])
        floor_val = torch.gather(img, dim=2, index=floor_idx.unsqueeze(0).expand(C, H, W))
        ceil_val = torch.gather(img, dim=2, index=ceil_idx.unsqueeze(0).expand(C, H, W))

        # Bilinear interpolation
        return floor_val * (1 - frac).unsqueeze(0).expand(C, -1, -1) + ceil_val * frac.unsqueeze(0).expand(C, -1, -1)

    # Aspect ratio padding (do once, avoid recomputation)
    def pad_to_aspect(img, target_ratio=(16, 9)):
        _, h, w = img.shape
        t_w, t_h = target_ratio
        r_img, r_t = w / h, t_w / t_h
        if abs(r_img - r_t) < 1e-3:
            return img
        if r_img > r_t:  # too wide
            new_h = int(round(w / r_t))
            pad_top = (new_h - h) // 2
            return F.pad(img, (0, 0, pad_top, new_h - h - pad_top))
        else:  # too tall
            new_w = int(round(h * r_t))
            pad_left = (new_w - w) // 2
            return F.pad(img, (pad_left, new_w - w - pad_left, 0, 0))
    # Generate views with bilinear resampling
    gen_left = sample_bilinear(rgb_c, idx_left)
    gen_right = sample_bilinear(rgb_c, idx_right)
    left, right = pad_to_aspect(gen_left), pad_to_aspect(gen_right)

    if display_mode == "TAB":
        out = torch.cat([left, right], dim=1)
    else:  # SBS
        out = torch.cat([left, right], dim=2)
    if display_mode != "Full-SBS":
        out = F.interpolate(out.unsqueeze(0), size=left.shape[1:], mode="area")[0]
    sbs = out.clamp(0, 255).to(torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy()
    return sbs