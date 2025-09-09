# depth.py
import torch
torch.set_num_threads(1)  # Limit PyTorch to use only 1 thread for CPU operations
import torch.nn.functional as F
from transformers import AutoModelForDepthEstimation
import numpy as np
from threading import Lock
import cv2
from utils import DEVICE_ID, MODEL_ID, CACHE_PATH, FP16, DEPTH_RESOLUTION, DILATION_SIZE, AA_STRENTH

# Model configuration - use float16 if FP16 flag is True, otherwise float32
DTYPE = torch.float16 if FP16 else torch.float32

# Initialize DirectML Device
def get_device(index=0):
    """Get the best available device for computation in this order: DirectML -> CUDA -> MPS -> CPU"""
    try:
        try:
            import torch_directml
            if torch_directml.is_available():
                return torch_directml.device(index), f"Using DirectML device: {torch_directml.device_name(index)}"
        except ImportError:
            pass
        if torch.cuda.is_available():
            return torch.device("cuda"), f"Using CUDA device: {torch.cuda.get_device_name(index)}"
        if torch.backends.mps.is_available():
            return torch.device("mps"), "Using Apple Silicon (MPS) device"
    except:
        return torch.device("cpu"), "Using CPU device"

# Initialize device and print device information
DEVICE, DEVICE_INFO = get_device(DEVICE_ID)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # Enable cuDNN benchmark for faster CUDA operations

print(f"{DEVICE_INFO}")
print(f"Model: {MODEL_ID}")

# Load depth estimation model from HuggingFace
model = AutoModelForDepthEstimation.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if FP16 else torch.float32,
    cache_dir=CACHE_PATH,
    weights_only=True
).to(DEVICE).eval()  # Set model to evaluation mode

# Convert model to half precision if FP16 is enabled
if FP16:
    try:
        model.half()
    except Exception:
        pass

# Get model's data type and set normalization parameters
MODEL_DTYPE = next(model.parameters()).dtype
MEAN = torch.tensor([0.485,0.456,0.406], device=DEVICE, dtype=MODEL_DTYPE).view(1,3,1,1)
STD = torch.tensor([0.229,0.224,0.225], device=DEVICE, dtype=MODEL_DTYPE).view(1,3,1,1)

# Test model with dummy input to check compatibility
with torch.no_grad():
    dummy = torch.zeros(1,3,DEPTH_RESOLUTION,DEPTH_RESOLUTION, device=DEVICE, dtype=MODEL_DTYPE)
    try:
        model(pixel_values=dummy)
    except Exception:
        if MODEL_DTYPE == torch.float16:
            # Fall back to float32 if float16 fails
            MODEL_DTYPE = torch.float32
            MEAN = MEAN.to(dtype=MODEL_DTYPE)
            STD = STD.to(dtype=MODEL_DTYPE)
            dummy = torch.zeros(1,3,DEPTH_RESOLUTION,DEPTH_RESOLUTION, device=DEVICE, dtype=MODEL_DTYPE)
            model.to(dtype=MODEL_DTYPE)
            model(pixel_values=dummy)

# Thread lock for thread-safe operations
lock = Lock()

def to_model_dtype(t: torch.Tensor) -> torch.Tensor:
    """Convert tensor to the model's preferred data type"""
    if not torch.is_floating_point(t):
        return t.to(dtype=MODEL_DTYPE)
    if t.dtype != MODEL_DTYPE:
        return t.to(dtype=MODEL_DTYPE)
    return t

def apply_foreground_scale_torch(depth: torch.Tensor, scale: float) -> torch.Tensor:
    """Apply foreground scaling to depth map using power function"""
    depth = to_model_dtype(depth)
    if scale > 0:
        return torch.pow(depth, 1.0 + scale)
    elif scale < 0:
        return torch.pow(depth, 1.0 / (1.0 - scale))
    else:
        return depth

def gaussian_blur_tensor(depth: torch.Tensor, k: int = 5, strength: float = 1.0) -> torch.Tensor:
    """Apply Gaussian blur to depth tensor"""
    depth = to_model_dtype(depth).unsqueeze(0).unsqueeze(0)
    if k <= 1 or strength <= 0:
        return depth.squeeze()
    sigma = max(0.001, 0.5 * strength)
    coords = torch.arange(k, device=depth.device, dtype=depth.dtype) - k // 2
    gauss = torch.exp(-(coords**2) / (2 * sigma**2))
    gauss = gauss / gauss.sum()
    # Apply separable Gaussian blur (horizontal then vertical)
    depth = F.conv2d(depth, gauss.view(1, 1, 1, -1), padding=(0, k // 2), groups=1)
    depth = F.conv2d(depth, gauss.view(1, 1, -1, 1), padding=(k // 2, 0), groups=1)
    return depth.squeeze()

def dilate_depth(depth: torch.Tensor, dilation_size: int = DILATION_SIZE) -> torch.Tensor:
    """Dilate depth map using max pooling"""
    if dilation_size <= 0:
        return depth
    if depth.dim() == 2:
        depth = depth.unsqueeze(0).unsqueeze(0)
    elif depth.dim() == 3:
        depth = depth.unsqueeze(1)
    kernel_size = dilation_size if dilation_size % 2 == 1 else dilation_size + 1
    pad = kernel_size // 2
    depth = to_model_dtype(depth)
    dilated = F.max_pool2d(depth, kernel_size, stride=1, padding=pad)
    return dilated.squeeze()

def process_tensor(img_rgb: np.ndarray, height) -> torch.Tensor:
    """Resize RGB image and convert to tensor"""
    if height < img_rgb.shape[0]:
        width = int(img_rgb.shape[1]/img_rgb.shape[0]*height)
        img_rgb = cv2.resize(img_rgb,(width,height), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(img_rgb).to(DEVICE, dtype=DTYPE)

def process(img_rgb: np.ndarray, height) -> np.ndarray:
    """Resize RGB image (numpy array version)"""
    if height < img_rgb.shape[0]:
        width = int(img_rgb.shape[1]/img_rgb.shape[0]*height)
        img_rgb = cv2.resize(img_rgb,(width,height), interpolation=cv2.INTER_AREA)
    return img_rgb

def predict_depth_tensor(image_rgb: np.ndarray):
    """Predict depth map from RGB image and return as tensor"""
    h, w = image_rgb.shape[:2]
    tensor = torch.from_numpy(image_rgb).to(DEVICE, dtype=DTYPE)
    rgb_c = tensor.permute(2,0,1).contiguous()
    tensor = rgb_c.unsqueeze(0).to(dtype=MODEL_DTYPE)/255
    tensor = F.interpolate(tensor, (DEPTH_RESOLUTION, DEPTH_RESOLUTION), mode="bilinear", align_corners=False)
    tensor = (tensor - MEAN) / STD  # Normalize
    
    with torch.no_grad():
        try:
            out = model(pixel_values=tensor)
        except Exception:
            if MODEL_DTYPE == torch.float16:
                out = model(pixel_values=tensor.to(dtype=torch.float32))
            else:
                raise
    
    # Handle different model output formats
    if hasattr(out, 'predicted_depth'):
        depth = out.predicted_depth
    elif isinstance(out, dict) and 'predicted_depth' in out:
        depth = out['predicted_depth']
    elif isinstance(out, dict) and 'depth' in out:
        depth = out['depth']
    else:
        depth = getattr(out, 'depth', None)
    if depth is None:
        raise RuntimeError("Unable to find depth output from model")
    
    depth = to_model_dtype(depth.squeeze())
    # Resize back to input resolution
    depth = F.interpolate(depth.unsqueeze(0).unsqueeze(0), size=(h,w), mode='bilinear', align_corners=False)[0,0]
    # Normalize depth adaptively
    depth_range = depth.max() - depth.min()
    depth = (depth - depth.min()) / (depth_range + 1e-6)
    depth = depth.clamp(0.2, 0.9)  # Clip extreme values
    depth = depth / depth.max()  # Normalize to [0,1]
    
    # Post-process
    depth = dilate_depth(depth, dilation_size=DILATION_SIZE)
    depth = gaussian_blur_tensor(depth, k=5, strength=AA_STRENTH)
    return depth, rgb_c

def predict_depth(image_rgb: np.ndarray):
    """Predict depth map from RGB image and return as numpy array"""
    h, w = image_rgb.shape[:2]
    tensor = torch.from_numpy(image_rgb).to(DEVICE, dtype=DTYPE)
    tensor = tensor.permute(2,0,1).unsqueeze(0).to(dtype=MODEL_DTYPE) / 255.0
    tensor = F.interpolate(tensor, (DEPTH_RESOLUTION, DEPTH_RESOLUTION), mode="bilinear", align_corners=False)
    tensor = (tensor - MEAN) / STD
    
    with torch.no_grad():
        try:
            out = model(pixel_values=tensor)
        except Exception:
            if MODEL_DTYPE == torch.float16:
                out = model(pixel_values=tensor.to(dtype=torch.float32))
            else:
                raise
    
    # Handle different model output formats
    if hasattr(out, 'predicted_depth'):
        depth = out.predicted_depth
    elif isinstance(out, dict) and 'predicted_depth' in out:
        depth = out['predicted_depth']
    elif isinstance(out, dict) and 'depth' in out:
        depth = out['depth']
    else:
        depth = getattr(out, 'depth', None)
    if depth is None:
        raise RuntimeError("Unable to find depth output from model")
    
    depth = to_model_dtype(depth.squeeze())
    # Resize back to input resolution
    depth = F.interpolate(depth.unsqueeze(0).unsqueeze(0), size=(h,w), mode='bilinear', align_corners=False)[0,0]
    # Normalize depth adaptively
    depth_range = depth.max() - depth.min()
    depth = (depth - depth.min()) / (depth_range + 1e-6)
    depth = depth.clamp(0.2, 0.9)
    depth = depth / depth.max()
    
    # Post-process
    depth = dilate_depth(depth, dilation_size=DILATION_SIZE)
    depth = gaussian_blur_tensor(depth, k=5, strength=AA_STRENTH)
    return depth.cpu().numpy().astype('float32')

def apply_antialias(depth: torch.Tensor, strength: float = AA_STRENTH) -> torch.Tensor:
    """Apply anti-aliasing to depth map"""
    depth = to_model_dtype(depth)
    if strength <= 0:
        return depth
    return gaussian_blur_tensor(depth, k=5, strength=strength)

def edge_detect(depth: torch.Tensor):
    """Apply Sobel edge detection to depth map"""
    depth = to_model_dtype(depth)
    # Sobel kernels for horizontal and vertical edges
    gx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], device=depth.device, dtype=depth.dtype).unsqueeze(0).unsqueeze(0)
    gy = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], device=depth.device, dtype=depth.dtype).unsqueeze(0).unsqueeze(0)
    depth4 = depth.unsqueeze(0).unsqueeze(0) if depth.dim() == 2 else depth.unsqueeze(1)
    grad_x = F.conv2d(depth4, gx, padding=1)
    grad_y = F.conv2d(depth4, gy, padding=1)
    grad = torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-8)  # Gradient magnitude
    return grad.squeeze()

def inpaint_depth(depth: torch.Tensor):
    """Inpaint missing depth values using OpenCV's NS inpainting"""
    d = depth.detach().to(device='cpu', dtype=torch.float32).numpy()
    mask = (d == 0).astype(np.uint8) * 255
    inpaint = cv2.inpaint((d*255).astype(np.uint8), mask, 3, cv2.INPAINT_NS)
    return torch.from_numpy(inpaint.astype(np.float32)/255.0).to(device=depth.device, dtype=depth.dtype)

def normalize_to_01(depth: torch.Tensor):
    """Normalize depth map to [0,1] range"""
    depth = to_model_dtype(depth)
    mn = depth.min()
    mx = depth.max()
    if mx <= mn:
        return torch.zeros_like(depth)
    return (depth - mn) / (mx - mn)

def make_sbs(rgb_c, depth, ipd_uv=0.064, depth_ratio=1.0, display_mode="Half-SBS"):
    """
    Create side-by-side stereo image from RGB and depth map.
    Args:
        rgb_c: RGB image tensor (C,H,W)
        depth: Depth map tensor (H,W)
        ipd_uv: Inter-pupillary distance in UV coordinates (0-1)
        depth_ratio: Depth effect strength
        display_mode: Output mode ("Half-SBS", "Full-SBS", "TAB")
    Returns:
        Side-by-side stereo image as numpy array
    """
    C, H, W = rgb_c.shape
    device = rgb_c.device
    depth_strength = 0.05  # Scaling factor for depth effect

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
        """Bilinear sampling of image at shifted coordinates"""
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
        """Pad image to match target aspect ratio"""
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
        out = torch.cat([left, right], dim=1)  # Top-and-bottom
    else:  # SBS
        out = torch.cat([left, right], dim=2)  # Side-by-side
    
    if display_mode != "Full-SBS":
        out = F.interpolate(out.unsqueeze(0), size=left.shape[1:], mode="area")[0]
    
    sbs = out.clamp(0, 255).to(torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy()
    return sbs