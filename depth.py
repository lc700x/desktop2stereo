# depth.py
import platform
# get system type
OS_NAME = platform.system()
if  OS_NAME == "Darwin":
    import os, warnings
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    warnings.filterwarnings(
        "ignore",
        message=".*aten::upsample_bicubic2d.out.*MPS backend.*",
        category=UserWarning
)
import torch
import torch.nn.functional as F
from transformers import AutoModelForDepthEstimation
import numpy as np
from threading import Lock
import cv2

# load customized settings
import yaml
with open("settings.yaml") as settings_yaml:
    try:
        settings = yaml.safe_load(settings_yaml)
    except yaml.YAMLError as exc:
        print(exc)

# Model configuration
MODEL_ID = settings["depth_model"]
CACHE_PATH = settings["download_path"]
DTYPE = torch.float16 if settings["fp_16"] else torch.float32 # Use float32 for DirectML compatibility

# Initialize DirectML Device
def get_device():
    """
    Returns a torch.device and a human‐readable device info string.
    """
    try:
        import torch_directml
        if torch_directml.is_available():
            dev = torch_directml.device()
            info = f"Using DirectML device: {torch_directml.device_name(0)}"
            return dev, info
    except ImportError:
        pass

    if torch.cuda.is_available():
        return torch.device("cuda"), f"Using CUDA device: {torch.cuda.get_device_name(0)}"
    if torch.backends.mps.is_available():
        return torch.device("mps"), "Using Apple Silicon (MPS) device"
    return torch.device("cpu"), "Using CPU device"


# Get the device and print information
DEVICE, DEVICE_INFO = get_device()

# Load model with same configuration as example
model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID, torch_dtype=DTYPE, cache_dir=CACHE_PATH, weights_only=True).half().to(DEVICE).eval()
INPUT_W= settings["depth_resolution"]   # model's native resolution

# Normalization parameters (same as example)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

# Warm-up with dummy input
with torch.no_grad():
    dummy = torch.zeros(1, 3, INPUT_W, INPUT_W, device=DEVICE, dtype=DTYPE)
    model(pixel_values=dummy)    

lock = Lock()

@torch.no_grad()
def predict_depth(image_rgb: np.ndarray) -> np.ndarray:
    """
    Predict depth map from RGB image (similar to pipeline example but optimized for DirectML)
    Args:
        image_rgb: Input RGB image as numpy array (H, W, 3) in uint8 format
    Returns:
        Depth map as numpy array (H, W) normalized to [0, 1]
    """
    # Convert to tensor and normalize (similar to pipeline's preprocessing)
    tensor = torch.from_numpy(image_rgb)              # CPU → CPU tensor (uint8)
    tensor = tensor.permute(2, 0, 1).float() / 255.  # HWC → CHW, 0-1 range
    tensor = tensor.unsqueeze(0).to(DEVICE, dtype=DTYPE, non_blocking=True)

    # Resize and normalize (same as pipeline)
    tensor = F.interpolate(tensor, (INPUT_W, INPUT_W), mode='bilinear', align_corners=False)
    tensor = (tensor - MEAN.to(DTYPE)) / STD.to(DTYPE)

    # Inference with thread safety
    with lock:
        depth = model(pixel_values=tensor).predicted_depth  # (1, H, W)

    # Post-processing (same as pipeline)
    h, w = image_rgb.shape[:2]
    depth = F.interpolate(depth.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False)[0, 0]

    # Normalize to [0, 1] (same as pipeline output)
    depth = depth / depth.max().clamp(min=1e-6)
    return depth.cpu().numpy().astype('float32')

def process(img_rgb: np.ndarray, size,  downscale: float = 0.5) -> np.ndarray:
        """
        Process raw BGR image: convert to RGB and apply downscale if set.
        This can be called in a separate thread.
        """
        img_rgb = img_rgb.reshape((size[0], size[1], 3))
        scaled_width, scaled_height = int(size[0] * downscale), int(size[1] * downscale)
        # Downscale if requested
        if downscale < 1.0:
            img_rgb = cv2.resize(img_rgb, (scaled_height, scaled_width),
                                 interpolation=cv2.INTER_AREA)
        return img_rgb

def process_tensor(img: np.ndarray, downscale: float = 0.5) -> torch.Tensor:
        img_bgr = torch.from_numpy(img).to(DEVICE, dtype=torch.uint8, non_blocking=True)  # H,W,C
        img_rgb = img_bgr[..., [2,1,0]]  # BGR to RGB
        chw = img_rgb.permute(2, 0, 1).float()  # (3,H,W)
        if downscale < 1.0:
            H, W, _ = img_rgb.shape
            new_h, new_w = int(H * downscale), int(W * downscale)
            chw = F.interpolate(chw.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False)
        # Add batch dim
        chw = chw.squeeze(0)  # (3,H,W)
        return chw

def predict_depth_tensor(image_rgb):
    """
    Predict depth map from RGB image (similar to pipeline example but optimized for DirectML)
    Args:
        image_rgb: Input RGB image as tensor (3, H, W) in uint8 format
    Returns:
        Depth map as tensor (384, 384) normalized to [0, 1]
    """
    tensor = image_rgb.float() / 255.0  # Convert to float32 in range [0, 1]
    # Resize and normalize (same as pipeline)
    tensor = F.interpolate(tensor.unsqueeze(0), (INPUT_W, INPUT_W), mode='bilinear', align_corners=False)
    tensor = tensor.squeeze(0)  # Remove batch dimension
    tensor = (tensor - MEAN) / STD

    # Inference with thread safety
    with lock:
        depth = model(pixel_values=tensor).predicted_depth  # (1, H, W)
    # Post-processing (same as pipeline)
    h, w = image_rgb.shape[1:3]
    depth = F.interpolate(depth.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False)[0, 0]

    # Normalize to [0, 1] (same as pipeline output)
    depth = depth / depth.max().clamp(min=1e-6)
    return depth.float()

def make_sbs(rgb: torch.Tensor, depth: torch.Tensor, ipd_uv: float = 0.064, depth_strength: float = 0.1, half: bool = True) -> np.ndarray:
        """
        Build a side-by-side stereo frame.

        Parameters
        ----------
        rgb : H×W×3 float32 array in range [0..1]
        depth : H×W float32 array in range [0..1]
        ipd_uv : float
            interpupillary distance in UV space (0–1, relative to image width)
        depth_strength : float
            multiplier applied to the per-pixel horizontal parallax
        half : bool, optional
            If True, returns “half-SBS”: the output width equals W.
            Otherwise returns full-width SBS (2W).

        Returns
        -------
        np.ndarray
            uint8 image of shape (H, 2W, 3) if half == False,
            or (H, W, 3) when half == True.
        """
        H, W = depth.shape
        inv = 1.0 - depth
        max_px = int(ipd_uv * W)
        shifts = (inv * max_px * depth_strength).astype(np.int32)

        left  = np.zeros_like(rgb)
        right = np.zeros_like(rgb)
        xs    = np.arange(W)[None, :]

        for y in range(H):
            s = shifts[y]
            xx_left  = np.clip(xs + (s // 2),  0, W-1)
            xx_right = np.clip(xs - (s // 2),  0, W-1)
            left[y]  = rgb[y, xx_left]
            right[y] = rgb[y, xx_right]

        # Full-resolution SBS: concatenate horizontally
        sbs_full = np.concatenate((left, right), axis=1)  # H × (2W) × 3
        if not half:
            return sbs_full.astype(np.uint8)

        # Half-SBS: simple 2:1 sub-sampling in X direction
        # (i.e., take every second column)
        sbs_half = sbs_full[:, ::2, :]  # H × W × 3
        # import cv2  # For debugging purposes, can be removed later
        # cv2.imwrite("sbs_half.jpg", sbs_half)  # Debugging line, can be removed
        # exit()
        return sbs_half.astype(np.uint8)
    
def make_sbs_tensor(rgb: torch.Tensor, depth: torch.Tensor, ipd_uv: float = 0.064, depth_strength: float = 0.1, half: bool = True) -> np.array:
    """
    Build a side-by-side stereo frame using PyTorch tensors with DirectML support.

    Parameters
    ----------
    rgb : Tensor (3, H, W) float32 in range [0..1]
    depth : Tensor (H, W) float32 in range [0..1]
    ipd_uv : float
        interpupillary distance in UV space (0-1, relative to image width)
    depth_strength : float
        multiplier applied to the per-pixel horizontal parallax
    half : bool, optional
        If True, returns "half-SBS": the output width equals W.
        Otherwise returns full-width SBS (2W).

    Returns
    -------
    torch.Tensor
        uint8 image of shape (3, H, 2W) if half == False,
        or (3, H, W) when half == True.
    """
    # Ensure tensors are on the same device
    _, H, W = rgb.shape
    device = rgb.device
    # reshape depth[384, 384] to match rgb dimensions [H, W]
    inv = 1.0 - depth
    max_px = int(ipd_uv * W)
    shifts = (inv * max_px * depth_strength).round().long()
    
    # Create coordinate grid
    xs = torch.arange(W, device=device).expand(H, -1)  # (H, W)
    
    # Calculate left and right coordinates
    shifts_half = (shifts // 2).clamp(min=0, max=W//2)
    left_coords = torch.clamp(xs + shifts_half, 0, W-1)
    right_coords = torch.clamp(xs - shifts_half, 0, W-1)
    
    # Gather pixels using advanced indexing
    rgb = rgb.permute(1, 2, 0)  # (H, W, 3) for easier indexing
    # Vectorized implementation using gather
    y_indices = torch.arange(H, device=device)[:, None].expand(-1, W)
    left = rgb[y_indices, left_coords]
    right = rgb[y_indices, right_coords]
    
    # Concatenate for full SBS
    sbs_full = torch.cat([left, right], dim=1)  # (H, 2W, 3)
    
    if not half:
        return (sbs_full).clamp(0, 255).byte().cpu().numpy()
    
    # For half-SBS, use strided sampling
    sbs_half = sbs_full[:, ::2, :]  # (H, 2W, 3) to (H, W, 3)
    return (sbs_half).clamp(0, 255).byte().cpu().numpy()
