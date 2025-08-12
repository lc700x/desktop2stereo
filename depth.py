# depth.py
import yaml
import os
from gui import OS_NAME
# Ignore wanning for MPS
if  OS_NAME == "Darwin":
    import os, warnings
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    warnings.filterwarnings(
        "ignore",
        message=".*aten::upsample_bicubic2d.out.*MPS backend.*",
        category=UserWarning
)

# load customized settings
with open("settings.yaml") as settings_yaml:
    try:
        settings = yaml.safe_load(settings_yaml)
    except yaml.YAMLError as exc:
        print(exc)

# Set Hugging Face environment variable
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# if len(sys.argv) >= 2 and sys.argv[1] == '--hf-mirror':
os.environ['HF_ENDPOINT'] = settings["HF Endpoint"]

import torch
torch.set_num_threads(1) # reduce cpu usage
import torch.nn.functional as F
from transformers import AutoModelForDepthEstimation
import numpy as np
from threading import Lock
import cv2

# Model configuration
MODEL_ID = settings["Depth Model"]
CACHE_PATH = settings["Download Path"]
DTYPE = torch.float16 if settings["FP16"] else torch.float32 # Use float32 for DirectML compatibility

# Initialize DirectML Device
def get_device(index=0):
    """
    Returns a torch.device and a human‐readable device info string.
    """
    try:
        import torch_directml
        if torch_directml.is_available():
            dev = torch_directml.device(index)
            info = f"Using DirectML device: {torch_directml.device_name(index)}"
            return dev, info
        if torch.cuda.is_available():
            return torch.device("cuda"), f"Using CUDA device: {torch.cuda.get_device_name(index)}"
        if torch.backends.mps.is_available():
            return torch.device("mps"), "Using Apple Silicon (MPS) device"
    except:
        pass
    return torch.device("cpu"), "Using CPU device"


# Get the device and print information
DEVICE, DEVICE_INFO = get_device(settings["Device"])
# Load model with same configuration as example
model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID, torch_dtype=DTYPE, cache_dir=CACHE_PATH, weights_only=True).half().to(DEVICE).eval()
INPUT_W= settings["Depth Resolution"]  # model's native resolution

# Normalization parameters (same as example)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

# Warm-up with dummy input
with torch.no_grad():
    dummy = torch.zeros(1, 3, INPUT_W, INPUT_W, device=DEVICE, dtype=DTYPE)
    model(pixel_values=dummy)    

lock = Lock()


def process(img_rgb: np.ndarray, size) -> np.ndarray:
        """
        Process raw BGR image: convert to RGB and apply downscale if set.
        This can be called in a separate thread.
        """
        # Downscale the image if needed
        if size[0] < img_rgb.shape[0]:
            img_rgb = cv2.resize(img_rgb, (size[0], size[1]), interpolation=cv2.INTER_AREA)
        return img_rgb

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
    tensor = torch.from_numpy(image_rgb).to(DEVICE, dtype=DTYPE)  # cpu usage related step
    tensor = tensor.permute(2, 0, 1).float() / 255.  # HWC → CHW, 0-1 range
    tensor = tensor.unsqueeze(0)

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
    return depth.detach().cpu().numpy().astype('float32')
    # return depth

def predict_depth2 (image_rgb: np.ndarray, size) -> np.ndarray:
        """
        Process raw BGR image: convert to RGB and apply downscale if set.
        This can be called in a separate thread.
        """
        # Convert to tensor and normalize (similar to pipeline's preprocessing)
        tensor = torch.from_numpy(image_rgb.copy())              # CPU → CPU tensor (uint8)
        tensor = tensor.permute(2, 0, 1).float() / 255.  # HWC → CHW, 0-1 range
        tensor = tensor.unsqueeze(0).to(DEVICE, dtype=DTYPE)

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
        return depth.detach().cpu().numpy().astype('float32')

def process_tensor(img: np.ndarray, size: float = 0.5) -> torch.Tensor:
        img_rgb = torch.from_numpy(img).to(DEVICE, dtype=torch.uint8, non_blocking=True)  # H,W,C
        chw = img_rgb.permute(2, 0, 1).float()  # (3,H,W)
        if size[0] < img_rgb.shape[0]:
            chw = F.interpolate(chw.unsqueeze(0), size=size, mode='bilinear', align_corners=False)
        # Add batch dim
        chw = chw.squeeze(0)  # (3,H,W)
        return chw

def predict_depth_tensor(image_rgb):
    """
    Predict depth map from RGB image (similar to pipeline example but optimized for DirectML)
    Args:
        image_rgb: Input RGB image as numpy array (H, W, 3) in uint8 format
    Returns:
        Depth map as numpy array (H, W) normalized to [0, 1]
    """
    # Convert to tensor and normalize (similar to pipeline's preprocessing)
    img_tensor = torch.from_numpy(image_rgb.copy()).to(DEVICE, dtype=DTYPE)              # CPU → CPU tensor (uint8)
    tensor = img_tensor.permute(2, 0, 1).float() / 255.  # HWC → CHW, 0-1 range
    tensor = tensor.unsqueeze(0)

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
    return img_tensor, depth

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

def make_sbs_tensor(rgb, depth, ipd_uv=0.03, depth_strength=1.0, half=False):
    """
    Build a side-by-side stereo frame using PyTorch tensors with DirectML support.

    Parameters
    ----------
    rgb : Tensor (H, W, 3) float32 in range [0..1]
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
        uint8 image of shape (H, 2W, 3) if half == False,
        or (H, W, 3) when half == True.
    """
    assert rgb.ndim == 3 and rgb.shape[2] == 3, "rgb must be (H, W, 3)"
    assert depth.shape == rgb.shape[:2], "depth must be (H, W)"
    
    H, W, _ = rgb.shape
    device = rgb.device

    # inverse depth to get parallax (closer = bigger shift)
    inv = 1.0 - depth
    max_px = int(ipd_uv * W)
    shifts = (inv * max_px * depth_strength).round().long()  # (H, W)

    # coordinate grid
    xs = torch.arange(W, device=device).unsqueeze(0).expand(H, -1)  # (H, W)

    # half shifts
    shifts_half = (shifts // 2).clamp(min=0, max=W//2)  # (H, W)

    # left/right coordinates
    left_coords  = torch.clamp(xs + shifts_half,  0, W-1)
    right_coords = torch.clamp(xs - shifts_half,  0, W-1)

    # prepare y indices
    y_indices = torch.arange(H, device=device).unsqueeze(1).expand(-1, W)

    # gather pixels
    left  = rgb[y_indices, left_coords]   # (H, W, 3)
    right = rgb[y_indices, right_coords]  # (H, W, 3)

    # concatenate SBS
    sbs_full = torch.cat([left, right], dim=1)  # (H, 2W, 3)

    if half:
        sbs_half = sbs_full[:, ::2, :]
        return sbs_half.clamp(0, 255).byte().cpu().numpy()  # (H, W, 3)

    return sbs_full.clamp(0, 255).byte().cpu().numpy()  # (H, 2W, 3)



