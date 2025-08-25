# depth.py
import torch
torch.set_num_threads(1) # Set to avoid high CPU usage caused by default full threads
import torch.nn.functional as F
from transformers import AutoModelForDepthEstimation
import numpy as np
from threading import Lock
import cv2
from utils import DEVICE_ID, MODEL_ID, CACHE_PATH, FP16, DEPTH_RESOLUTION

# Model configuration
DTYPE = torch.float16 if FP16 else torch.float32 # Use float32 for DirectML compatibility

# Initialize DirectML Device
def get_device(index=0):
    """
    Returns a torch.device and a human‐readable device info string.
    """
    try:
        import torch_directml
        if torch_directml.is_available():
            return torch_directml.device(index), f"Using DirectML device: {torch_directml.device_name(index)}"
    except:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda"), f"Using CUDA device: {torch.cuda.get_device_name(index)}"
    if torch.backends.mps.is_available():
        return torch.device("mps"), "Using Apple Silicon (MPS) device"
    return torch.device("cpu"), "Using CPU device"

# Get the device and print information
DEVICE, DEVICE_INFO = get_device(DEVICE_ID)

# Enalbe cudnn  benchmark
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# Output Info
print(f"{DEVICE_INFO}")
print(f"Model: {MODEL_ID}")

# Load model with same configuration as example
model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID, torch_dtype=DTYPE, cache_dir=CACHE_PATH, weights_only=True).half().to(DEVICE).eval()

MODEL_DTYPE = next(model.parameters()).dtype
# Normalization parameters (same as example)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

# Warm-up with dummy input
with torch.no_grad():
    dummy = torch.zeros(1, 3, DEPTH_RESOLUTION, DEPTH_RESOLUTION, device=DEVICE, dtype=MODEL_DTYPE)
    model(pixel_values=dummy)    

lock = Lock()

def process_tensor(img_rgb: np.ndarray, size) -> np.ndarray:
    """
    Process raw BGR image: convert to RGB and apply downscale if set.
    This can be called in a separate thread.
    """
    # Downscale the image if needed
    if size[0] < img_rgb.shape[0]:
        img_rgb = cv2.resize(img_rgb, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    rgb_tensor = torch.from_numpy(img_rgb).to(DEVICE, dtype=DTYPE) # CPU → CPU tensor (uint8)
    return rgb_tensor

def process(img_rgb: np.ndarray, size) -> np.ndarray:
    """
    Process raw BGR image: convert to RGB and apply downscale if set.
    This can be called in a separate thread.
    """
    # Downscale the image if needed
    if size[0] < img_rgb.shape[0]:
        img_rgb = cv2.resize(img_rgb, (size[1], size[0]), interpolation=cv2.INTER_AREA)
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
    tensor = torch.from_numpy(image_rgb).to(DEVICE, dtype=DTYPE, non_blocking=True) # CPU → CPU tensor (uint8)
    tensor = tensor.permute(2, 0, 1).float() / 255  # HWC → CHW, 0-1 range
    tensor = tensor.unsqueeze(0) # set to improve performance

    # Resize and normalize (same as pipeline)
    tensor = F.interpolate(tensor, (DEPTH_RESOLUTION, DEPTH_RESOLUTION), mode='bilinear', align_corners=False)
    tensor = (tensor - MEAN) / STD

    # Inference with thread safety
    with lock:
        tensor = tensor.to(dtype=MODEL_DTYPE)
        depth = model(pixel_values=tensor).predicted_depth  # (1, H, W)

    # Post-processing (same as pipeline)
    h, w = image_rgb.shape[:2]
    depth = F.interpolate(depth.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False)[0, 0]

    # Normalize to [0, 1] (same as pipeline output)
    depth = depth / depth.max().clamp(min=1e-6)
    return depth
    # return depth.detach().cpu().numpy().astype('float32')
    
def predict_depth_tensor(img_rgb):
    """
    Predict depth map from RGB image (similar to pipeline example but optimized for DirectML)
    Args:
        image_rgb: Input RGB image as numpy array (H, W, 3) in uint8 format
    Returns:
        Depth map as numpy array (H, W) normalized to [0, 1]
    """
    # Ensure input is numpy (H,W,3)
    if isinstance(img_rgb, torch.Tensor):
        img_rgb = img_rgb.cpu().numpy()

    assert img_rgb.ndim == 3 and img_rgb.shape[2] == 3, \
        f"Expected HWC numpy image, got {img_rgb.shape}"

    rgb_tensor = torch.from_numpy(img_rgb.copy()).to(DEVICE, dtype=DTYPE) # CPU → CPU tensor (uint8)
    tensor = rgb_tensor.permute(2, 0, 1).float() / 255  # HWC → CHW, 0-1 range
    tensor = tensor.unsqueeze(0) # set to improve performance

    # Resize and normalize (same as pipeline)
    tensor = F.interpolate(tensor, (DEPTH_RESOLUTION, DEPTH_RESOLUTION), mode='bilinear', align_corners=False)
    tensor = (tensor - MEAN) / STD

    # Inference with thread safety
    with lock:
        tensor = tensor.to(dtype=MODEL_DTYPE)
        depth = model(pixel_values=tensor).predicted_depth  # (1, H, W)

    # Post-processing (same as pipeline)
    h, w = img_rgb.shape[:2]
    depth = F.interpolate(depth.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False)[0, 0]

    # Normalize to [0, 1] (same as pipeline output)
    depth = depth / depth.max().clamp(min=1e-6)
    return depth, rgb_tensor
    # return depth.detach().cpu().numpy().astype('float32')
    
def make_sbs(rgb: np.ndarray, depth, ipd_uv: float = 0.064,
                depth_strength: float = 0.1, display_mode: str = "Half-SBS") -> np.ndarray:
    """
    Build a side-by-side stereo frame using NumPy arrays.

    Parameters
    ----------
    rgb : ndarray (H, W, 3), float32 or uint8, range [0..1] or [0..255]
    depth : ndarray (H, W), float32 normalized [0..1]
    ipd_uv : float
        interpupillary distance in UV space (0-1, relative to image width)
    depth_strength : float
        multiplier applied to the per-pixel horizontal parallax
    display_mode : str
        "Full-SBS", "Half-SBS", or "TAB"

    Returns
    -------
    ndarray (H, W, 3) uint8
        - Full-SBS: (H, 2W, 3)
        - Half-SBS: (H, W, 3)
        - TAB: (2H, W, 3)
    """
    # Ensure correct dtype/range
    depth = depth.detach().cpu().numpy().astype('float32')
    h, w, _ = rgb.shape

    # inverse depth & pixel shift
    depth_inv = 1.0 - depth
    max_px = int(ipd_uv * w)
    shifts = np.round(depth_inv * max_px * depth_strength).astype(np.int32)

    # coordinate grid
    xs = np.arange(w)[None, :]  # (1, W)
    shifts_half = np.clip(shifts // 2, 0, w // 2)

    left_coords = np.clip(xs + shifts_half, 0, w - 1)
    right_coords = np.clip(xs - shifts_half, 0, w - 1)

    rows = np.arange(h)[:, None]  # (H, 1)

    left = rgb[rows, left_coords]   # (H, W, 3)
    right = rgb[rows, right_coords] # (H, W, 3)

    if display_mode == "TAB":
        output = np.concatenate([left, right], axis=0)  # (2H, W, 3)
        output = cv2.resize(output, (w, h), interpolation=cv2.INTER_AREA)
    else:
        output = np.concatenate([left, right], axis=1)  # (H, 2W, 3)
        if display_mode == "Half-SBS":
            # resize back to (H, W, 3)
            output = cv2.resize(output, (w, h), interpolation=cv2.INTER_AREA)

    return output.astype(np.uint8)

def make_sbs_tensor(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    ipd_uv: float = 0.064,
    depth_strength: float = 0.1,
    display_mode: str = "Half-SBS",   # "Full-SBS" | "Half-SBS" | "TAB"
):
    """
    Inputs:
        rgb: Tensor, either (C, H, W) or (H, W, C). Float32 expected.
            If values are in [0..1], set assume_rgb_range_0_1=True.
        depth: Tensor, (H, W) float32 in [0..1]
        ipd_uv: interpupillary distance in normalized image width
        depth_strength: multiplier for parallax
        display_mode: "Full-SBS", "Half-SBS", or "TAB" (TAB stacks vertically)
    Returns: numpy uint8 image (H, W or H, 2W, 3) depending on mode
    """
    # quick checks and canonicalize shapes
    rgb_c = rgb.permute(2,0,1).contiguous()
    device = rgb_c.device
    C, H, W = rgb_c.shape

    # compute per-pixel integer shift (H, W)
    inv = 1.0 - depth       # nearer -> smaller inv -> smaller shift? keep your original logic
    max_px = float(ipd_uv) * float(W)
    shifts = (inv * max_px * float(depth_strength)).round().to(torch.long, non_blocking=True)  # (H, W)

    # half-shift each eye (you used shifts//2 previously)
    shifts_half = (shifts // 2).clamp(min=0, max=W // 2)   # (H, W) long

    # create base x coordinates (H, W) without creating extra big copies in python loop
    xs = torch.arange(W, device=device, dtype=torch.long).unsqueeze(0).expand(H, W)  # (H, W) long

    left_idx = (xs + shifts_half).clamp(0, W - 1)   # (H, W)
    right_idx = (xs - shifts_half).clamp(0, W - 1)  # (H, W)

    # gather expects index shape to match src for the gathered dim, so expand across channels
    # shapes for gather: src = (C, H, W), index = (C, H, W)
    idx_left = left_idx.unsqueeze(0).expand(C, H, W)
    idx_right = right_idx.unsqueeze(0).expand(C, H, W)

    # gather along width dimension (dim=2)
    # no_grad helps avoid autograd overhead if not needed
    with torch.no_grad():
        left = torch.gather(rgb_c, 2, idx_left)   # (C, H, W)
        right = torch.gather(rgb_c, 2, idx_right) # (C, H, W)

        # arrange output according to display_mode
        if display_mode == "TAB":
            # stack vertically: (2H, W, C) we'll create (C, 2H, W)
            out = torch.cat([left, right], dim=1)  # (C, 2H, W)
        else:
            # side-by-side: (C, H, 2W)
            out = torch.cat([left, right], dim=2)  # (C, H, 2W)

        # if user wants "Half-SBS" (output width == W), resize on device:
        # you used cv2.resize to (W,H) when display_mode != "Full-SBS" previously.
        # We'll interpret "Half-SBS" as "stacked into W width" and use F.interpolate.
        if display_mode != "Full-SBS":
            # we need to resize to (C, H, W) — do interpolate on device (preserves speed)
            target_w = W
            target_h = H
            out = F.interpolate(out.unsqueeze(0), size=(target_h, target_w), mode="area")[0]  # (C,H,W)

        # clamp and convert to uint8
        out = out.to(torch.float32).clamp(0, 255).to(torch.uint8)

    # convert to (H, W, C) uint8 numpy on CPU
    out_cpu = out.permute(1, 2, 0).contiguous().cpu().numpy()
    return out_cpu
