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

def process_tensor(img_rgb: np.ndarray, height) -> np.ndarray:
    """
    Process raw BGR image: convert to RGB and apply downscale if set.
    This can be called in a separate thread.
    """
    # Downscale the image if needed
    if height < img_rgb.shape[0]:
        width = int(img_rgb.shape[1] / img_rgb.shape[0] * height)
        img_rgb = cv2.resize(img_rgb, (width, height), interpolation=cv2.INTER_AREA)
    rgb_tensor = torch.from_numpy(img_rgb).to(DEVICE, dtype=DTYPE) # CPU → CPU tensor (uint8)
    return rgb_tensor

def process(img_rgb: np.ndarray, height) -> np.ndarray:
    """
    Process raw BGR image: convert to RGB and apply downscale if set.
    This can be called in a separate thread.
    """
    # Downscale the image if needed
    if height < img_rgb.shape[0]:
        width = int(img_rgb.shape[1] / img_rgb.shape[0] * height)
        img_rgb = cv2.resize(img_rgb, (width, height), interpolation=cv2.INTER_AREA)
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

    # Inference with thread safety
    with lock:
        rgb_tensor = torch.from_numpy(img_rgb).to(DEVICE, dtype=DTYPE) # CPU → CPU tensor (uint8)
        tensor = rgb_tensor.permute(2, 0, 1).float() / 255  # HWC → CHW, 0-1 range
        tensor = tensor.unsqueeze(0) # set to improve performance

        # Resize and normalize (same as pipeline)
        tensor = F.interpolate(tensor, (DEPTH_RESOLUTION, DEPTH_RESOLUTION), mode='bilinear', align_corners=False)
        tensor = (tensor - MEAN) / STD
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

def pad_to_aspect_ratio(img: torch.Tensor, aspect_ratio=(16, 9)):
    """
    Pads an image tensor (C, H, W) to the given aspect ratio with black pixels.

    Args:
        img (torch.Tensor): Input image tensor of shape (C, H, W).
        aspect_ratio (tuple): Desired aspect ratio (W, H). Default is (16, 9).

    Returns:
        torch.Tensor: Padded image tensor (C, H_new, W_new).
    """
    _, H, W = img.shape
    target_w, target_h = aspect_ratio

    # Current aspect ratio
    img_ratio = W / H
    target_ratio = target_w / target_h

    if abs(img_ratio-target_ratio) <= 0.001:
        return img
    elif img_ratio > target_ratio:
        # Too wide → pad height
        new_H = int(round(W / target_ratio))
        pad_total = new_H - H
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        padding = (0, 0, pad_top, pad_bottom)  # (left, right, top, bottom)
    else:
        # Too tall → pad width
        new_W = int(round(H * target_ratio))
        pad_total = new_W - W
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        padding = (pad_left, pad_right, 0, 0)  # (left, right, top, bottom)

    return F.pad(img, padding, mode="constant", value=0)

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
        depth: Tensor, (H, W) float32 in [0..1]
        ipd_uv: interpupillary distance in normalized image width
        depth_strength: multiplier for parallax
        display_mode: "Full-SBS", "Half-SBS", or "TAB" (TAB stacks vertically)
    Returns:
        numpy uint8 image (H, W or H, 2W, 3) depending on mode
    """

    with lock:  # <<< Thread safety for all GPU ops
        # quick checks and canonicalize shapes
        rgb_c = rgb.permute(2, 0, 1).contiguous()
        _, H, W = rgb_c.shape

        # compute per-pixel integer shift (H, W)
        depth_sampled = depth[::8, ::8]
        depth_min = torch.quantile(depth_sampled, 0.2)
        depth_max = torch.quantile(depth_sampled, 0.98)
        depth = (depth - depth_min) / (depth_max - depth_min + 1e-6)
        depth = torch.clamp(depth, 0.0, 1.0)
        
        inv = torch.ones((H, W), device=DEVICE, dtype=DTYPE) - depth
        max_px = ipd_uv * W
        shifts = (inv * max_px * depth_strength / 5).round().to(torch.long, non_blocking=True)

        # half-shift each eye
        shifts_half = (shifts // 2).clamp(min=0, max=W // 2)

        # base x coordinates
        xs = torch.arange(W, device=DEVICE, dtype=torch.long).unsqueeze(0).expand(H, W)

        left_idx = (xs + shifts_half).clamp(0, W - 1)
        right_idx = (xs - shifts_half).clamp(0, W - 1)

        # expand across channels
        idx_left = left_idx.unsqueeze(0).expand(C, H, W)
        idx_right = right_idx.unsqueeze(0).expand(C, H, W)

        with torch.no_grad():
            gen_left = torch.gather(rgb_c, 2, idx_left)
            gen_right = torch.gather(rgb_c, 2, idx_right)
            left = pad_to_aspect_ratio(gen_left)
            right = pad_to_aspect_ratio(gen_right)

            # arrange output according to display_mode
            if display_mode == "TAB":
                out = torch.cat([left, right], dim=1)  # (C, 2H, W)
            else:
                out = torch.cat([left, right], dim=2)  # (C, H, 2W)

            if display_mode != "Full-SBS":
                out = F.interpolate(out.unsqueeze(0), size=left.shape[1:], mode="area")[0]

            out = out.to(torch.float32).clamp(0, 255).to(torch.uint8)

        # convert to (H, W, C) on CPU
        out_cpu = out.permute(1, 2, 0).contiguous().cpu().numpy()

    return out_cpu
