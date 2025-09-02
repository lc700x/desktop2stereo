# depth.py (optimized with TinyRefiner for FPS)
import torch
torch.set_num_threads(1)
import torch.nn.functional as F
from transformers import AutoModelForDepthEstimation
import numpy as np
from threading import Lock
import cv2
from utils import DEVICE_ID, MODEL_ID, CACHE_PATH, FP16, DEPTH_RESOLUTION, DILATION_SIZE, AA_STRENTH

# Model configuration
DTYPE = torch.float16 if FP16 else torch.float32

# Initialize DirectML Device
def get_device(index=0):
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

if FP16:
    model.half()

MODEL_DTYPE = next(model.parameters()).dtype
MEAN = torch.tensor([0.485,0.456,0.406], device=DEVICE).view(1,3,1,1)
STD = torch.tensor([0.229,0.224,0.225], device=DEVICE).view(1,3,1,1)
with torch.no_grad():
    dummy = torch.zeros(1,3,DEPTH_RESOLUTION,DEPTH_RESOLUTION, device=DEVICE, dtype=MODEL_DTYPE)
    model(pixel_values=dummy)

lock = Lock()

# main functions
def depth_guided_fill(rgb: torch.Tensor, depth: torch.Tensor, idx_map: torch.Tensor) -> torch.Tensor:
    """
    Depth-guided hole filling for horizontally shifted images.
    - rgb: [C, H, W] (same dtype/device as rgb_c)
    - depth: [H, W] normalized in [0,1]
    - idx_map: [H, W] long tensor giving source column indices for each target column
      (this is the same idx_left / idx_right used for torch.take_along_dim)
    Returns:
      rgb with holes filled in-place (but still returns the tensor).
    """
    # Ensure proper types
    C, H, W = rgb.shape
    device = rgb.device

    # Work in float for interpolation/copy
    orig_dtype = rgb.dtype
    rgb = rgb.to(torch.float32)

    # Loop per row (vectorized inside each row)
    for y in range(H):
        idx_row = idx_map[y]  # shape [W], dtype long/int
        # Build occupancy: which source columns were used by any target
        occ = torch.zeros((W,), dtype=torch.bool, device=device)
        # Mark used source columns
        occ[idx_row] = True

        # If no holes on this row, continue
        if occ.all():
            continue

        valid_cols = torch.nonzero(occ).squeeze(1)
        hole_cols = torch.nonzero(~occ).squeeze(1)

        # If no valid columns (degenerate), skip
        if valid_cols.numel() == 0:
            continue

        # Sort valid columns (should already be sorted but be safe)
        valid_cols, _ = torch.sort(valid_cols)

        # Use searchsorted to find nearest valid columns for each hole
        # insertion indices i satisfy valid_cols[i-1] < hole <= valid_cols[i]
        ins = torch.searchsorted(valid_cols, hole_cols)

        # Determine left/right existence
        left_exists = ins > 0
        right_exists = ins < valid_cols.numel()

        # Build left_idx / right_idx (valid column indices) with safe defaults
        left_idx = torch.zeros_like(ins)
        right_idx = torch.zeros_like(ins)
        if left_exists.any():
            left_idx[left_exists] = valid_cols[ins[left_exists] - 1]
        if right_exists.any():
            right_idx[right_exists] = valid_cols[ins[right_exists]]

        # Gather depths at those source columns (for selection)
        # If left/right do not exist for a hole, their depths will be ignored
        left_depth = depth[y, left_idx]
        right_depth = depth[y, right_idx]

        # For each channel, pick values from the side with greater depth (farther away)
        for c in range(C):
            row = rgb[c, y]  # shape [W]
            left_vals = row[left_idx]   # values at nearest-left valid col (for each hole)
            right_vals = row[right_idx] # nearest-right valid col values

            # Prepare filled values tensor (float32)
            filled = torch.empty_like(left_vals)

            # Cases:
            # 1) only left exists -> copy left
            only_left = left_exists & (~right_exists)
            if only_left.any():
                filled[only_left] = left_vals[only_left]

            # 2) only right exists -> copy right
            only_right = (~left_exists) & right_exists
            if only_right.any():
                filled[only_right] = right_vals[only_right]

            # 3) both exist -> choose side with greater depth (prefer background)
            both = left_exists & right_exists
            if both.any():
                choose_left = left_depth[both] >= right_depth[both]
                # where choose_left True -> left, else -> right
                idx_both = torch.nonzero(both).squeeze(1)
                if idx_both.numel() > 0:
                    # assign elementwise
                    bl = idx_both[choose_left]
                    br = idx_both[~choose_left]
                    if bl.numel() > 0:
                        filled[bl] = left_vals[bl]
                    if br.numel() > 0:
                        filled[br] = right_vals[br]

            # Edge safety: if neither side exists (shouldn't happen), copy nearest valid column (fallback)
            neither = (~left_exists) & (~right_exists)
            if neither.any():
                # fallback: copy mean of whole row valid pixels
                fallback_val = row[valid_cols].mean() if valid_cols.numel() > 0 else 0.0
                filled[neither] = fallback_val

            # Write back into the row at hole columns
            row[hole_cols] = filled
            rgb[c, y] = row

    # Convert back to original dtype and return
    return rgb.to(orig_dtype)


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

def edge_dilate(depth: torch.Tensor, dilation_size: int = 2) -> torch.Tensor:
    """
    Perform edge dilation on depth map using max pooling (PyTorch / DirectML compatible).

    Args:
        depth (torch.Tensor): Normalized depth map tensor with shape [H, W] or [B, 1, H, W], values in [0,1].
        dilation_size (int): Size of the dilation kernel; 0 disables dilation.

    Returns:
        torch.Tensor: Dilated depth map tensor with the same shape as input.
    """
    if dilation_size <= 0:
        return depth

    # Ensure 4D shape for pooling: [B, C, H, W]
    if depth.dim() == 2:
        depth = depth.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    elif depth.dim() == 3:
        depth = depth.unsqueeze(1)  # [B,1,H,W]

    kernel_size = dilation_size if dilation_size % 2 == 1 else dilation_size + 1  # odd kernel size is typical

    # Apply max pooling as morphological dilation
    dilated = F.max_pool2d(depth, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    # Remove extra dims if needed
    if dilated.shape[0] == 1 and dilated.shape[1] == 1:
        return dilated.squeeze(0).squeeze(0)
    elif dilated.shape[1] == 1:
        return dilated.squeeze(1)
    else:
        return dilated

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

def predict_depth(image_rgb: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(image_rgb).to(DEVICE,dtype=DTYPE)
    tensor = tensor.permute(2,0,1).float().unsqueeze(0)/255
    tensor = F.interpolate(tensor,(DEPTH_RESOLUTION,DEPTH_RESOLUTION),mode='bilinear',align_corners=False)
    tensor = ((tensor-MEAN)/STD).contiguous()
    with lock:
        with torch.no_grad():
            tensor = tensor.to(dtype=MODEL_DTYPE)
            depth = model(pixel_values=tensor).predicted_depth
    h,w = image_rgb.shape[:2]
    depth = F.interpolate(depth.unsqueeze(1),size=(h,w),mode='bilinear',align_corners=False)[0,0]
    # Normalize depth with adaptive range
    if "MPS" in DEVICE_INFO:
        depth_sampled = depth[::8, ::8].to(torch.float32)
        depth_min = torch.quantile(depth_sampled, 0.2)
        depth_max = torch.quantile(depth_sampled, 0.98)

        depth = (depth - depth_min) / (depth_max - depth_min + 1e-6)
        depth = depth.clamp(0, 1)
    else:
        depth = depth / depth.max().clamp(min=1e-6)

    depth = edge_dilate(depth, dilation_size=DILATION_SIZE)
    depth = anti_alias(depth, strength=AA_STRENTH)
    
    return depth

def predict_depth_tensor(image_rgb: np.ndarray) -> tuple:
    tensor = torch.from_numpy(image_rgb).to(DEVICE,dtype=DTYPE)
    rgb_c = tensor.permute(2,0,1).contiguous()
    tensor = rgb_c.unsqueeze(0)/255
    tensor = F.interpolate(tensor,(DEPTH_RESOLUTION,DEPTH_RESOLUTION),mode='bilinear',align_corners=False)
    tensor = ((tensor-MEAN)/STD).contiguous()
    with lock:
        with torch.no_grad():
            tensor = tensor.to(dtype=MODEL_DTYPE)
            depth = model(pixel_values=tensor).predicted_depth
    h,w = image_rgb.shape[:2]
    depth = F.interpolate(depth.unsqueeze(1),size=(h,w),mode='bilinear',align_corners=False)[0,0]
    
    if "MPS" in DEVICE_INFO:
        depth_sampled = depth[::8, ::8].to(torch.float32)
        depth_min = torch.quantile(depth_sampled, 0.2)
        depth_max = torch.quantile(depth_sampled, 0.98)

        depth = (depth - depth_min) / (depth_max - depth_min + 1e-6)
        depth = depth.clamp(0, 1)
    else:
        depth = depth / depth.max().clamp(min=1e-6)
        
    depth = edge_dilate(depth, dilation_size=DILATION_SIZE)
    depth = anti_alias(depth, strength=AA_STRENTH)
    
    return depth, rgb_c

def make_sbs(rgb_c, depth, ipd_uv=0.064, depth_strength=1.0, display_mode="Half-SBS"):
    with lock:
        C, H, W = rgb_c.shape
        device = rgb_c.device

        # Precompute pixel coordinates (cache this tensor outside if called repeatedly!)
        xs = torch.arange(W, device=device).view(1, -1)  # shape [1,W]

        # Depth inversion & shifts
        inv = 1.0 - depth
        max_px = ipd_uv * W
        shifts_half = ((inv * max_px * float(depth_strength) / 10) / 2).round().clamp(0, W // 2).to(torch.int32)

        # Build shifted indices efficiently (broadcasting instead of expand)
        idx_left = (xs + shifts_half).clamp(0, W - 1)
        idx_right = (xs - shifts_half).clamp(0, W - 1)

        # Index select across width (faster than gather with full expand)
        gen_left = torch.take_along_dim(rgb_c, idx_left.unsqueeze(0).expand(C, H, W), dim=2)
        gen_right = torch.take_along_dim(rgb_c, idx_right.unsqueeze(0).expand(C, H, W), dim=2)

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

        left, right = pad_to_aspect(gen_left), pad_to_aspect(gen_right)

        if display_mode == "TAB":
            out = torch.cat([left, right], dim=1)
        else:  # SBS
            out = torch.cat([left, right], dim=2)
        if display_mode != "Full-SBS":
            out = F.interpolate(out.unsqueeze(0), size=left.shape[1:], mode="area")[0]
        sbs = out.clamp(0, 255).to(torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy()
        return sbs
