#!/usr/bin/env python3
"""
Fixed stereo_from_depth script: uses depth.py's predict_depth and image_rgb.
This version handles CPU / DirectML backends by ensuring grid_sample runs in float32.
"""

import os
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F

import depth_visualize as depthpy  # your depth.py must be in the same folder

def ensure_device():
    device = getattr(depthpy, "DEVICE", None)
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return device

def to_torch_rgb(image_rgb, device, dtype=torch.float32):
    """Convert HxWx3 image to 1x3xHxW float tensor on device."""
    img = image_rgb
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    t = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(device=device, dtype=dtype)
    return t  # 1x3xHxW

def disparity_from_depth_tensor(depth_torch, baseline_px=80.0, zero_parallax=0.6):
    if depth_torch.dim() == 2:
        d = depth_torch
    else:
        d = depth_torch.squeeze()
    depth_norm = (1.0 - d) *2 # depth.py uses 1=near,0=far -> convert to 0=near,1=far
    disp = baseline_px * (zero_parallax - depth_norm)
    return disp, depth_norm

def warp_with_grid_sample_torch(rgb_t, disp_t, direction='left', align_corners=True):
    """
    Warp using grid_sample. Automatically casts to float32 on CPU/backends that don't support float16.
    rgb_t: 1x3xHxW float tensor on device
    disp_t: HxW tensor (pixels) on same device
    """
    _,_,H,W = rgb_t.shape
    # Prepare coordinate grids (same dtype/device as disp_t)
    xs = torch.linspace(0, W-1, W, device=disp_t.device, dtype=disp_t.dtype).view(1, W).expand(H, W)
    ys = torch.linspace(0, H-1, H, device=disp_t.device, dtype=disp_t.dtype).view(H, 1).expand(H, W)

    half_disp = disp_t / 2.0
    if direction == 'left':
        src_x = xs - half_disp
    else:
        src_x = xs + half_disp
    src_y = ys

    # Normalize to [-1,1]
    nx = (src_x / (W - 1)) * 2.0 - 1.0
    ny = (src_y / (H - 1)) * 2.0 - 1.0
    grid = torch.stack((nx, ny), dim=2).unsqueeze(0)  # 1xHxWx2

    # If running on CPU (or any device that doesn't support float16 grid_sample), use float32
    run_dtype = rgb_t.dtype
    if rgb_t.device.type != 'cuda' and rgb_t.dtype == torch.float16:
        # cast rgb & grid to float32, run, then cast result back to float16
        rgb_fp32 = rgb_t.to(dtype=torch.float32)
        grid_fp32 = grid.to(dtype=torch.float32)
        sampled = F.grid_sample(rgb_fp32, grid_fp32, mode='bilinear',
                                                  padding_mode='zeros', align_corners=align_corners)
        # Cast back to original dtype (here likely float16)
        sampled = sampled.to(dtype=rgb_t.dtype)
    else:
        # safe to call directly (CUDA supports float16)
        sampled = F.grid_sample(rgb_t, grid, mode='bilinear',
                                                  padding_mode='zeros', align_corners=align_corners)
    return sampled  # 1x3xHxW

def tensor_to_bgr_uint8(tensor):
    arr = tensor.squeeze(0).cpu().numpy().transpose(1,2,0)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr

def inpaint_holes_bgr(bgr_img):
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    mask = (gray == 0).astype('uint8') * 255
    if mask.sum() == 0:
        return bgr_img
    filled = cv2.inpaint(bgr_img, mask, 3, cv2.INPAINT_TELEA)
    return filled

def save_all(out_prefix, sbs_bgr, left_bgr, right_bgr, depth_norm_np):
    cv2.imwrite(out_prefix + "_sbs.png", sbs_bgr)
    cv2.imwrite(out_prefix + "_left.png", left_bgr)
    cv2.imwrite(out_prefix + "_right.png", right_bgr)
    depth_vis = (depth_norm_np * 255.0).astype(np.uint8)
    depth_vis_bgr = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(out_prefix + "_depth_norm.png", depth_vis_bgr)
    print("Saved:", out_prefix + "_sbs.png", out_prefix + "_left.png", out_prefix + "_right.png", out_prefix + "_depth_norm.png")

def main(args):
    device = ensure_device()
    print("Using device:", device)
    # Grab image and compute depth via your depth.py predict_depth
    image_rgb = getattr(depthpy, "image_rgb")
    depth_t = depthpy.predict_depth(image_rgb, return_tuple=False, use_temporal_smooth=True)
    print("Depth tensor:", tuple(depth_t.shape), "dtype:", depth_t.dtype, "device:", depth_t.device)

    disp_t, depth_norm_t = disparity_from_depth_tensor(depth_t, baseline_px=args.baseline, zero_parallax=args.zero_parallax)

    # Choose warp dtype: prefer float16 on CUDA, otherwise float32
    if depth_t.device.type == 'cuda' and depth_t.dtype == torch.float16:
        warp_dtype = torch.float16
    else:
        warp_dtype = torch.float32

    rgb_t = to_torch_rgb(image_rgb, device=depth_t.device, dtype=warp_dtype)

    # If disparity is not warp_dtype, cast disp to warp_dtype
    if disp_t.dtype != warp_dtype:
        disp_t = disp_t.to(dtype=warp_dtype)

    # Warp (grid_sample handles internal casting now)
    left_t = warp_with_grid_sample_torch(rgb_t, disp_t, direction='left', align_corners=True)
    right_t = warp_with_grid_sample_torch(rgb_t, disp_t, direction='right', align_corners=True)

    # Convert and inpaint
    left_bgr = tensor_to_bgr_uint8(left_t.to(dtype=torch.float32))
    right_bgr = tensor_to_bgr_uint8(right_t.to(dtype=torch.float32))

    left_filled = inpaint_holes_bgr(left_bgr)
    right_filled = inpaint_holes_bgr(right_bgr)

    sbs = np.concatenate([left_filled, right_filled], axis=1)
    out_prefix = args.out_prefix
    save_all(out_prefix, sbs, left_filled, right_filled, depth_norm_t.cpu().numpy())
    print("Done. Try --baseline and --zero_parallax to tune the effect.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=float, default=10.0)
    parser.add_argument("--zero_parallax", type=float, default=0.6)
    parser.add_argument("--out_prefix", type=str, default="stereo_sbs_from_depthpy")
    args = parser.parse_args()
    main(args)
