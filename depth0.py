# depth.py
import torch
torch.set_num_threads(1)  # Set to avoid high CPU usage caused by default full threads
import torch.nn.functional as F
from transformers import AutoModelForDepthEstimation
import numpy as np
from threading import Lock
import cv2
from utils import DEVICE_ID, MODEL_ID, CACHE_PATH, FP16, DEPTH_RESOLUTION

# Model configuration
DTYPE = torch.float16 if FP16 else torch.float32  # Use float32 for DirectML compatibility

# Initialize DirectML Device
def get_device(index=0):
    """Returns a torch.device and a human-readable device info string."""
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

# Get the device and print information
DEVICE, DEVICE_INFO = get_device(DEVICE_ID)

# Enable cudnn benchmark if CUDA is available
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# Output Info
print(f"{DEVICE_INFO}")
print(f"Model: {MODEL_ID}")

# Load model with same configuration as example
model = AutoModelForDepthEstimation.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if FP16 else torch.float32,
    cache_dir=CACHE_PATH,
    weights_only=True
).to(DEVICE).eval()

if FP16:
    model.half()

MODEL_DTYPE = next(model.parameters()).dtype
# Normalization parameters (same as example)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

# Warm-up with dummy input
with torch.no_grad():
    dummy = torch.zeros(1, 3, DEPTH_RESOLUTION, DEPTH_RESOLUTION, device=DEVICE, dtype=MODEL_DTYPE)
    model(pixel_values=dummy)

lock = Lock()

# main functions
def process_tensor(img_rgb: np.ndarray, height) -> np.ndarray:
    """
    Process raw BGR image: convert to RGB and apply downscale if set.
    This can be called in a separate thread.
    """
    # Downscale the image if needed
    if height < img_rgb.shape[0]:
        width = int(img_rgb.shape[1] / img_rgb.shape[0] * height)
        img_rgb = cv2.resize(img_rgb, (width, height), interpolation=cv2.INTER_AREA)
    rgb_tensor = torch.from_numpy(img_rgb).to(DEVICE, dtype=DTYPE)
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

    # Inference with thread safety
    with lock:
        tensor = tensor.permute(2, 0, 1).float().contiguous() / 255  # HWC → CHW, 0-1 range
        tensor = tensor.unsqueeze(0) # set to improve performance

        # Resize and normalize (same as pipeline)
        tensor = F.interpolate(tensor, (DEPTH_RESOLUTION, DEPTH_RESOLUTION), mode='bilinear', align_corners=False)
        tensor = ((tensor - MEAN) / STD).contiguous()  # ensure contiguous before model
        with torch.no_grad():
            tensor = tensor.to(dtype=MODEL_DTYPE).contiguous()
            depth = model(pixel_values=tensor).predicted_depth  # (1, H, W)

    # Post-processing (same as pipeline)
    h, w = image_rgb.shape[:2]
    depth = F.interpolate(depth.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False)[0, 0]

    # Normalize to [0, 1] (same as pipeline output)
    depth = depth / depth.max().clamp(min=1e-6)
    return depth
    # return depth.detach().cpu().numpy().astype('float32')
    
def predict_depth_tensor(image_rgb: np.ndarray) -> np.ndarray:
    """
    Predict depth map from RGB image (similar to pipeline example but optimized for DirectML)
    Args:
        image_rgb: Input RGB image as numpy array (H, W, 3) in uint8 format
    Returns:
        Depth map as numpy array (H, W) normalized to [0, 1]
    """
   
    with lock: 
        # Convert to tensor and normalize (similar to pipeline's preprocessing)
        tensor = torch.from_numpy(image_rgb).to(DEVICE, dtype=DTYPE, non_blocking=True) # CPU → CPU tensor (uint8)

        # Inference with thread safety
        rgb_c = tensor.permute(2, 0, 1).contiguous()
        tensor = rgb_c / 255
        tensor = tensor.unsqueeze(0) # set to improve performance and normalize to (0,1)

        # Resize and normalize (same as pipeline)
        tensor = F.interpolate(tensor, (DEPTH_RESOLUTION, DEPTH_RESOLUTION), mode='bilinear', align_corners=False)
        tensor = ((tensor - MEAN) / STD).contiguous()  # ensure contiguous before model
        with torch.no_grad():
            tensor = tensor.to(dtype=MODEL_DTYPE).contiguous()
            depth = model(pixel_values=tensor).predicted_depth  # (1, H, W)

        # Post-processing (same as pipeline)
        h, w = image_rgb.shape[:2]
        depth = F.interpolate(depth.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False)[0, 0]

        # Normalize to [0, 1] (same as pipeline output)
        depth = depth / depth.max().clamp(min=1e-6)
        return depth, rgb_c
        # return depth.detach().cpu().numpy().astype('float32')
    
def make_sbs(rgb_c, depth, ipd_uv=0.064, depth_strength=1.0, display_mode="Half-SBS"):
    """
    Predict depth map from RGB image (similar to pipeline example but optimized for DirectML)
    Args:
        image_rgb: Input RGB image as numpy array (H, W, 3) in uint8 format
    Returns:
        Depth map as numpy array (H, W) normalized to [0, 1]
    """
    with lock:
        # Inference with thread safety
        C,H,W = rgb_c.shape
        inv = torch.ones((H, W), device=DEVICE, dtype=DTYPE) - depth
        max_px = ipd_uv * W
        shifts = (inv * max_px * depth_strength / 10).round().to(torch.long, non_blocking=True)
        shifts_half = (shifts//2).clamp(0, W//2)

        xs = torch.arange(W, device=DEVICE).unsqueeze(0).expand(H,W)
        left_idx = (xs + shifts_half).clamp(0,W-1)
        right_idx = (xs - shifts_half).clamp(0,W-1)
        idx_left = left_idx.unsqueeze(0).expand(C,H,W)
        idx_right = right_idx.unsqueeze(0).expand(C,H,W)
        gen_left = torch.gather(rgb_c,2,idx_left)
        gen_right = torch.gather(rgb_c,2,idx_right)

        def pad_to_aspect(img, target_ratio=(16,9)):
            _, h, w = img.shape
            t_w, t_h = target_ratio
            r_img = w/h
            r_t = t_w/t_h
            if abs(r_img-r_t)<0.001:
                return img
            elif r_img>r_t:
                new_H = int(round(w/r_t))
                pad_top = (new_H-h)//2
                pad_bottom = new_H-h-pad_top
                return F.pad(img,(0,0,pad_top,pad_bottom),value=0)
            else:
                new_W = int(round(h*r_t))
                pad_left = (new_W-w)//2
                pad_right = new_W-w-pad_left
                return F.pad(img,(pad_left,pad_right,0,0),value=0)

        left = pad_to_aspect(gen_left)
        right = pad_to_aspect(gen_right)

        if display_mode=="TAB":
            out = torch.cat([left,right],dim=1)
        else:
            out = torch.cat([left,right],dim=2)

        if display_mode!="Full-SBS":
            out = F.interpolate(out.unsqueeze(0),size=left.shape[1:],mode="area")[0]

        out = out.clamp(0,255).to(torch.uint8)
    sbs = out.permute(1,2,0).contiguous().cpu().numpy()
    return sbs