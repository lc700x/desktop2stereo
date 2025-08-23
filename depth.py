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