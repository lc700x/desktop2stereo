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

# Pre-allocate normalization tensors
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1).to(MODEL_DTYPE)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1).to(MODEL_DTYPE)

# Warm-up with dummy input
with torch.no_grad():
    dummy = torch.zeros(1, 3, DEPTH_RESOLUTION, DEPTH_RESOLUTION, device=DEVICE, dtype=MODEL_DTYPE)
    model(pixel_values=dummy)

lock = Lock()

def process_tensor(img_rgb: np.ndarray, height: int) -> torch.Tensor:
    """Process raw RGB image and return as tensor."""
    if height < img_rgb.shape[0]:
        width = int(img_rgb.shape[1] / img_rgb.shape[0] * height)
        img_rgb = cv2.resize(img_rgb, (width, height), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(img_rgb).to(DEVICE, dtype=DTYPE)

def process(img_rgb: np.ndarray, height: int) -> np.ndarray:
    """Process raw RGB image and return as numpy array."""
    if height < img_rgb.shape[0]:
        width = int(img_rgb.shape[1] / img_rgb.shape[0] * height)
        img_rgb = cv2.resize(img_rgb, (width, height), interpolation=cv2.INTER_AREA)
    return img_rgb

@torch.no_grad()
def predict_depth(image_rgb: np.ndarray) -> np.ndarray:
    """Predict depth map from RGB image."""
    # Convert to tensor and normalize
    tensor = torch.from_numpy(image_rgb).to(DEVICE, dtype=DTYPE, non_blocking=True)
    tensor = tensor.permute(2, 0, 1).float().div_(255).unsqueeze_(0).contiguous()
    
    # Resize and normalize
    tensor = F.interpolate(tensor, (DEPTH_RESOLUTION, DEPTH_RESOLUTION), 
                          mode='bilinear', align_corners=False)
    tensor = tensor.sub_(MEAN).div_(STD).to(MODEL_DTYPE).contiguous()

    # Inference with thread safety
    with lock:
        depth = model(pixel_values=tensor).predicted_depth

    # Post-processing
    h, w = image_rgb.shape[:2]
    depth = F.interpolate(depth.unsqueeze(1), size=(h, w), 
                         mode='bilinear', align_corners=False)[0, 0]
    
    # Normalize to [0, 1]
    depth.div_(depth.max().clamp_(min=1e-6))
    return depth

@torch.no_grad()
def predict_depth_tensor(img_rgb) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict depth map from RGB image and return both depth and RGB tensors."""
    if isinstance(img_rgb, torch.Tensor):
        img_rgb = img_rgb.cpu().numpy()

    assert img_rgb.ndim == 3 and img_rgb.shape[2] == 3, \
        f"Expected HWC numpy image, got {img_rgb.shape}"

    # Convert to tensor and normalize
    rgb_tensor = torch.from_numpy(img_rgb).to(DEVICE, dtype=DTYPE)
    tensor = rgb_tensor.permute(2, 0, 1).float().div_(255).unsqueeze_(0).contiguous()

    # Resize and normalize
    tensor = F.interpolate(tensor, (DEPTH_RESOLUTION, DEPTH_RESOLUTION), 
                          mode='bilinear', align_corners=False)
    tensor = tensor.sub_(MEAN).div_(STD).to(MODEL_DTYPE).contiguous()

    # Inference with thread safety
    with lock:
        depth = model(pixel_values=tensor).predicted_depth

    # Post-processing
    h, w = img_rgb.shape[:2]
    depth = F.interpolate(depth.unsqueeze(1), size=(h, w), 
                         mode='bilinear', align_corners=False)[0, 0]
    
    # Normalize to [0, 1]
    depth.div_(depth.max().clamp_(min=1e-6))
    return depth, rgb_tensor