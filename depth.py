import torch
import torch_directml
import torch.nn.functional as F
from transformers import AutoModelForDepthEstimation
import numpy as np
from threading import Lock

# Initialize DirectML Device
DML = torch_directml.device()
print(f"Using DirectML device: {torch_directml.device_name(0)}")
DTYPE = torch.float16
MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

# Load model with same configuration as example
model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DML).half().eval()
INPUT_H, INPUT_W = 384, 384  # model's native resolution

# Normalization parameters (same as example)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DML).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DML).view(1, 3, 1, 1)

# Warm-up with dummy input
with torch.no_grad():
    dummy = torch.zeros(1, 3, INPUT_H, INPUT_W, device=DML, dtype=DTYPE)
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
    tensor = tensor.unsqueeze(0).to(DML, dtype=DTYPE, non_blocking=True)

    # Resize and normalize (same as pipeline)
    tensor = F.interpolate(tensor, (INPUT_H, INPUT_W), mode='bilinear', align_corners=False)
    tensor = (tensor - MEAN) / STD

    # Inference with thread safety
    with lock:
        depth = model(pixel_values=tensor).predicted_depth  # (1, H, W)

    # Post-processing (same as pipeline)
    h, w = image_rgb.shape[:2]
    depth = F.interpolate(depth.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False)[0, 0]

    # Normalize to [0, 1] (same as pipeline output)
    depth = depth / depth.max().clamp(min=1e-6)
    return depth.cpu().numpy().astype('float32')