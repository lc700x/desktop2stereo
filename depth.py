from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import torch_directml
import numpy as np
from threading import Lock

DML = torch_directml.device()
MODEL_ID = "depth-anything/Depth-Anything-V2-Base-hf"
DTYPE = torch.float16  # Use float16 for DirectML

processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=True)
model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
model = model.to(DML).half().eval()  # Convert entire model to float16


# Warm-up
with torch.no_grad():
    dummy = torch.zeros(1, 3, 384, 384, device=DML, dtype=DTYPE)
    model(pixel_values=dummy)

model_lock = Lock()

def predict_depth(rgb_np: np.ndarray) -> np.ndarray:
    # Convert input to float32 numpy array if it's not already
    # rgb_np = rgb_np.astype(np.float32) if rgb_np.dtype != np.float32 else rgb_np
    rgb_np = np.ascontiguousarray(rgb_np.astype(np.float32))  # Ensure float32 & contiguous
    inputs = processor(images=rgb_np, return_tensors="pt")
    inputs_on_dml = {k: v.to(DML).to(DTYPE) for k, v in inputs.items()}  # Explicitly convert to float16

    with model_lock, torch.no_grad():
        outputs = model(**inputs_on_dml)

    depth = outputs.predicted_depth
    depth_upscaled = torch.nn.functional.interpolate(
        depth.unsqueeze(1),
        size=rgb_np.shape[:2],
        mode="bilinear",
        align_corners=False,
    )
    max_val = depth_upscaled.max()
    depth_normalized = (depth_upscaled / max_val) if max_val > 0 else depth_upscaled
    return depth_normalized[0, 0].cpu().numpy().astype("float32")