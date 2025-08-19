


import torch
from transformers import Dinov2Model, DepthAnythingConfig, DepthAnythingForDepthEstimation
import os

DEVICE = "cpu"
MODEL_ID = "C:/Users/xul/Downloads/video_depth_anything_large"
MODEL_PATH = MODEL_ID + "/video_depth_anything_vitl.pth"
DTYPE = torch.float16
BACK_BONE = "facebook/dinov2-small"
OUTPUT_ID = "depth_anything_video_large_hf" # cannot use "-"
# Load DINOv2 backbone
dinov2 = Dinov2Model.from_pretrained(BACK_BONE)

# Create Depth Anything config
config = DepthAnythingConfig(
    backbone=BACK_BONE,
    use_pretrained_backbone=False,
    depth_estimation_type="relative"
)

# Initialize Depth Anything model
model = DepthAnythingForDepthEstimation(config)

# Inject the pretrained DINOv2 backbone
model.backbone = dinov2

# Load the rest of the checkpoint (decoder, etc.)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)

# print("Missing keys:", missing_keys)
# print("Unexpected keys:", unexpected_keys)

# Save the converted model
model.save_pretrained(OUTPUT_ID)
print(f"✅ Model saved to {OUTPUT_ID}")
