from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
from PIL import Image
import os
DEVICE_ID = 0
FP16 = False
DTYPE = torch.float16 if FP16 else torch.float32
# Initialize DirectML Device
def get_device(index=0):
    try:
        try:
            import torch_directml
            if torch_directml.is_available():
                return torch_directml.device(index), f"Using DirectML device: {torch_directml.device_name(index)}"
        except ImportError:
            pass
        if torch.backends.mps.is_available() and index==0:
            return torch.device("mps"), "Using Apple Silicon (MPS) device"
        if torch.cuda.is_available():
            return torch.device("cuda"), f"Using CUDA device: {torch.cuda.get_device_name(index)}"
        else:
            return torch.device("cpu"), "Using CPU device"
    except:
        return torch.device("cpu"), "Using CPU device"
    
DEVICE, DEVICE_INFO = get_device(DEVICE_ID)
# Optimization for CUDA
if "CUDA" in DEVICE_INFO and "NVIDIA" in DEVICE_INFO:
    torch.backends.cudnn.benchmark = True
    if not FP16:
        # Enable TF32 for matrix multiplications
        torch.backends.cuda.matmul.allow_tf32 = True
        # Enable TF32 for cuDNN (convolution operations)
        torch.backends.cudnn.allow_tf32 = True
        # Enable TF32 matrix multiplication for better performance
        torch.set_float32_matmul_precision('high')
    
    # Enable math attention
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"] ="1" # Debug for torch.compile
    
elif "CUDA" in DEVICE_INFO and "AMD" in DEVICE_INFO:
    torch.backends.cudnn.enabled = False # Add for AMD ROCm
    os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1" # Add for AMD ROCm7
    # Enable math attention
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
image  = Image.open("C:/Users/xul/Pictures/test.jpg").convert("RGB")

image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf", use_fast=True)
model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to(DEVICE, dtype=DTYPE)

# prepare image for the model
inputs = image_processor(images=image, return_tensors="pt")

inputs = {k: v.to(DEVICE, dtype=DTYPE) for k, v in inputs.items()}

for k, v in inputs.items():
    print(v.shape)

with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

# interpolate to original size
depth = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),
    size=image.size[::-1],
    mode="bilinear",
    align_corners=False,
).squeeze()

import matplotlib.pyplot as plt
plt.imshow(depth.cpu().detach().numpy(), cmap='inferno')
plt.colorbar()
plt.show()
plt.close()
plt.imshow(depth.cpu().detach().numpy(), cmap='inferno')
plt.colorbar()
plt.savefig("test.png", dpi=300)