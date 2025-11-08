from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
from PIL import Image
import requests
DEVICE_ID = 0
FP16 = True
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
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
image  = Image.open("C:/Users/zjuli/Pictures/test.png").convert("RGB")

image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf", use_fast=True)
model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to(DEVICE, dtype=DTYPE)

# prepare image for the model
inputs = image_processor(images=image, return_tensors="pt")

inputs = {k: v.to(DEVICE, dtype=DTYPE) for k, v in inputs.items()}

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