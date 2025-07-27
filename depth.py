import torch, torch_directml, torch.nn.functional as F
from transformers import AutoModelForDepthEstimation
import numpy as np
from threading import Lock

DML = torch_directml.device()
print(f"Using DirectML device: {torch_directml.device_name(0)}")
DTYPE = torch.float16
MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

model = (AutoModelForDepthEstimation.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DML).half().eval())
INPUT_H, INPUT_W = 384, 384 # model’s native resolution
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DML).view(1,3,1,1)
STD  = torch.tensor([0.229, 0.224, 0.225], device=DML).view(1,3,1,1)

with torch.no_grad(): # warm-up
    dummy = torch.zeros(1,3,INPUT_H,INPUT_W, device=DML, dtype=DTYPE)
    model(pixel_values=dummy)

lock = Lock()

@torch.no_grad()
def predict_depth(rgb_np: np.ndarray) -> np.ndarray:
    # upload only once
    tensor = torch.from_numpy(rgb_np)              # CPU → CPU tensor (uint8)
    tensor = tensor.permute(2,0,1).float() / 255.  # HWC → CHW, 0-1 range
    tensor = tensor.unsqueeze(0).to(DML, dtype=DTYPE, non_blocking=True)

    # GPU resize & normalise
    tensor = F.interpolate(tensor, (INPUT_H, INPUT_W), mode='bilinear', align_corners=False)
    tensor = (tensor - MEAN) / STD

    with lock, torch.no_grad():
        depth = model(pixel_values=tensor).predicted_depth  # (1, H, W)

    # upscale back to original and normalise once
    h, w = rgb_np.shape[:2]
    depth = F.interpolate(depth.unsqueeze(1), size=(h,w), mode='bilinear', align_corners=False)[0,0]

    depth /= depth.max().clamp(min=1e-6)           # keep on GPU for a sec
    depth_gpu = depth.detach()
    return depth.cpu().numpy().astype('float32')   # GPU → CPU