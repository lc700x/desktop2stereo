# depth.py (optimized with TinyRefiner for FPS)
import torch
torch.set_num_threads(1)
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForDepthEstimation
import numpy as np
from threading import Lock
import cv2
from utils import DEVICE_ID, MODEL_ID, CACHE_PATH, FP16, DEPTH_RESOLUTION

# Model configuration
DTYPE = torch.float16 if FP16 else torch.float32

# Initialize DirectML Device
def get_device(index=0):
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

DEVICE, DEVICE_INFO = get_device(DEVICE_ID)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

print(f"{DEVICE_INFO}")
print(f"Model: {MODEL_ID}")

# Load depth model
model = AutoModelForDepthEstimation.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if FP16 else torch.float32,
    cache_dir=CACHE_PATH,
    weights_only=True
).to(DEVICE).eval()
if FP16:
    model.half()
MODEL_DTYPE = next(model.parameters()).dtype
MEAN = torch.tensor([0.485,0.456,0.406], device=DEVICE).view(1,3,1,1)
STD = torch.tensor([0.229,0.224,0.225], device=DEVICE).view(1,3,1,1)
with torch.no_grad():
    dummy = torch.zeros(1,3,DEPTH_RESOLUTION,DEPTH_RESOLUTION, device=DEVICE, dtype=MODEL_DTYPE)
    model(pixel_values=dummy)

lock = Lock()

# --- TinyRefiner definition ---
class TinyRefiner(nn.Module):
    def __init__(self, in_channels=10, out_channels=3, features=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, features, 3, padding=1)
        self.conv2 = nn.Conv2d(features, features, 3, padding=1)
        self.conv3 = nn.Conv2d(features, features, 3, padding=1)
        self.conv4 = nn.Conv2d(features, out_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        return torch.sigmoid(x)

# Load refiner weights
REFINER_WEIGHTS_PATH = "refiner_kitti.pt"
refiner = TinyRefiner().to(DEVICE).eval()
try:
    sd = torch.load(REFINER_WEIGHTS_PATH, map_location=DEVICE, weights_only=True)
    refiner.load_state_dict(sd)
except Exception as e:
    print(f"Refiner failed to load, falling back to CPU")
    sd = torch.load(REFINER_WEIGHTS_PATH, map_location="cpu", weights_only=True)
    refiner.load_state_dict(sd)

# --- main functions ---
def process_tensor(img_rgb: np.ndarray, height) -> torch.Tensor:
    if height < img_rgb.shape[0]:
        width = int(img_rgb.shape[1]/img_rgb.shape[0]*height)
        img_rgb = cv2.resize(img_rgb,(width,height), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(img_rgb).to(DEVICE, dtype=DTYPE)

def process(img_rgb: np.ndarray, height) -> np.ndarray:
    if height < img_rgb.shape[0]:
        width = int(img_rgb.shape[1]/img_rgb.shape[0]*height)
        img_rgb = cv2.resize(img_rgb,(width,height), interpolation=cv2.INTER_AREA)
    return img_rgb

def predict_depth(image_rgb: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(image_rgb).to(DEVICE,dtype=DTYPE)
    with lock:
        tensor = tensor.permute(2,0,1).float().unsqueeze(0)/255
        tensor = F.interpolate(tensor,(DEPTH_RESOLUTION,DEPTH_RESOLUTION),mode='bilinear',align_corners=False)
        tensor = ((tensor-MEAN)/STD).contiguous()
        with torch.no_grad():
            tensor = tensor.to(dtype=MODEL_DTYPE)
            depth = model(pixel_values=tensor).predicted_depth
    h,w = image_rgb.shape[:2]
    depth = F.interpolate(depth.unsqueeze(1),size=(h,w),mode='bilinear',align_corners=False)[0,0]
    return depth/depth.max().clamp(min=1e-6)

def predict_depth_tensor(image_rgb: np.ndarray) -> tuple:
    with lock:
        tensor = torch.from_numpy(image_rgb).to(DEVICE,dtype=DTYPE)
        rgb_c = tensor.permute(2,0,1).contiguous()
        tensor = rgb_c.unsqueeze(0)/255
        tensor = F.interpolate(tensor,(DEPTH_RESOLUTION,DEPTH_RESOLUTION),mode='bilinear',align_corners=False)
        tensor = ((tensor-MEAN)/STD).contiguous()
        with torch.no_grad():
            tensor = tensor.to(dtype=MODEL_DTYPE)
            depth = model(pixel_values=tensor).predicted_depth
        h,w = image_rgb.shape[:2]
        depth = F.interpolate(depth.unsqueeze(1),size=(h,w),mode='bilinear',align_corners=False)[0,0]
        depth = depth/depth.max().clamp(min=1e-6)
        
        # Normalize depth with adaptive range
        depth_sampled = depth[::8, ::8].to(torch.float32)
        depth_min = torch.quantile(depth_sampled, 0.2)
        depth_max = torch.quantile(depth_sampled, 0.98)

        depth = (depth - depth_min) / (depth_max - depth_min + 1e-6)
        depth = depth.clamp(0, 1)
        
        return depth, rgb_c

def make_sbs(rgb_c, depth, ipd_uv=0.064, depth_strength=1.0, display_mode="Half-SBS"):
    with lock:
        C,H,W = rgb_c.shape
        inv = torch.ones((H,W), device=DEVICE,dtype=DTYPE)-depth
        max_px = ipd_uv*W
        shifts = (inv*max_px*depth_strength/10).round().to(torch.long)
        shifts_half = (shifts//2).clamp(0,W//2)

        xs = torch.arange(W, device=DEVICE).unsqueeze(0).expand(H,W)
        idx_left = (xs+shifts_half).clamp(0,W-1).unsqueeze(0).expand(C,H,W)
        idx_right = (xs-shifts_half).clamp(0,W-1).unsqueeze(0).expand(C,H,W)
        gen_left = torch.gather(rgb_c,2,idx_left)
        gen_right = torch.gather(rgb_c,2,idx_right)

        def pad_to_aspect(img,target_ratio=(16,9)):
            _,h,w = img.shape
            t_w,t_h = target_ratio
            r_img = w/h
            r_t = t_w/t_h
            if abs(r_img-r_t)<0.001: return img
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
        rgb_c = pad_to_aspect(rgb_c)
        depth = pad_to_aspect(depth.unsqueeze(0)).squeeze(0)

        try:
            ref_input = torch.cat([rgb_c/255.0, left/255.0, right/255.0, depth.unsqueeze(0)], dim=0).unsqueeze(0).float()
            with torch.no_grad():
                ref_out = refiner(ref_input).squeeze(0)
            right_f = (ref_out * 255.0 ).half()
        except Exception:
            right_f = right.half()

        if display_mode=="TAB":
            out = torch.cat([left,right_f],dim=1)
        else:
            out = torch.cat([left,right_f],dim=2)

        if display_mode!="Full-SBS":
            out = F.interpolate(out.unsqueeze(0),size=left.shape[1:],mode="area")[0]

        out = out.clamp(0,255).to(torch.uint8)
    sbs = out.permute(1,2,0).contiguous().cpu().numpy()
    return sbs
