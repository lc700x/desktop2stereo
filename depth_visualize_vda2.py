# depth.py
import torch
torch.set_num_threads(1)
import torch.nn.functional as F
from torchvision.transforms.functional import adjust_contrast
from transformers import AutoModelForDepthEstimation
import numpy as np
from threading import Lock
from PIL import Image
img  = Image.open("assets/cats.jpg").convert("RGB")
image_rgb = np.array(img)
AA_STRENGTH = 4
DILATION_SIZE = 0
FP16 = False
DTYPE = torch.float16 if FP16 else torch.float32
CACHE_PATH = "models"
DEVICE_ID = 0
MODEL_ID = "depth-anything/Video-Depth-Anything-Small"
DEPTH_RESOLUTION = 336

def get_device(index=0):
    try:
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
    except:
        return torch.device("cpu"), "Using CPU device"
DEVICE, DEVICE_INFO = get_device(DEVICE_ID)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

print(f"{DEVICE_INFO}")
# print(f"Model: {MODEL_ID}")

def get_video_depth_anything_model(model_id=MODEL_ID):
    """ Load Video Depth Anything model from HuggingFace hub. """
    from huggingface_hub import hf_hub_download
    from .video_depth_anything.vda2_s import VideoDepthAnything
    # Preparation for video depth anything models
    encoder_dict = {'depth-anything/Video-Depth-Anything-Small': 'vits',
                    'depth-anything/Video-Depth-Anything-Base': 'vitb',
                    'depth-anything/Video-Depth-Anything-Large': 'vitl',
                    'depth-anything/Metric-Video-Depth-Anything-Small': 'vits',
                    'depth-anything/Metric-Video-Depth-Anything-Base': 'vitb',
                    'depth-anything/Metric-Video-Depth-Anything-Large': 'vitl'}

    encoder = encoder_dict.get(model_id, 'vits')

    if 'depth-anything/Video-Depth-Anything' in model_id:
        checkpoint_name = f'video_depth_anything_{encoder}.pth'
    elif 'depth-anything/Metric-Video-Depth-Anything' in model_id:
        checkpoint_name = f'metric_video_depth_anything_{encoder}.pth'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    checkpoint_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name, cache_dir=CACHE_PATH)

    model = VideoDepthAnything(**model_configs[encoder])
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True), strict=True)
    return model

# Load model
if 'Video-Depth-Anything' in MODEL_ID:
    model = get_video_depth_anything_model(MODEL_ID)

else:
    # Load depth model
    model = AutoModelForDepthEstimation.from_pretrained(
        MODEL_ID,
        dtype=torch.float16 if FP16 else torch.float32,
        cache_dir=CACHE_PATH,
        weights_only=True
    ).to(DEVICE).eval()


model = model.to(DEVICE, dtype=DTYPE).eval()
if FP16:
    model.half()

MODEL_DTYPE = next(model.parameters()).dtype
MEAN = torch.tensor([0.485,0.456,0.406], device=DEVICE).view(1,3,1,1)
STD = torch.tensor([0.229,0.224,0.225], device=DEVICE).view(1,3,1,1)


# with torch.no_grad():
#     dummy = torch.zeros(1,3,DEPTH_RESOLUTION,DEPTH_RESOLUTION, device=DEVICE, dtype=MODEL_DTYPE)
#     model(pixel_values=dummy)

lock = Lock()



def apply_stretch(x: torch.Tensor, low: float = 2.0, high: float = 98.0) -> torch.Tensor:
    """
    Percentile-based clipping + normalization.
    Fully DirectML compatible (no torch.clamp).
    """
    if x.numel() < 2:
        return x

    # Downsample to reduce cost
    x_sampled = x[::8, ::8] if x.dim() == 2 else x.flatten()[::8]
    flat = x_sampled.flatten()

    if flat.numel() < 2:
        return x

    # Sort values
    vals, _ = torch.sort(flat)

    # Compute percentile indices
    k_low = int((low / 100.0) * (vals.numel() - 1))
    k_high = int((high / 100.0) * (vals.numel() - 1))
    k_low = max(0, min(k_low, vals.numel() - 1))
    k_high = max(0, min(k_high, vals.numel() - 1))

    lo = vals[k_low]
    hi = vals[k_high]

    # Avoid divide-by-zero
    scale = hi - lo
    if scale <= 0:
        return torch.zeros_like(x)

    # Manual clamp: DirectML supports min/max
    x = torch.maximum(x, lo)
    x = torch.minimum(x, hi)

    return (x - lo) / (scale + 1e-6)

def apply_gamma(depth: torch.Tensor, gamma: float = 0.8) -> torch.Tensor:
    """
    Apply gamma correction to exaggerate depth differences.
    Here 1=near, 0=far.
    gamma < 1 -> expand far (background)
    gamma > 1 -> expand near (foreground)
    """
    depth = torch.clamp(depth, 0.0, 1.0)
    return depth.pow(gamma)

def apply_sigmoid(depth: torch.Tensor, k: float = 10.0, midpoint: float = 0.5) -> torch.Tensor:
    """
    Apply sigmoid mapping to emphasize mid-range depth.
    Larger k makes it steeper.
    1=near, 0=far convention.
    """
    depth = torch.clamp(depth, 0.0, 1.0)
    return 1.0 / (1.0 + torch.exp(-k * (depth - midpoint)))

def apply_foreground_scale(depth: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Apply foreground/background scaling to a depth map.

    Args:
        depth (torch.Tensor): depth map of shape (H, W, 1), values normalized in [0, 1].
                              0 = near (foreground), 1 = far (background).
        scale (float): scaling factor
                       > 0: foreground closer, background further
                       < 0: foreground flatter, background closer
                       = 0: no change

    Returns:
        torch.Tensor: scaled depth map of same shape as input
    """
    if not torch.is_floating_point(depth):
        depth = depth.float()

    if scale > 0:
        # Exaggerate separation: foreground closer, background further
        return torch.pow(depth, 1.0 + scale)
    elif scale < 0:
        # Compress foreground, pull background closer
        return 1.0 - torch.pow(1.0 - depth, 1.0 + abs(scale))
    else:
        return depth
    
def anti_alias(depth: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    """
    Apply anti-aliasing to reduce jagged edges in depth maps.
    
    Args:
        depth (torch.Tensor): Normalized depth map tensor [H,W] or [B,1,H,W] with values in [0,1].
        strength (float): Blur strength; higher = smoother edges. Recommended range [0.5, 2.0].
    
    Returns:
        torch.Tensor: Smoothed depth map with same shape.
    """
    if depth.dim() == 2:
        depth = depth.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    elif depth.dim() == 3:
        depth = depth.unsqueeze(1)  # [B,1,H,W]

    # Kernel size scales with strength
    k = int(3 * strength) | 1  # force odd number
    if k < 3:
        return depth.squeeze()

    # Gaussian blur kernel
    sigma = 0.5 * strength
    coords = torch.arange(k, device=depth.device, dtype=depth.dtype) - k // 2
    gauss = torch.exp(-(coords**2) / (2 * sigma**2))
    gauss /= gauss.sum()

    # Separable convolution (X then Y)
    depth = F.conv2d(depth, gauss.view(1,1,1,-1), padding=(0, k//2), groups=1)
    depth = F.conv2d(depth, gauss.view(1,1,-1,1), padding=(k//2, 0), groups=1)

    return depth.squeeze()

def normalize_tensor(tensor):
    return (tensor - tensor.min())/(tensor.max() - tensor.min()+1e-6)
# Temporal depth stabilizer (EMA)
class DepthStabilizer:
    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.prev = None
        self.enabled = True
        self.lock = Lock()

    def __call__(self, depth: torch.Tensor):
        if not self.enabled:
            return depth
        with self.lock:
            if self.prev is None or self.prev.shape != depth.shape or self.prev.device != depth.device:
                self.prev = depth.detach()
                return depth
            out = self.alpha * self.prev + (1.0 - self.alpha) * depth
            self.prev = out.detach()
            return out

depth_stabilizer = DepthStabilizer(alpha=0.9)  # increase alpha for more stability

# Guided-like smoothing (edge-aware)
def box_blur(x, r: int):
    if r <= 0:
        return x
    k = 2 * r + 1
    w = torch.ones(1, 1, 1, k, device=x.device, dtype=x.dtype) / k
    h = torch.ones(1, 1, k, 1, device=x.device, dtype=x.dtype) / k
    x = F.conv2d(x, w, padding=(0, r), groups=1)
    x = F.conv2d(x, h, padding=(r, 0), groups=1)
    return x

def guided_smooth(depth: torch.Tensor, rgb: torch.Tensor, r=2, eps=1e-3):
    """
    depth: [H,W] / [1,1,H,W], rgb: [1,3,H,W] in [0,1].
    """
    if depth.dim() == 2:
        depth = depth.unsqueeze(0).unsqueeze(0)
    elif depth.dim() == 3:
        depth = depth.unsqueeze(1)
    if rgb.dim() == 3:
        rgb = rgb.unsqueeze(0)

    # luminance guidance
    I = 0.2989 * rgb[:,0:1] + 0.5870 * rgb[:,1:2] + 0.1140 * rgb[:,2:3]

    mean_I = box_blur(I, r)
    mean_D = box_blur(depth, r)
    corr_I = box_blur(I * I, r)
    corr_ID = box_blur(I * depth, r)

    var_I = corr_I - mean_I * mean_I
    cov_ID = corr_ID - mean_I * mean_D

    a = cov_ID / (var_I + eps)
    b = mean_D - a * mean_I

    mean_a = box_blur(a, r)
    mean_b = box_blur(b, r)

    q = mean_a * I + mean_b
    return q.squeeze()

# Piecewise remap (foreground boost)
def apply_piecewise(
    depth: torch.Tensor,
    split: float = 0.5,
    near_gamma: float = 2.0,
    far_gamma: float = 0.8
    ) -> torch.Tensor:
    """
    Efficient piecewise gamma remap for depth maps.
    Assumes 1=near, 0=far.
    near_gamma -> [split, 1]
    far_gamma  -> [0, split]
    """
    depth = depth.clamp(0.0, 1.0)

    # Near branch
    near_val = (((depth - split).clamp(min=0) / (1 - split + 1e-6))
                .pow(near_gamma) * (1 - split)) + split

    # Far branch
    far_val = (((depth).clamp(max=split) / (split + 1e-6))
               .pow(far_gamma) * split)

    # Select branch without indexing
    out = torch.where(depth >= split, near_val, far_val)
    return out


# Selective edge dilation to reduce holes near edges while avoiding halos
def edge_dilate(depth: torch.Tensor, dilation_size: int = 2, edge_thresh: float = 0.02):
    if dilation_size <= 0:
        return depth
    if depth.dim() == 2:
        d = depth.unsqueeze(0).unsqueeze(0)
    elif depth.dim() == 3:
        d = depth.unsqueeze(1)
    else:
        d = depth

    sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], device=d.device, dtype=d.dtype).view(1,1,3,3)/8
    sobel_y = sobel_x.transpose(2,3)
    gx = F.conv2d(d, sobel_x, padding=1)
    gy = F.conv2d(d, sobel_y, padding=1)
    mag = torch.sqrt(gx*gx + gy*gy)

    m = (mag > edge_thresh).float()
    kernel_size = dilation_size if dilation_size % 2 == 1 else dilation_size + 1
    dilated = F.max_pool2d(d, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    out = d * (1 - m) + dilated * m

    if out.shape[0] == 1 and out.shape[1] == 1:
        return out.squeeze(0).squeeze(0)
    elif out.shape[1] == 1:
        return out.squeeze(1)
    else:
        return out

def add_contrast(tensor, contrast_factor):
    tensor = tensor.unsqueeze(0)
    tensor = adjust_contrast(tensor, contrast_factor)
    return tensor.squeeze(0)

def predict_depth(image_rgb: np.ndarray, return_tuple=False,
                  use_temporal_smooth: bool = True):
    """
    Returns depth in [0,1], 1=near, 0=far. Optionally returns (depth, rgb_c).
    """

    depth = model.predict_depth_vda2(image_rgb, input_size=DEPTH_RESOLUTION, device=DEVICE, dtype=DTYPE) # reset state for new video

    depth = normalize_tensor(depth)
    if 'Metric' in MODEL_ID:
        depth = 1.0 - depth  # invert for metric models
    # depth = depth.clamp(0.2, 0.9)
    # depth = normalize_tensor(depth)
    
    # Robust normalize
    depth = apply_stretch(depth, 5, 95)
    # Post dept processing
    # depth = add_contrast(depth, 1.1)
    # depth = apply_gamma(depth, 1.2)
    depth = apply_sigmoid(depth, k=4, midpoint=0.618)
    depth = apply_piecewise(depth, split=0.618, near_gamma=1.2, far_gamma=0.6)
    depth = apply_foreground_scale(depth, scale=2)
    depth = normalize_tensor(depth)
    # Mild AA to reduce jaggies
    depth = anti_alias(depth, strength=AA_STRENTH)
    
    # Optional temporal stabilization (EMA)
    if use_temporal_smooth:
        depth = depth_stabilizer(depth)

    if return_tuple:
        rgb_tensor = torch.from_numpy(image_rgb).to(DEVICE, dtype=DTYPE)
        rgb_tensor = rgb_tensor.permute(2,0,1).contiguous()  # [C,H,W]
        return depth, rgb_tensor
    else:
        return depth

if __name__ == "__main__":
    depth = predict_depth(image_rgb)

    import matplotlib.pyplot as plt
    plt.imshow(depth.cpu().numpy(), cmap='inferno')
    plt.colorbar()
    plt.show()
    plt.close()
    plt.imshow(depth.cpu().numpy(), cmap='inferno')
    plt.colorbar()
    plt.savefig("test.png", dpi=300)