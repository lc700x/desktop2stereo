from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
from PIL import Image
import requests
import matplotlib.pyplot as plt
import cv2

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
        if torch.backends.mps.is_available() and index == 0:
            return torch.device("mps"), "Using Apple Silicon (MPS) device"
        if torch.cuda.is_available():
            return torch.device("cuda"), f"Using CUDA device: {torch.cuda.get_device_name(index)}"
        else:
            return torch.device("cpu"), "Using CPU device"
    except:
        return torch.device("cpu"), "Using CPU device"

DEVICE, DEVICE_INFO = get_device(DEVICE_ID)
print(DEVICE_INFO)

# Load and prepare image
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
image = Image.open("C:/Users/zjuli/Pictures/test2.png").convert("RGB")

# Load model
image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf", use_fast=True)
model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to(DEVICE, dtype=DTYPE)

# Generate depth map
inputs = image_processor(images=image, return_tensors="pt")
inputs = {k: v.to(DEVICE, dtype=DTYPE) for k, v in inputs.items()}
with torch.no_grad():
    outputs = model(**inputs)
predicted_depth = outputs.predicted_depth

# Interpolate to original size
depth = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),
    size=image.size[::-1],
    mode="bilinear",
    align_corners=False,
).squeeze()

# Convert depth to numpy (normalize to [0,1], assuming higher values are closer)
depth_np = depth.cpu().detach().numpy()
depth_normalized = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)  # [0,1], closer = higher

# Visualize depth map (display only, no save)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(depth_normalized, cmap='inferno')
plt.title("Predicted Depth Map")
plt.colorbar()
plt.axis('off')
plt.tight_layout()
plt.show()
plt.close()

# Depth to Stereo Conversion Function using OpenCV with Reduced Artifacts
def depth_to_stereo(color_img, depth_img, alpha=None, target_max_edge_disp=None, max_alpha_cap=None, gamma=0.7, inpaint_radius=3, hole_threshold=0.1, inpaint_method=cv2.INPAINT_NS):
    """
    Convert color image + depth map to side-by-side stereo pair using OpenCV, with nonlinear depth boost and conservative inpainting for stronger, cleaner 3D.
    
    Args:
    - color_img: np.array (H, W, 3) uint8 or float [0,255] RGB.
    - depth_img: np.array (H, W) float [0,1], higher = closer.
    - alpha: float, disparity scale (pixels for max depth). If None, auto-tuned.
    - target_max_edge_disp: float, target max disparity at edges (default: 3% of width).
    - max_alpha_cap: float, cap on alpha to prevent excessive shifts/distortion (default: 3% of width).
    - gamma: float <1 to boost foreground depth (default: 0.7 for stronger near-field pop).
    - inpaint_radius: int, radius for inpainting holes (default: 3, smaller for less artifacts).
    - hole_threshold: float, intensity threshold for detecting holes (default: 0.1, higher for fewer inpaints).
    - inpaint_method: cv2 inpaint flag (default: INPAINT_NS for higher quality, less artifacts).
    
    Returns:
    - stereo_img: np.array (H, 2W, 3) uint8 side-by-side (RGB).
    - left_img: np.array (H, W, 3) uint8 left eye view (RGB).
    - right_img: np.array (H, W, 3) uint8 right eye view (RGB).
    """
    # Ensure float types
    if color_img.dtype != np.float32:
        color_img = color_img.astype(np.float32)
    if depth_img.dtype != np.float32:
        depth_img = depth_img.astype(np.float32)
    
    h, w = depth_img.shape
    depth = np.clip(depth_img, 0, 1)  # Normalize/sanitize
    
    # Nonlinear boost for stronger foreground depth (like Owl3D's enhanced effect)
    depth = np.power(depth, gamma)
    
    # Smooth disparity map lightly to reduce artifacts from sharp edges
    depth = cv2.GaussianBlur(depth, (3, 3), 0)
    
    # Auto-tune alpha if not provided (using OpenCV Sobel)
    if alpha is None:
        if target_max_edge_disp is None:
            target_max_edge_disp = 0.03 * w  # 3% for balanced depth
        if max_alpha_cap is None:
            max_alpha_cap = 0.03 * w
        # Compute depth edges (horizontal + vertical gradients)
        grad_x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)
        depth_grad_x = np.abs(grad_x)
        depth_grad_y = np.abs(grad_y)
        max_depth_edge = np.max(depth_grad_x + depth_grad_y)
        if max_depth_edge > 0:
            alpha = target_max_edge_disp / max_depth_edge
        else:
            alpha = target_max_edge_disp
        # Cap alpha to avoid distortion
        alpha = min(alpha, max_alpha_cap)
    
    print(f"Using alpha = {alpha:.2f} (max edge disp ~{target_max_edge_disp:.1f} px, capped at {max_alpha_cap:.1f}, gamma={gamma} for boosted foreground)")
    
    disparity = alpha * depth
    half_disp = disparity / 2.0
    print(f"Max total disparity: {np.max(disparity):.1f} px (half: {np.max(half_disp):.1f} px)")
    
    # Create coordinate grids
    yy, xx = np.ogrid[0:h, 0:w]
    y_grid = np.broadcast_to(yy, (h, w)).astype(np.float32)
    x_grid = np.broadcast_to(xx, (h, w)).astype(np.float32)
    
    # Prepare color for remap: normalize to [0,1]
    color_float = color_img / 255.0
    
    # Left view: shift x right by half_disp (sample from x + half_disp)
    map_x_left = x_grid + half_disp
    map_y_left = y_grid
    # Clamp x to [0, w-1]
    map_x_left = np.clip(map_x_left, 0, w - 1)
    # Stack into single map: (H, W, 2) with [x, y]
    map_left = np.dstack([map_x_left, map_y_left]).astype(np.float32)
    
    left_float = cv2.remap(color_float, map_left, None, cv2.INTER_LINEAR, 
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    
    # Right view: shift x left by half_disp (sample from x - half_disp)
    map_x_right = x_grid - half_disp
    map_y_right = y_grid
    # Clamp x to [0, w-1]
    map_x_right = np.clip(map_x_right, 0, w - 1)
    # Stack into single map: (H, W, 2) with [x, y]
    map_right = np.dstack([map_x_right, map_y_right]).astype(np.float32)
    
    right_float = cv2.remap(color_float, map_right, None, cv2.INTER_LINEAR, 
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    
    # Create masks for holes (less sensitive threshold to reduce inpainting artifacts)
    left_mask = np.all(left_float < hole_threshold, axis=2).astype(np.uint8) * 255
    right_mask = np.all(right_float < hole_threshold, axis=2).astype(np.uint8) * 255
    
    # Convert to uint8 for inpainting (cv2.inpaint requires 8-bit 3-channel)
    left_temp = (np.clip(left_float, 0, 1) * 255).astype(np.uint8)
    right_temp = (np.clip(right_float, 0, 1) * 255).astype(np.uint8)
    
    # Inpaint holes (NS method for higher quality, fewer artifacts; smaller radius)
    left_inpaint = cv2.inpaint(left_temp, left_mask, inpaint_radius, inpaint_method)
    right_inpaint = cv2.inpaint(right_temp, right_mask, inpaint_radius, inpaint_method)
    
    # Assign to outputs
    left_img = left_inpaint
    right_img = right_inpaint
    
    # Side-by-side
    stereo = np.hstack((left_img, right_img))
    
    return stereo, left_img, right_img

# Generate stereo image with reduced artifacts
color_np = np.array(image)  # (H, W, 3) uint8 RGB
stereo_img, left_img, right_img = depth_to_stereo(color_np, depth_normalized, target_max_edge_disp=0.03 * color_np.shape[1], gamma=0.7, inpaint_radius=3, hole_threshold=0.1, inpaint_method=cv2.INPAINT_NS)

# Save only the side-by-side stereo image (convert RGB to BGR for cv2.imwrite)
cv2.imwrite("test_stereo.png", cv2.cvtColor(stereo_img, cv2.COLOR_RGB2BGR))

# Visualize stereo with matplotlib (display only, no save)
plt.figure(figsize=(16, 8))
plt.imshow(stereo_img)
plt.title("Side-by-Side Stereoscopic Image (Left | Right) - Reduced Artifacts")
plt.axis('off')
plt.show()
plt.close()

print("Side-by-side stereoscopic image saved as 'test_stereo.png' (with conservative inpainting and smoothing)")