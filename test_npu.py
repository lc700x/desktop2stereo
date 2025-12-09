import os
import sys
import shutil
import subprocess
import numpy as np
import torch
import onnx
import onnxruntime as ort
import cv2
import matplotlib.pyplot as plt

from pathlib import Path
from transformers import AutoModelForDepthEstimation

# ------------------------------------------------------------------------------
# Step 0: Enumerate the device to check which AMD hardware is present
# ------------------------------------------------------------------------------
def get_npu_info():
    """
    Enumerates PCI devices via 'pnputil' (Windows) to detect an AMD NPX, 
    e.g. PHX/HPT or STX.

    Returns:
        A string indicator: 'PHX/HPT', 'STX', 'KRK', or '' if not found.
    """
    command = r'pnputil /enum-devices /bus PCI /deviceids'
    process = subprocess.Popen(command, shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    # Check for known hardware IDs
    output_str = stdout.decode()
    npu_type = ''
    if 'PCI\\VEN_1022&DEV_1502&REV_00' in output_str:
        npu_type = 'PHX/HPT'
    if 'PCI\\VEN_1022&DEV_17F0&REV_00' in output_str:
        npu_type = 'STX'
    if 'PCI\\VEN_1022&DEV_17F0&REV_10' in output_str:
        npu_type = 'STX'
    if 'PCI\\VEN_1022&DEV_17F0&REV_11' in output_str:
        npu_type = 'STX'
    if 'PCI\\VEN_1022&DEV_17F0&REV_20' in output_str:
        npu_type = 'KRK'
    return npu_type

npu_type = get_npu_info()
print(f"Detected NPU type: {npu_type if npu_type else 'Not found'}")

# ------------------------------------------------------------------------------
# Step 1: Environment and provider setup (AMD official style)
# ------------------------------------------------------------------------------
# RYZEN_AI_INSTALLATION_PATH must be set
try:
    install_dir = os.environ['RYZEN_AI_INSTALLATION_PATH']
except KeyError:
    print("RYZEN_AI_INSTALLATION_PATH is not set. Please set your environment.")
    sys.exit(1)

# ONNX model path
ONNX_PATH = "my_depth_model.onnx"

# Example paths from official quicktest (not strictly used here)
model_path_in_example = os.path.join(install_dir, 'quicktest', 'test_model.onnx')
config_file = os.path.join(install_dir, 'voe-4.0-win_amd64', 'vaip_config.json')

# Cache directory for VAIP
cache_dir = Path(__file__).parent.resolve()
cache_key = 'my_depth_cache_key'

# Clean any stale cache
cache_path = cache_dir / cache_key
if cache_path.exists() and cache_path.is_dir():
    shutil.rmtree(cache_path)
    print(f"{cache_path} has been removed for a fresh run.")

# Provider configuration
providers = ['VitisAIExecutionProvider']
if npu_type == 'PHX/HPT':
    print("[Info] Setting environment for PHX/HPT xclbin usage.")
    xclbin_file = os.path.join(
        install_dir,
        'voe-4.0-win_amd64',
        'xclbins',
        'phoenix',
        '4x4.xclbin'  # Example xclbin from official sample
    )
    provider_options = [{
        'cache_dir': str(cache_dir),
        'cache_key': cache_key,
        'target': 'X1',
        'xclbin': xclbin_file
    }]
else:
    provider_options = [{
        'cache_dir': str(cache_dir),
        'cache_key': cache_key
    }]

# ------------------------------------------------------------------------------
# Step 2: Load Hugging Face Depth model and export to ONNX (fixed shape)
# ------------------------------------------------------------------------------
MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
CACHE_PATH = "models"
DEVICE = torch.device("cpu")  # Export from CPU to keep dependencies simple

print(f"Loading Hugging Face model: {MODEL_ID}")
model = AutoModelForDepthEstimation.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_PATH
).to(DEVICE)
model.eval()

# Fixed input resolution
HEIGHT, WIDTH = 512, 512
DUMMY_INPUT = torch.randn(1, 3, HEIGHT, WIDTH, device=DEVICE, dtype=torch.float32)

print(f"Exporting to ONNX: {ONNX_PATH}")
# NOTE: No dynamic_axes/dynamic_shapes, fixed shape only
torch.onnx.export(
    model,
    DUMMY_INPUT,
    ONNX_PATH,
    input_names=["pixel_values"],
    output_names=["predicted_depth"],
    opset_version=18,       # Let exporter use opset 18 directly
    do_constant_folding=True
)

# Validate ONNX
onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)
print("ONNX model export verified.")

# ------------------------------------------------------------------------------
# Step 3: Create ONNX Runtime Session with Vitis AI EP
# ------------------------------------------------------------------------------
try:
    print("Creating InferenceSession using VitisAIExecutionProvider...")
    session = ort.InferenceSession(
        ONNX_PATH,
        providers=providers,
        provider_options=provider_options
    )
except Exception as e:
    print(f"Failed to create an InferenceSession: {e}")
    sys.exit(1)
else:
    print("Session created successfully.")

# Get input name
input_name = session.get_inputs()[0].name
print(f"Model input name: {input_name}")

# ------------------------------------------------------------------------------
# Step 4: Run inference on a test image (resized to 512×512)
# ------------------------------------------------------------------------------
TEST_IMAGE_PATH = "C:/Users/zjuli/Pictures/test1.jpg"
print(f"Reading test image: {TEST_IMAGE_PATH}")
bgr_img = cv2.imread(TEST_IMAGE_PATH)
if bgr_img is None:
    raise FileNotFoundError(f"Could not read: {TEST_IMAGE_PATH}")

# Convert BGR to RGB
rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

# Resize to model's expected size (512x512)
resized_rgb = cv2.resize(rgb_img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

# Normalize to [0,1] and convert to NCHW
image_float = resized_rgb.astype(np.float32) / 255.0
image_float = np.transpose(image_float, (2, 0, 1))  # HWC -> CHW
image_float = np.expand_dims(image_float, axis=0)   # (1, C, H, W)

print("Running inference...")
try:
    outputs = session.run([], {input_name: image_float})
except Exception as e:
    print(f"Failed to run the InferenceSession: {e}")
    sys.exit(1)
else:
    print("Inference finished.")

# Predicted depth is outputs[0]
predicted_depth = outputs[0]          # shape (1, H_out, W_out) e.g. (1, 504, 504)
depth_map = predicted_depth[0]        # (H_out, W_out)

# Normalize for visualization
depth_min, depth_max = depth_map.min(), depth_map.max()
if depth_max > depth_min:
    depth_img = (depth_map - depth_min) / (depth_max - depth_min)
else:
    depth_img = depth_map

# Resize depth map to match the 512x512 image for nicer side-by-side display
depth_img_resized = cv2.resize(depth_img, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)

# Visualize
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(resized_rgb)
plt.title("Input Image (512x512)")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(depth_img_resized, cmap="inferno")
plt.title("Predicted Depth")
plt.axis("off")

# plt.show()

print("\nDone! Depth map displayed successfully.")

# ------------------------------------------------------------------------------
# Step 5: Optional random-data test with correct spatial size (512x512)
# ------------------------------------------------------------------------------
def preprocess_random_image():
    # Must match model input shape: (1,3,512,512)
    image_array = np.random.rand(1, 3, HEIGHT, WIDTH).astype(np.float32)
    return image_array

try:
    random_data = preprocess_random_image()
    _ = session.run(None, {input_name: random_data})
except Exception as e:
    print(f"Failed to run InferenceSession on random data: {e}")
    sys.exit(1)
else:
    print("Random data test passed!")
