# depth.py
import torch
torch.set_num_threads(1)
from utils import DEVICE_ID, MODEL_ID, CACHE_PATH, FP16, DEPTH_RESOLUTION, AA_STRENTH, FOREGROUND_SCALE, USE_TORCH_COMPILE, USE_TENSORRT, RECOMPILE_TRT
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

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    # if not FP16:
    # Enable TF32 for matrix multiplications
    torch.backends.cuda.matmul.allow_tf32 = True
    # Enable TF32 for cuDNN (convolution operations)
    torch.backends.cudnn.allow_tf32 = True
    # Enable TF32 matrix multiplication for better performance
    torch.set_float32_matmul_precision('high')

print(DEVICE_INFO)
print(f"Model: {MODEL_ID}")


import torch.nn.functional as F
from transformers import AutoModelForDepthEstimation
import numpy as np
from threading import Lock
import cv2
import os, warnings

if USE_TORCH_COMPILE and torch.cuda.is_available():
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"torch\._inductor\.lowering"
    )

# Model configuration
DTYPE = torch.float16 if FP16 else torch.float32
# Folder to store compiled model / cache 
MODEL_FOLDER = os.path.join(CACHE_PATH, "models--"+MODEL_ID.replace("/", "--"))
# Load depth model - either original or ONNX
DTYPE_INFO = "fp16" if FP16 else "fp32"
ONNX_PATH = os.path.join(MODEL_FOLDER, f"model_{DTYPE_INFO}_{DEPTH_RESOLUTION}.onnx")
TRT_PATH = os.path.join(MODEL_FOLDER, f"model_{DTYPE_INFO}_{DEPTH_RESOLUTION}.trt")


# Single character digits and letters for "FPS: XX.X"
font_dict = {
    "0": ["111","101","101","101","111"],
    "1": ["010","110","010","010","111"],
    "2": ["111","001","111","100","111"],
    "3": ["111","001","111","001","111"],
    "4": ["101","101","111","001","001"],
    "5": ["111","100","111","001","111"],
    "6": ["111","100","111","101","111"],
    "7": ["111","001","010","100","100"],
    "8": ["111","101","111","101","111"],
    "9": ["111","101","111","001","111"],
    "F": ["111","100","110","100","100"],
    "P": ["110","101","110","100","100"],
    "S": ["111","100","111","001","111"],
    ":": ["000","010","000","010","000"],
    ".": ["000","000","000","000","010"],  # for decimal point
    " ": ["000","000","000","000","000"],
}

# Post-processing functions
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

def process_tensor(img_rgb: np.ndarray, height) -> torch.Tensor:
    """
    Convert BGR/UMat numpy image to normalized GPU tensor in model dtype.
    Keeps transfers efficient and uses non_blocking for pinned tensors where possible.
    """
    if isinstance(img_rgb, cv2.UMat):
        img_rgb = img_rgb.get()

    h0, w0 = img_rgb.shape[:2]
    if height < h0:
        width = int(img_rgb.shape[1] / h0 * height)
        img_rgb = cv2.resize(img_rgb, (width, height), interpolation=cv2.INTER_AREA)

    # Ensure contiguous numpy array (uint8)
    np_img = np.ascontiguousarray(img_rgb)
    # convert to torch tensor on CPU (uint8) then float on device to avoid double copy
    t_cpu = torch.from_numpy(np_img)  # shape H,W,C dtype=uint8
    # move to device and convert in one step
    t = t_cpu.permute(2, 0, 1).contiguous().unsqueeze(0).to(device=DEVICE, dtype=MODEL_DTYPE, non_blocking=True)
    t = t / 255.0
    return t

def process(img_bgr: np.ndarray, height) -> np.ndarray:
    """
    Resize BGR/UMat numpy image to target height, keeping aspect ratio.
    """
    if isinstance(img_bgr, cv2.UMat):
        img_bgr = img_bgr.get()
    h0 = img_bgr.shape[0]
    if height < h0:
        width = int(img_bgr.shape[1] / h0 * height)
        img_bgr = cv2.resize(img_bgr, (width, height), interpolation=cv2.INTER_AREA)
    frame_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2RGB)
    return frame_rgb

def normalize_tensor(tensor):
    """ Normalize tensor to [0,1] """
    return (tensor - tensor.min())/(tensor.max() - tensor.min()+1e-6)

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

# Load Video Depth Anything Model
def get_video_depth_anything_model(model_id=MODEL_ID):
    """ Load Video Depth Anything model from HuggingFace hub. """
    from huggingface_hub import hf_hub_download
    from models.video_depth_anything.vda2_s import VideoDepthAnything
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
    return model.to(DEVICE).eval()

# TensorRT Optimization
def optimize_with_tensorrt(onnx_path=ONNX_PATH, trt_path=TRT_PATH):
    """
    Convert ONNX model to TensorRT engine using TensorRT's Python API only.
    Returns None if compilation fails.
    """
    try:
        if os.path.exists(trt_path) and RECOMPILE_TRT == False:
            print(f"Loaded existing TensorRT engine: {trt_path}")
            return trt_path
        
        import tensorrt as trt
        
        # Initialize logger and builder
        logger = trt.Logger(trt.Logger.ERROR)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        
        # Load ONNX model
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print("[Error]", parser.get_error(error))
                return None
        
        # Build configuration
        config = builder.create_builder_config()
        if FP16:
            config.set_flag(trt.BuilderFlag.FP16)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
        
        # Set dynamic shapes profile
        profile = builder.create_optimization_profile()
        input_name = network.get_input(0).name
        # input_shape = network.get_input(0).shape
        min_shape = (1, 3, DEPTH_RESOLUTION//2, DEPTH_RESOLUTION//2)
        opt_shape = (1, 3, DEPTH_RESOLUTION, DEPTH_RESOLUTION)
        max_shape = (1, 3, DEPTH_RESOLUTION*2, DEPTH_RESOLUTION*2)
        
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
        
        # Build engine
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            print("[Error] TensorRT engine build failed")
            return None
            
        with open(trt_path, "wb") as f:
            f.write(serialized_engine)
        
        print(f"[Main] TensorRT engine saved to {trt_path}")
        return trt_path
        
    except Exception as e:
        print(f"[Error] TensorRT optimization failed: {str(e)}")
        return None

# Export to ONNX
def export_to_onnx(model, output_path="depth_model.onnx", device=DEVICE, dtype=DTYPE):
    """
    Export the depth estimation model to ONNX format with dynamic axes.
    """
    dummy_input = torch.randn(1, 3, DEPTH_RESOLUTION, DEPTH_RESOLUTION, device=DEVICE, dtype=dtype)
    
    input_names = ["pixel_values"]
    output_names = ["predicted_depth"]
    dynamic_axes = {
        'pixel_values': {0: 'batch_size', 2: 'height', 3: 'width'},
        'predicted_depth': {0: 'batch_size', 1: 'height', 2: 'width'}
    }

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        export_params=True,
        verbose=False
    )
    
    print(f"ONNX model generated, TensorRT engine compling may take a while...")

# TensorRT Engine Wrapper Class (Without PyCUDA)
class TensorRTEngine:
    def __init__(self, engine_path, device, dtype):
        """
        Initialize TensorRT engine using binding names instead of deprecated methods.
        """
        self.device = device
        self.dtype = dtype
        
        try:
            import tensorrt as trt
            
            # Load TensorRT engine
            with open(engine_path, "rb") as f:
                engine_data = f.read()
            
            logger = trt.Logger(trt.Logger.ERROR)
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(engine_data)
            self.context = self.engine.create_execution_context()
            
            # Get binding information using names instead of deprecated methods
            self.input_binding_indices = []
            self.output_binding_indices = []
            
            for binding in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(binding)
                
                # Use name pattern matching to identify inputs/outputs
                if "input" in name.lower() or "pixel_values" in name.lower():
                    self.input_binding_indices.append(binding)
                else:
                    self.output_binding_indices.append(binding)
            
            # Pre-allocate output tensors
            self.output_shapes = {}
            for binding in self.output_binding_indices:
                name = self.engine.get_tensor_name(binding)
                self.output_shapes[name] = self.engine.get_tensor_shape(name)
            
        except ImportError:
            raise ImportError("TensorRT not available")

    def __call__(self, tensor):
        """Execute inference with TensorRT using native API."""
        # Set input binding dimensions
        input_shape = tuple(tensor.shape)
        name = self.engine.get_tensor_name(0)
        self.context.set_input_shape(name, input_shape)
        
        # Prepare output tensors
        outputs = {}
        bindings = [None] * self.engine.num_io_tensors

        # Set input binding
        bindings[0] = tensor.data_ptr()
        
        # Allocate output tensors
        for i, binding in enumerate(self.output_binding_indices, 1):
            name = self.engine.get_tensor_name(binding)
            dims = self.context.get_tensor_shape(name)
            shape_tuple = tuple(dims)  # Convert Dims to tuple
            output = torch.empty(shape_tuple, device=self.device, dtype=self.dtype)
            outputs[name] = output
            bindings[binding] = output.data_ptr()
        
        # Execute inference
        self.context.execute_v2(bindings=bindings)
        
        # Return the main output (predicted_depth)
        return outputs['predicted_depth']

# Model Wrapper Class
class DepthModelWrapper:
    def __init__(self, model_path, device, device_info, dtype, size=None,
                 onnx_path=ONNX_PATH, trt_path=TRT_PATH):
        """
        Wrapper class that handles both PyTorch and TensorRT backends.
        """
        self.device = device
        self.device_info = device_info
        self.dtype = dtype
        self.model_path = model_path
        self.onnx_path = onnx_path
        self.trt_path = trt_path
        self.size = size
        self.use_torch_compile = USE_TORCH_COMPILE
        
        # Determine backend based on device
        self.is_cuda = "CUDA" in device_info
        
        if self.is_cuda and USE_TENSORRT:
            # Use TensorRT backend for CUDA
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            try:
                # First try TensorRT
                self.backend = "TensorRT"
                self.model = self._load_tensorrt_engine()
                if self.model is None:
                    # Fall back to PyTorch if TensorRT fails
                    print("[Error] TensorRT failed, falling back to PyTorch")
                    self.backend = "PyTorch"
                    self.use_torch_compile = True  # Enable torch.compile for fallback
                    self.model = self._load_pytorch_model()
            except Exception as e:
                print(f"[Error] TensorRT initialization failed: {str(e)}, falling back to PyTorch")
                self.backend = "PyTorch"
                self.use_torch_compile = True  # Enable torch.compile for fallback
                self.model = self._load_pytorch_model()
        else:
            # Use PyTorch backend for DirectML/MPS/CPU
            self.backend = "PyTorch"
            self.model = self._load_pytorch_model()
        
        print(f"Using backend: {self.backend}")
    
    def _load_pytorch_model(self):
        """Load the original PyTorch model."""
        # Load model
        if 'Video-Depth-Anything' in MODEL_ID:
            model = get_video_depth_anything_model(MODEL_ID)

        else:
            # Load depth model
            model = AutoModelForDepthEstimation.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16 if FP16 else torch.float32,
                cache_dir=CACHE_PATH,
                weights_only=True
            ).to(DEVICE).eval()
        
        if FP16:
            model.half()
        
        if  "CUDA" in self.device_info and self.use_torch_compile and not USE_TENSORRT:
            model = torch.compile(model)
            print("Processing torch.compile with Triton, it may take a while...")
        
        return model
    
    def _load_tensorrt_engine(self):
        """Load or create TensorRT engine."""
        # First, load PyTorch model to export ONNX
        pytorch_model = self._load_pytorch_model()
        
        if FP16:
            pytorch_model.half()
        
        # Export to ONNX if not exists
        if RECOMPILE_TRT or not os.path.exists(self.onnx_path):
            export_to_onnx(pytorch_model, self.onnx_path, self.device, self.dtype)
        
        # Build or load TensorRT engine
        trt_engine_path = optimize_with_tensorrt(self.onnx_path, self.trt_path)
        if trt_engine_path is None:
            return None
        try:
            return TensorRTEngine(trt_engine_path, self.device, self.dtype)
        except Exception as e:
            print(f"[Error] TensorRT engine loading failed: {str(e)}")
            return None
    
    def __call__(self, tensor):
        """Run inference using the active backend."""
        with torch.no_grad():
            if self.backend == "PyTorch":
                if "Video-Depth-Anything" in MODEL_ID:
                    return self.model(pixel_values=tensor)
                return self.model(pixel_values=tensor).predicted_depth
            else:
                return self.model(tensor)

# Initialize model wrapper
model_wraper = DepthModelWrapper(
    model_path=MODEL_ID,
    device=DEVICE,
    device_info=DEVICE_INFO,
    dtype=DTYPE
)

MODEL_DTYPE = next(model_wraper.model.parameters()).dtype if hasattr(model_wraper.model, 'parameters') else DTYPE
MEAN = torch.tensor([0.485,0.456,0.406], device=DEVICE).view(1,3,1,1)
STD = torch.tensor([0.229,0.224,0.225], device=DEVICE).view(1,3,1,1)

if USE_TORCH_COMPILE and "CUDA" in DEVICE_INFO:
    try:
        anti_alias = torch.compile(anti_alias, fullgraph=True)
        apply_piecewise = torch.compile(apply_piecewise, fullgraph=True)
        apply_sigmoid = torch.compile(apply_sigmoid, fullgraph=True)
        apply_foreground_scale = torch.compile(apply_foreground_scale, fullgraph=True)
    except Exception as e:
        print(f"[Warning] torch.compile failed: {str(e)}, running without it.")

# Initialize with dummy input for warmup
def warmup_model(model_wraper, steps: int = 3):
    with torch.no_grad():
        for i in range(steps):
            dummy = torch.randn(1, 3, DEPTH_RESOLUTION, DEPTH_RESOLUTION,
                                device=DEVICE, dtype=MODEL_DTYPE)
            model_wraper(dummy)
    # print(f"Warmup complete with {steps} iterations.")

warmup_model(model_wraper, steps=3)

lock = Lock()

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
                self.prev = depth.detach().clone()
                return depth
            out = self.alpha * self.prev + (1.0 - self.alpha) * depth
            self.prev = out.detach().clone()
            return out

depth_stabilizer = DepthStabilizer(alpha=0.9)  # increase alpha for more stability
if USE_TORCH_COMPILE and "CUDA" in DEVICE_INFO:
    depth_stabilizer.__call__ = torch.compile(depth_stabilizer.__call__, fullgraph=True)

# Modified predict_depth function with improved TRT integration
def predict_depth(image_rgb: np.ndarray, return_tuple=False, use_temporal_smooth: bool = True):
    """
    Returns depth in [0,1], 1=near, 0=far. Optionally returns (depth, rgb_c).
    """
    h, w = image_rgb.shape[:2]
    if return_tuple:
        tensor = torch.from_numpy(image_rgb).to(DEVICE, dtype=DTYPE)
        rgb_c = tensor.permute(2,0,1).contiguous()  # [C,H,W]
        tensor = rgb_c.unsqueeze(0) / 255.0
        tensor = F.interpolate(tensor, (DEPTH_RESOLUTION, DEPTH_RESOLUTION), mode='bilinear', align_corners=True)
    else:
        # Resize input on CPU to model resolution for efficiency
        target_size = (DEPTH_RESOLUTION, DEPTH_RESOLUTION)
        if (h, w) != target_size:
            interpolation = cv2.INTER_AREA if max(h, w) > DEPTH_RESOLUTION else cv2.INTER_LINEAR
            input_rgb = cv2.resize(image_rgb, target_size, interpolation=interpolation)
        
        tensor = torch.from_numpy(input_rgb).permute(2,0,1).contiguous().unsqueeze(0).to(DEVICE, dtype=DTYPE) / 255.0
    
    tensor = ((tensor - MEAN) / STD).contiguous()
    tensor = tensor.to(dtype=MODEL_DTYPE)
        
    # Use model wrapper instead of direct model call
    if 'Video-Depth-Anything' in MODEL_ID:
        depth = model_wraper(tensor)
    else:
        with torch.no_grad():
            depth = model_wraper(tensor)
    
    # Interpolate output to original size
    depth = F.interpolate(depth.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=True)
    
    # Robust normalize and Post depth processing
    depth = apply_stretch(depth, 5, 95)
    
    # invert for metric models
    if 'Metric' in MODEL_ID:
        depth = 1.0 - depth
        
    # post processing
    depth = apply_sigmoid(depth, k=4, midpoint=0.618)
    depth = apply_piecewise(depth, split=0.618, near_gamma=1.2, far_gamma=0.6)
    depth = apply_foreground_scale(depth, scale=FOREGROUND_SCALE)
    depth = normalize_tensor(depth)
    # Mild AA to reduce jaggies
    depth = anti_alias(depth, strength=AA_STRENTH)

    # Optional temporal stabilization (EMA)
    if use_temporal_smooth:
        depth = depth_stabilizer(depth)
    
    if return_tuple:
        return depth, rgb_c
    else:
        return depth   
    
def build_font(device="cpu", dtype=torch.float32):
    chars = sorted(font_dict.keys())
    font_tensor = torch.stack([
        torch.tensor([[1.0 if c=="1" else 0.0 for c in row] for row in font_dict[ch]],
                     dtype=dtype, device=device)
        for ch in chars
    ])  # [num_chars, 5, 3]
    return chars, font_tensor  # mapping and bitmap tensor

def overlay_fps(rgb: torch.Tensor, fps: float, color=(0.0, 255.0, 0.0)) -> torch.Tensor:
    """
    Vectorized FPS overlay in PyTorch (compilable).
    rgb: [C,H,W] image tensor
    fps: float value
    color: overlay color (tuple for RGB or scalar for grayscale)
    """
    device, dtype = rgb.device, rgb.dtype
    chars, font_tensor = build_font(device=device, dtype=dtype)

    txt = f"FPS: {fps:.1f}"
    idxs = torch.tensor([chars.index(ch) if ch in chars else chars.index(" ") for ch in txt],
                        device=device)

    H, W = rgb.shape[1:]
    scale = max(1, min(8, H // 60))
    char_h, char_w = 5 * scale, 3 * scale
    spacing = scale
    margin_x, margin_y = 2 * scale, 2 * scale

    # Expand bitmaps for each character
    glyphs = font_tensor[idxs]  # [len_txt, 5, 3]
    glyphs = glyphs.repeat_interleave(scale, dim=1).repeat_interleave(scale, dim=2)  # [len_txt, h, w]

    # Compute x/y offsets for each character
    offsets_x = margin_x + torch.arange(len(txt), device=device) * (char_w + spacing)
    offsets_y = torch.full((len(txt),), margin_y, device=device)

    # Create global mask
    mask = torch.zeros((H, W), device=device, dtype=dtype)
    for i, glyph in enumerate(glyphs):
        h_b, w_b = glyph.shape
        x0, y0 = int(offsets_x[i]), int(offsets_y[i])
        x1, y1 = min(W, x0 + w_b), min(H, y0 + h_b)
        mask[y0:y1, x0:x1] = torch.maximum(mask[y0:y1, x0:x1], glyph[:y1-y0, :x1-x0])

    # Broadcast to channels
    alpha = mask.unsqueeze(0).expand(rgb.shape[0], -1, -1)
    overlay_color = torch.tensor(color, device=device, dtype=dtype).view(-1,1,1).expand_as(rgb)

    return rgb * (1.0 - alpha) + overlay_color * alpha

# generate left and right eye view for streamer 
def disparity_from_depth_tensor(depth_torch, baseline_px=80.0, zero_parallax=0.6, depth_ratio=1.0):
    """
    Convert depth map (1=near, 0=far) to disparity map.
    
    Args:
        depth_torch: Depth tensor [H,W] or [1,H,W] with values in [0,1] (1=near, 0=far)
        baseline_px: Baseline in pixels (controls stereo separation strength)
        zero_parallax: Depth value that will have zero disparity (screen plane)
        
    Returns:
        disp: Disparity map [H,W] (positive values shift right, negative shift left)
        depth_norm: Normalized depth [0=near,1=far] for visualization
    """
    if depth_torch.dim() == 2:
        depth_torch = depth_torch
    else:
        depth_torch = depth_torch.squeeze()
    depth_norm = 1.0 - depth_torch * depth_ratio  # depth.py uses 1=near, 0=far -> convert to 0=near,1=far
    disp = baseline_px * (zero_parallax - depth_norm)
    return disp

def warp_with_grid_sample_torch(rgb_t, disp_t, direction='left', align_corners=True):
    """
    Warp using grid_sample or gather-based fallback for DirectML.
    rgb_t: 1x3xHxW float tensor on device
    disp_t: HxW tensor (pixels) on same device
    """
    _, C, H, W = rgb_t.shape
    
    # For DirectML devices, use gather-based method
    if "DirectML" in DEVICE_INFO:
        # Create base coordinate grid
        base = torch.arange(W, device=disp_t.device).view(1, W).expand(H, W).float()
        
        # Calculate half disparity
        half_disp = disp_t / 2.0
        
        # Determine shift direction
        if direction == 'left':
            coords = (base - half_disp).clamp(0, W - 1).long()
        else:
            coords = (base + half_disp).clamp(0, W - 1).long()
        
        # Create gather indices
        gather_idx = coords.unsqueeze(0).expand(C, H, W).unsqueeze(0)  # [1,C,H,W]
        
        # Perform gather operation
        warped = torch.gather(rgb_t, 3, gather_idx)
        return warped
    
    # For CUDA devices, use optimized grid_sample
    elif "CUDA" in DEVICE_INFO:
        # Prepare coordinate grid
        xs = torch.linspace(0, W-1, W, device=disp_t.device, dtype=disp_t.dtype).view(1, W).expand(H, W)
        ys = torch.linspace(0, H-1, H, device=disp_t.device, dtype=disp_t.dtype).view(H, 1).expand(H, W)
        
        # Apply disparity shift
        half_disp = disp_t / 2.0
        if direction == 'left':
            src_x = xs - half_disp
        else:
            src_x = xs + half_disp
        src_y = ys
        
        # Normalize to [-1,1]
        nx = (src_x / (W - 1)) * 2.0 - 1.0
        ny = (src_y / (H - 1)) * 2.0 - 1.0
        grid = torch.stack((nx, ny), dim=2).unsqueeze(0)  # 1xHxWx2
        
        # Run grid_sample
        return F.grid_sample(rgb_t, grid, mode='bilinear', padding_mode='zeros', align_corners=align_corners)
    
    # Fallback for other devices (MPS/CPU)
    else:
        # Handle dtype conversion for non-CUDA devices
        if rgb_t.device.type != 'cuda' and rgb_t.dtype == torch.float16:
            rgb_fp32 = rgb_t.to(dtype=torch.float32)
            grid_fp32 = grid.to(dtype=torch.float32)
            sampled = F.grid_sample(rgb_fp32, grid_fp32, mode='bilinear',
                                    padding_mode='zeros', align_corners=align_corners)
            return sampled.to(dtype=rgb_t.dtype)
        else:
            return F.grid_sample(rgb_t, grid, mode='bilinear',
                                 padding_mode='zeros', align_corners=align_corners)


# Updated make_sbs_core function using the new disparity functions
def make_sbs_core(rgb: torch.Tensor,
                 depth: torch.Tensor,
                 ipd_uv=0.064,
                 depth_ratio=1.0,
                 display_mode="Half-SBS") -> torch.Tensor:
    """
    Updated core tensor operations for side-by-side stereo using disparity_from_depth_tensor.
    """
    C, H, W = rgb.shape
    img = rgb.unsqueeze(0)  # [1,C,H,W]
    
    # Convert depth to disparity
    baseline_px = ipd_uv * W * 0.2  # Convert IPD from UV to pixels
    disp = disparity_from_depth_tensor(depth, baseline_px=baseline_px, zero_parallax=0.5, depth_ratio=depth_ratio)
    # Warp left and right views
    left = warp_with_grid_sample_torch(img, disp, direction='left')
    right = warp_with_grid_sample_torch(img, disp, direction='right')
    
    # Aspect pad helper
    def pad_to_aspect_tensor(tensor, target_ratio=(16, 9)):
        _, h, w = tensor.shape
        t_w, t_h = target_ratio
        r_img, r_t = w / h, t_w / t_h
        if abs(r_img - r_t) < 1e-3:
            return tensor
        if r_img > r_t:  # too wide -> pad height
            new_h = int(round(w / r_t))
            pad_top = (new_h - h) // 2
            return F.pad(tensor, (0, 0, pad_top, new_h - h - pad_top))
        else:  # too tall -> pad width
            new_w = int(round(h * r_t))
            pad_left = (new_w - w) // 2
            return F.pad(tensor, (pad_left, new_w - w - pad_left, 0, 0))

    # Aspect pad & arrange SBS/TAB
    left = pad_to_aspect_tensor(left[0])  # Remove batch dim
    right = pad_to_aspect_tensor(right[0])

    if display_mode == "TAB":
        out = torch.cat([left, right], dim=1)
    else:
        out = torch.cat([left, right], dim=2)

    if display_mode != "Full-SBS":
        out = F.interpolate(out.unsqueeze(0), size=left.shape[1:], mode="area")[0]

    return out.clamp(0, 255)

def make_sbs(rgb_c, depth, ipd_uv=0.064, depth_ratio=1.0, display_mode="Half-SBS", fps=None):
    """
    Full function: adds optional FPS overlay and converts output to numpy uint8.
    Calls `make_sbs_core` for tensor computations (torch.compile compatible).
    """
    if depth.dim() == 3 and depth.shape[0] == 1:
        depth = depth[0]
    rgb = rgb_c.to(device=DEVICE, dtype=MODEL_DTYPE)

    # Optional FPS overlay can stay in Python side (avoids torch.compile recompiles)
    if fps is not None:
        rgb = overlay_fps(rgb, fps)  # your existing overlay function

    sbs_tensor = make_sbs_core(rgb, depth, ipd_uv, depth_ratio, display_mode)
    return sbs_tensor.to(torch.uint8).permute(1,2,0).contiguous().cpu().numpy()

if USE_TORCH_COMPILE and "CUDA" in DEVICE_INFO:
    # Compile the new functions for CUDA
    disparity_from_depth_tensor = torch.compile(disparity_from_depth_tensor)
    warp_with_grid_sample_torch = torch.compile(warp_with_grid_sample_torch)
    make_sbs_core = torch.compile(make_sbs_core)