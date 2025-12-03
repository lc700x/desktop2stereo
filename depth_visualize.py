# depth.py
import torch, cv2
torch.set_num_threads(1)
import torch.nn.functional as F
from transformers import AutoModelForDepthEstimation
import numpy as np
from threading import Lock
from PIL import Image
import os, warnings
img  = Image.open("C:/Users/zjuli/Pictures/test1.jpg").convert("RGB")
image_rgb = np.array(img)
AA_STRENGTH = 2
FP16 = True
DTYPE = torch.float16 if FP16 else torch.float32
CACHE_PATH = "models"
DEVICE_ID = 0
FILL_16_9 = True
# MODEL_ID = "depth-anything/Video-Depth-Anything-Small"
MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
# MODEL_ID = "depth-anything/DA3-SMALL"
# MODEL_ID = "depth-anything/DA3MONO-LARGE"
DEPTH_RESOLUTION = 518
FOREGROUND_SCALE = 0
USE_TORCH_COMPILE = False
USE_TENSORRT = False
RECOMPILE_TENSORRT=False

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
print(DEVICE_INFO)
print(f"Model: {MODEL_ID}")

IS_CUDA = "CUDA" in DEVICE_INFO
IS_NVIDIA = "CUDA" in DEVICE_INFO and "NVIDIA" in DEVICE_INFO
IS_AMD_ROCM = "CUDA" in DEVICE_INFO and "AMD" in DEVICE_INFO

# Optimization for CUDA
if IS_NVIDIA:
    torch.backends.cudnn.benchmark = True
    if not FP16:
        # Enable TF32 for matrix multiplications
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Enable TF32 matrix multiplication for better performance
        torch.set_float32_matmul_precision('high')
    # Enable math attention
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"] ="1" # Debug for torch.compile
    if USE_TORCH_COMPILE:
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module=r"torch\._inductor\.lowering"
        )
    
elif IS_AMD_ROCM:
    torch.backends.cudnn.enabled = False # Add for AMD ROCm
    os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1" # Add for AMD ROCm7
    # Enable math attention
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    USE_TORCH_COMPILE = False  # Disable torch.compile for AMD ROCm7 due to current issues

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
def apply_foreground_scale(depth: torch.Tensor, scale: float, mid: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    """
    Scale depth contrast so that:
      - depth in [0,1], where 0 = background (far), 1 = foreground (near)
      - scale > 0 : increase separation (foreground closer -> values move toward 1, background farther -> values move toward 0)
      - scale < 0 : reduce separation (flatten)
      - scale = 0 : identity

    Args:
        depth: torch.Tensor shape (..., 1) or (...), values in [0,1]
        scale: float, must be > -1.0 (we avoid scale == -1 which would divide by zero)
        mid: midpoint for separation (default 0.5)
        eps: small eps to avoid numerical issues

    Returns:
        Tensor same shape as depth, clamped to [0,1].
    """
    if not (-1.0 + 1e-12 < scale):  # avoid scale <= -1
        raise ValueError("scale must be greater than -1.0")

    d = depth.clamp(0.0, 1.0)
    if abs(scale) < eps:
        return d

    exponent = 1.0 / (1.0 + scale)  # >1 if scale<0 (flatten), <1 if scale>0 (exaggerate)
    dist = d - mid
    out = mid + torch.sign(dist) * torch.pow(torch.abs(dist), exponent)
    return out.clamp(0.0, 1.0)
       
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
    t = t_cpu.permute(2, 0, 1).contiguous().unsqueeze(0).to(device=DEVICE, dtype=MODEL_DTYPE)
    t = t / 255.0
    return t

def process(img_rgb: np.ndarray, height) -> np.ndarray:
    """
    Resize BGR/UMat numpy image to target height, keeping aspect ratio.
    """
    if isinstance(img_rgb, cv2.UMat):
        img_rgb = img_rgb.get()
    h0 = img_rgb.shape[0]
    if height < h0:
        width = int(img_rgb.shape[1] / h0 * height)
        img_rgb = cv2.resize(img_rgb, (width, height), interpolation=cv2.INTER_AREA)
    return img_rgb

def apply_gamma(depth, gamma=1.2):
    return torch.pow(depth, gamma)

def apply_contrast(depth, factor=1.2):
    mean = depth.mean(dim=(-2, -1), keepdim=True)  # per image mean
    return torch.clamp((depth - mean) * factor + mean, 0, 1)

def normalize_tensor(tensor):
    """ Normalize tensor to [0,1] """
    return (tensor - tensor.min())/(tensor.max() - tensor.min()+1e-6)

def post_process_depth(depth):
    depth = normalize_tensor(depth).squeeze()
    if 'metric' in MODEL_ID.lower() or 'da3' in MODEL_ID.lower():
        depth = 1.0 - depth
    depth = apply_gamma(depth)
    depth = apply_contrast(depth)
    depth = apply_foreground_scale(depth, scale=FOREGROUND_SCALE)
    depth = anti_alias(depth, strength=AA_STRENGTH)
    depth = normalize_tensor(depth).squeeze()
    return depth
        
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

    if 'depth-anything/video-depth-anything' in model_id.lower():
        checkpoint_name = f'video_depth_anything_{encoder}.pth'
    elif 'depth-anything/metric-video-depth-anything' in model_id.lower():
        checkpoint_name = f'metric_video_depth_anything_{encoder}.pth'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    checkpoint_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name, cache_dir=CACHE_PATH)

    model = VideoDepthAnything(**model_configs[encoder])
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True), strict=True)
    return model.to(DEVICE)

# Load Depth-Anything-V3 Model
def get_da3_model(model_id=MODEL_ID):
    from models.depth_anything_3.api_n import DepthAnything3
    model = DepthAnything3.from_pretrained(model_id, cache_dir=CACHE_PATH)
    return model.to(DEVICE)

# TensorRT Optimization
def optimize_with_tensorrt(onnx_path=ONNX_PATH, trt_path=TRT_PATH):
    """
    Convert ONNX model to TensorRT engine using TensorRT's Python API only.
    Supports FP32, FP16, and INT8 precisions based on global flags.
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
        
        # Set precision flags based on global configuration
        config.set_flag(trt.BuilderFlag.FP16)
        
        # Set workspace memory (essential for all precision modes) 
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4 GB Workspace 
        
        # Set dynamic shapes profile 
        profile = builder.create_optimization_profile()
        input_name = network.get_input(0).name
        # Updated shape ranges to better match typical input sizes
        min_shape = (1, 3, 224, 224)  # 224 = 14 * 16
        opt_shape = (1, 3, (DEPTH_RESOLUTION//14)*14, (DEPTH_RESOLUTION//14)*14)
        max_shape = (1, 3, 3920, 3920)  # 896 = 14 * 64
        
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
        
        # Optional: Enable additional optimizations that work well with FP32 [5](@ref)
        # These optimizations can improve performance regardless of precision
        config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)  # Enable sparse weights
        config.set_flag(trt.BuilderFlag.REJECT_EMPTY_ALGORITHMS)  # Reject empty algorithms
        
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
    dummy_input = torch.randn(1, 3, DEPTH_RESOLUTION, DEPTH_RESOLUTION, device=device, dtype=dtype)
    
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
        opset_version=16,
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
        self.is_cuda = IS_CUDA
        
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
                    self.model = self._load_pytorch_model(enable_trt=False)
            except Exception as e:
                print(f"[Error] TensorRT initialization failed: {str(e)}, falling back to PyTorch")
                self.backend = "PyTorch"
                self.model = self._load_pytorch_model(enable_trt=False)
        else:
            # Use PyTorch backend for DirectML/MPS/CPU
            
            # Ignore specific warning message
            warnings.filterwarnings("ignore", message="User provided device_type of 'cuda', but CUDA is not available")

            self.backend = "PyTorch"
            self.model = self._load_pytorch_model()
        
        print(f"Using backend: {self.backend}")
    
    def _load_pytorch_model(self, enable_trt=USE_TENSORRT):
        """Load the original PyTorch model."""
        # Load model
        if 'video-depth-anything' in MODEL_ID.lower():
            model = get_video_depth_anything_model(MODEL_ID)
        elif 'da3'  in MODEL_ID.lower():
            model = get_da3_model(MODEL_ID)
        else:
            # Load depth model
            model = AutoModelForDepthEstimation.from_pretrained(
                MODEL_ID,
                dtype=torch.float16 if FP16 else torch.float32,
                cache_dir=CACHE_PATH,
                weights_only=True
            ).to(DEVICE)
        
        if FP16 and 'da3' not in MODEL_ID.lower():
            model.half()
        
        if self.is_cuda and 'NVIDIA' in self.device_info and self.use_torch_compile and not enable_trt:
            model = torch.compile(model)
            print("Processing torch.compile with Triton, it may take a while...")
        
        return model.eval()
    
    def _load_tensorrt_engine(self):
        """Load or create TensorRT engine."""
        # First, load PyTorch model to export ONNX
        pytorch_model = self._load_pytorch_model()
        
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
        if self.is_cuda:
            with torch.inference_mode():
                with torch.amp.autocast('cuda'):
                    if self.backend == "PyTorch":
                        if "video-depth-anything" in MODEL_ID.lower():
                            return self.model(pixel_values=tensor)
                        elif "da3" in MODEL_ID.lower():
                            return self.model.predict_depth(tensor)
                        return self.model(pixel_values=tensor).predicted_depth
                    else:
                        return self.model(tensor)
        else:
            with torch.no_grad():
                if self.backend == "PyTorch":
                    if "video-depth-anything" in MODEL_ID.lower():
                        return self.model(pixel_values=tensor)
                    elif "da3" in MODEL_ID.lower():
                        return self.model.predict_depth(tensor)
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

if USE_TORCH_COMPILE and IS_CUDA:
    try:
        # Compile the model as before, but SKIP compiling lightweight post-processing functions，avoid FX re-tracing conflicts. These are fast without it.  
        post_process_depth = torch.compile(post_process_depth)
        # Assign to a global or module-level var if needed for access
        globals()['post_process_depth'] = post_process_depth  # Or use a class/module attribute
        
    except Exception as e:
        print(f"[Warning] torch.compile failed: {str(e)}, running without it.")

# Initialize with dummy input for warmup
def warmup_model(model_wraper, steps: int = 3):
    if IS_CUDA:
        with torch.inference_mode():
            with torch.amp.autocast('cuda' , dtype=DTYPE):
                for i in range(steps):
                    dummy = torch.randn(1, 3, DEPTH_RESOLUTION, DEPTH_RESOLUTION,
                                        device=DEVICE, dtype=MODEL_DTYPE)
                    model_wraper(dummy)
    else:
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
if USE_TORCH_COMPILE and IS_CUDA:
    depth_stabilizer.__call__ = torch.compile(depth_stabilizer.__call__, fullgraph=True)

# Modified predict_depth function with improved TRT integration
def predict_depth(image_rgb: np.ndarray, return_tuple=False, use_temporal_smooth: bool = True):
    """
    Returns depth in [0,1], 1=near, 0=far. Optionally returns (depth, rgb_c).
    """
    # Get input dimensions
    h, w = image_rgb.shape[:2]
    
    # Compute target size: shortest edge to DEPTH_RESOLUTION, preserve aspect ratio
    scale = DEPTH_RESOLUTION / min(h, w)
    target_h, target_w = int(h * scale), int(w * scale)
    # Ensure dimensions are divisible by 14 (ViT patch size)
    if "anything" in MODEL_ID.lower():
        target_h = (target_h // 14) * 14
        target_w = (target_w // 14) * 14
        if "video-depth-anything" in MODEL_ID.lower(): # fix for Video-Depth-Anything
            target_h, target_w = (DEPTH_RESOLUTION, DEPTH_RESOLUTION)
    else:
        target_h, target_w = (DEPTH_RESOLUTION, DEPTH_RESOLUTION)
    
    if return_tuple:
        # Convert to tensor and prepare rgb_c
        tensor = torch.from_numpy(image_rgb).to(device=DEVICE, dtype=MODEL_DTYPE, non_blocking=True)
        rgb_c = tensor.permute(2, 0, 1).contiguous()  # [C,H,W]
        tensor = rgb_c.unsqueeze(0) / 255.0
        # Resize using bilinear interpolation
        tensor = F.interpolate(tensor, size=(target_h, target_w), mode='bilinear', align_corners=False)
    else:
        # Resize on CPU with bilinear interpolation
        if (h, w) != (target_h, target_w):
            input_rgb = cv2.resize(image_rgb, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        else:
            input_rgb = image_rgb
        # Convert to tensor
        tensor = torch.from_numpy(input_rgb).permute(2, 0, 1).contiguous().unsqueeze(0).to(DEVICE, dtype=DTYPE) / 255.0
    
    tensor = ((tensor - MEAN) / STD).contiguous()
    tensor = tensor.to(dtype=MODEL_DTYPE)
        
    # Use model wrapper instead of direct model call
    if "video-depth-anything" in MODEL_ID.lower():
        with torch.no_grad():  # Add for safety in non-autocast paths
            depth = model_wraper(tensor)
    else:
        if IS_CUDA:
            with torch.no_grad():
                with torch.amp.autocast('cuda' , dtype=DTYPE):
                    depth = model_wraper(tensor)
        else:
            with torch.no_grad():
                depth = model_wraper(tensor)
    
    # Batched post-processing (compiled as a unit to avoid per-function tracing)
    with torch.no_grad():  # Prevent any gradient tracing
        depth = post_process_depth(depth)
    
    # Optional temporal stabilization (EMA)
    if use_temporal_smooth:
        depth = depth_stabilizer(depth)
        
    # Interpolate output to original size
    depth = F.interpolate(depth.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False)
    depth = depth.squeeze(0)
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
    Vectorized FPS overlay in PyTorch (DirectML / CUDA / MPS compatible).
    Ensures all operations use rgb.dtype (typically float32).
    """
    device, dtype = rgb.device, rgb.dtype
    chars, font_tensor = build_font(device=device, dtype=dtype)

    # Text to render
    txt = f"FPS: {fps:.1f}"
    idxs = torch.tensor([chars.index(ch) if ch in chars else chars.index(" ")
                         for ch in txt], device=device, dtype=torch.long)

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

    # Create the global mask (same dtype as rgb)
    mask = torch.zeros((H, W), device=device, dtype=dtype)

    for i, glyph in enumerate(glyphs):
        h_b, w_b = glyph.shape
        x0, y0 = int(offsets_x[i]), int(offsets_y[i])
        x1, y1 = min(W, x0 + w_b), min(H, y0 + h_b)
        # Clamp in case of overflows at edges
        if x0 < W and y0 < H:
            mask[y0:y1, x0:x1] = torch.maximum(mask[y0:y1, x0:x1],
                                               glyph[:y1 - y0, :x1 - x0])

    # Broadcast to channels and ensure correct dtype/device
    alpha = mask.unsqueeze(0).expand(rgb.shape[0], -1, -1)
    overlay_color = torch.as_tensor(color, device=device, dtype=dtype).view(-1, 1, 1).expand_as(rgb)

    one = torch.tensor(1.0, device=device, dtype=dtype)
    alpha = alpha.to(dtype)

    # Blend safely in float32
    return rgb * (one - alpha) + overlay_color * alpha


# generate left and right eye view for streamer 
def make_sbs_core(rgb: torch.Tensor,
                  depth: torch.Tensor,
                  ipd_uv=0.064,
                  depth_ratio=1.0,
                  display_mode="Half-SBS",
                  fill_16_9=FILL_16_9,
                  device=DEVICE) -> torch.Tensor:
    """
    Core tensor operations for side-by-side stereo.
    Keeps CUDA fast path (grid_sample) and fallback path (gather).
    Compatible with torch.compile.
    Inputs:
        rgb: [C,H,W] float tensor
        depth: [H,W] float tensor
    Returns:
        SBS image [C,H,W] float tensor (0-255 range)
    """
    # Cast to float32 for DirectML compatibility (avoids float64 ops)
    rgb = rgb.to(dtype=torch.float32, device=device)
    depth = depth.to(dtype=torch.float32, device=device)
    
    C, H, W = rgb.shape
    img = rgb.unsqueeze(0)  # [1,C,H,W]
    
    # Cast scalars to float32 tensors
    depth_ratio = torch.tensor(depth_ratio, dtype=torch.float32, device=device)
    ipd_uv = torch.tensor(ipd_uv, dtype=torch.float32, device=device)
    depth_strength = torch.tensor(0.05, dtype=torch.float32, device=device)
    
    inv = torch.ones_like(depth) - depth * depth_ratio
    max_px = ipd_uv * torch.tensor(W, dtype=torch.float32, device=device)
    shifts = inv * max_px * depth_strength
    
    # CUDA fast path: grid_sample
    if IS_CUDA:
        xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=torch.float32).view(1, 1, W).expand(1, H, W)
        ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=torch.float32).view(1, H, 1).expand(1, H, W)
        shift_norm = shifts * (2.0 / (W - 1))
        grid_left = torch.stack([xs + shift_norm, ys], dim=-1)
        grid_right = torch.stack([xs - shift_norm, ys], dim=-1)
        left = F.grid_sample(img, grid_left, mode="bilinear",
                             padding_mode="border", align_corners=False)[0]
        right = F.grid_sample(img, grid_right, mode="bilinear",
                              padding_mode="border", align_corners=False)[0]
    # Fallback path: vectorized gather (DirectML / MPS / CPU safe)
    else:
        base = torch.arange(W, device=device, dtype=torch.int64).view(1, -1).expand(H, -1)
        # Ensure shifts is float32 for addition
        shifts = shifts.to(dtype=torch.float32)
        coords_left = (base.to(dtype=torch.float32) + shifts).clamp(0, W - 1).long()  # [H,W]
        coords_right = (base.to(dtype=torch.float32) - shifts).clamp(0, W - 1).long()  # [H,W]
        # Left eye
        gather_idx_left = coords_left.unsqueeze(0).expand(C, H, W).unsqueeze(0)  # [1,C,H,W]
        left = torch.gather(img.expand(1, C, H, W), 3, gather_idx_left)[0]  # [C,H,W]
        # Right eye
        gather_idx_right = coords_right.unsqueeze(0).expand(C, H, W).unsqueeze(0)
        right = torch.gather(img.expand(1, C, H, W), 3, gather_idx_right)[0]
    
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
    if fill_16_9:
        left = pad_to_aspect_tensor(left)
        right = pad_to_aspect_tensor(right)
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
        
    # Handle input type conversion
    if isinstance(rgb_c, np.ndarray):
        # Convert numpy array to tensor with matching device/dtype
        rgb = torch.from_numpy(rgb_c).to(device=depth.device, dtype=depth.dtype)
        # Convert from HWC to CHW format
        if rgb.ndim == 3 and rgb.shape[2] == 3:
            rgb = rgb.permute(2, 0, 1)
    else:
        # Ensure tensor is on correct device and dtype
        rgb = rgb_c.to(device=depth.device, dtype=depth.dtype)

    # Optional FPS overlay can stay in Python side (avoids torch.compile recompiles)
    if fps is not None:
        rgb = overlay_fps(rgb, fps)  # your existing overlay function

    sbs_tensor = make_sbs_core(rgb, depth, ipd_uv, depth_ratio, display_mode)
    return sbs_tensor.to(torch.uint8).permute(1,2,0).contiguous().cpu().numpy()

if USE_TORCH_COMPILE and IS_CUDA:
    make_sbs_core = torch.compile(make_sbs_core)
 
if __name__ == "__main__":
    depth = predict_depth(image_rgb).squeeze(0)
    depth = depth.cpu().detach().numpy()
    import matplotlib.pyplot as plt
    plt.imshow(depth, cmap='inferno')
    plt.colorbar()
    plt.show()
    plt.close()
    plt.imshow(depth, cmap='inferno')
    plt.colorbar()
    plt.savefig("test.png", dpi=300)