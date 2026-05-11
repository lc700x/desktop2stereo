import os
import torch
from transformers import AutoModelForDepthEstimation
from huggingface_hub import hf_hub_download

# Set your model folder
MODEL_FOLDER = "./models"  # Change this to your desired folder
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Model loading functions extracted from depth.py

def load_standard_depth_model(model_id, cache_dir=None, dtype=torch.float32, device="cpu"):
    """
    Load standard depth estimation models from HuggingFace.
    
    Args:
        model_id: HuggingFace model ID (e.g., "depth-anything/Depth-Anything-V2-Small")
        cache_dir: Directory to cache the model
        dtype: torch.dtype (float32 or float16)
        device: Device to load model to
    
    Returns:
        Loaded model in eval mode
    """
    cache_dir = cache_dir or MODEL_FOLDER
    
    try:
        # Try loading with local_files_only=True first
        model = AutoModelForDepthEstimation.from_pretrained(
            model_id,
            dtype=dtype,
            cache_dir=cache_dir,
            weights_only=True,
            local_files_only=True
        )
    except Exception:
        # If not available locally, download
        print(f"Downloading model: {model_id}")
        model = AutoModelForDepthEstimation.from_pretrained(
            model_id,
            dtype=dtype,
            cache_dir=cache_dir,
            weights_only=True
        )
    
    model = model.to(device)
    
    if dtype == torch.float16:
        model.half()
    
    return model.eval()


def load_video_depth_anything_model(model_id, cache_dir=None):
    """
    Load Video Depth Anything models.
    
    Supported models:
        - depth-anything/Video-Depth-Anything-Small
        - depth-anything/Video-Depth-Anything-Base
        - depth-anything/Video-Depth-Anything-Large
        - depth-anything/Metric-Video-Depth-Anything-Small
        - depth-anything/Metric-Video-Depth-Anything-Base
        - depth-anything/Metric-Video-Depth-Anything-Large
    """
    from models.video_depth_anything.vda2_s import VideoDepthAnything
    
    cache_dir = cache_dir or MODEL_FOLDER
    
    encoder_dict = {
        'depth-anything/Video-Depth-Anything-Small': 'vits',
        'depth-anything/Video-Depth-Anything-Base': 'vitb',
        'depth-anything/Video-Depth-Anything-Large': 'vitl',
        'depth-anything/Metric-Video-Depth-Anything-Small': 'vits',
        'depth-anything/Metric-Video-Depth-Anything-Base': 'vitb',
        'depth-anything/Metric-Video-Depth-Anything-Large': 'vitl'
    }
    
    encoder = encoder_dict.get(model_id, 'vits')
    
    if 'depth-anything/video-depth-anything' in model_id.lower():
        checkpoint_name = f'video_depth_anything_{encoder}.pth'
    elif 'depth-anything/metric-video-depth-anything' in model_id.lower():
        checkpoint_name = f'metric_video_depth_anything_{encoder}.pth'
    else:
        raise ValueError(f"Unknown Video Depth Anything model: {model_id}")
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    
    # Download checkpoint
    try:
        checkpoint_path = hf_hub_download(
            repo_id=model_id, 
            filename=checkpoint_name, 
            cache_dir=cache_dir, 
            local_files_only=True
        )
    except:
        print(f"Downloading checkpoint: {checkpoint_name}")
        checkpoint_path = hf_hub_download(
            repo_id=model_id, 
            filename=checkpoint_name, 
            cache_dir=cache_dir
        )
    
    model = VideoDepthAnything(**model_configs[encoder])
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True), strict=True)
    
    return model


def load_da3_model(model_id, cache_dir=None, dtype=torch.float32):
    """
    Load Depth-Anything-V3 models.
    
    Args:
        model_id: DepthAnything3 model ID
        cache_dir: Cache directory
        dtype: torch.dtype
    """
    from models.depth_anything_3.api_n import DepthAnything3
    
    cache_dir = cache_dir or MODEL_FOLDER
    
    try:
        model = DepthAnything3.from_pretrained(
            model_id, 
            cache_dir=cache_dir, 
            dtype=dtype, 
            local_files_only=True
        )
    except:
        print(f"Downloading DepthAnything3 model: {model_id}")
        model = DepthAnything3.from_pretrained(
            model_id, 
            cache_dir=cache_dir, 
            dtype=dtype
        )
    
    return model


def download_model(model_id, cache_dir=None, device="cpu", dtype=torch.float32):
    """
    Universal model downloader that detects model type and downloads accordingly.
    
    Args:
        model_id: HuggingFace model ID
        cache_dir: Directory to save the model
        device: Device to load model to (cpu/cuda)
        dtype: torch.float32 or torch.float16
    
    Returns:
        Loaded model
    """
    cache_dir = cache_dir or MODEL_FOLDER
    
    print(f"\n{'='*60}")
    print(f"Downloading model: {model_id}")
    print(f"Destination: {cache_dir}")
    print(f"{'='*60}\n")
    
    # Detect model type and load appropriately
    if 'video-depth-anything' in model_id.lower():
        model = load_video_depth_anything_model(model_id, cache_dir)
        model = model.to(device)
    elif 'da3' in model_id.lower() or 'depth-anything-3' in model_id.lower():
        model = load_da3_model(model_id, cache_dir, dtype)
        model = model.to(device)
    else:
        model = load_standard_depth_model(model_id, cache_dir, dtype, device)
    
    print(f"\n✓ Successfully downloaded and loaded: {model_id}")
    return model


# Example usage - put your model IDs here
if __name__ == "__main__":
    # List of models to download (add your model IDs here)
    model_ids = [
        "depth-anything/Depth-Anything-V2-Small-hf",
        "depth-anything/Depth-Anything-V2-Base-hf",
        "depth-anything/Depth-Anything-V2-Large-hf",
        "depth-anything/Video-Depth-Anything-Small",
        "depth-anything/Video-Depth-Anything-Base",
        "depth-anything/Video-Depth-Anything-Large",
        "depth-anything/DA3-SMALL",
        "depth-anything/DA3-BASE",
        "depth-anything/DA3-LARGE-1.1",
        "depth-anything/DA3METRIC-LARGE",
        "depth-anything/DA3MONO-LARGE",
        "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
        "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
        "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
        "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
        "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
        "depth-anything/Metric-Video-Depth-Anything-Small",
        "depth-anything/Metric-Video-Depth-Anything-Base",
        "depth-anything/Metric-Video-Depth-Anything-Large",
        "LiheYoung/depth-anything-small-hf",
        "LiheYoung/depth-anything-base-hf",
        "LiheYoung/depth-anything-large-hf",
        "xingyang1/Distill-Any-Depth-Small-hf",
        "lc700x/Distill-Any-Depth-Base-hf",
        "xingyang1/Distill-Any-Depth-Large-hf",
    ]
    
    # Download each model
    for model_id in model_ids:
        try:
            model = download_model(model_id, cache_dir="./models")
            print(f"Model loaded successfully: {model_id}")
            
            # Print model info
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Total parameters: {total_params:,}")
            print("-" * 60)
            
        except Exception as e:
            print(f"Failed to download {model_id}: {e}")