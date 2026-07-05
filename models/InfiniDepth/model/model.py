import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from safetensors.torch import load_file
from .block.config import dinov3_model_configs
from .block.implicit_decoder import ImplicitHead
from .block.convolution import BasicEncoder

def _autocast_device_type(device: torch.device):
    """Device-type string for torch.autocast, or None if unsupported (DirectML, CPU)."""
    t = device.type
    if t in ("cuda", "mps", "xpu"):
        return t
    return None


def _acc_dtype(device: torch.device) -> torch.dtype:
    """Best reduced-precision dtype for the given device, evaluated at runtime.

    Returns the dtype that matches what TensorRT would use for this device:
      - CUDA Ampere+ (cap >= 8): bfloat16  (safe exponent range, TF32 on TRT)
      - ROCm: float16  (more consistent InfiniDepth values than bf16 on RDNA)
      - CUDA older / ROCm older: float16
      - MPS (Apple Silicon): float16  (bf16 not yet supported by MPS backend)
      - XPU (Intel Arc/Xe): bfloat16
      - CPU / DirectML / other: float32  (no autocast — _autocast_device_type returns None)
    """
    t = device.type
    if t == "cuda":
        if getattr(torch.version, "hip", None) is not None:
            return torch.float16
        cap = torch.cuda.get_device_capability(device)[0]
        return torch.bfloat16 if cap >= 8 else torch.float16
    if t == "mps":
        return torch.float16
    if t == "xpu":
        return torch.bfloat16
    return torch.float32


def _resolve_local_dinov3_repo() -> str:
    """Always use the in-repo local DINOv3 torchhub path."""
    dinov3_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "block", "torchhub", "dinov3"))
    if not os.path.isdir(dinov3_repo):
        raise FileNotFoundError(
            "DINOv3 local torchhub repo not found at fixed path: "
            f"{dinov3_repo}"
        )
    return dinov3_repo


def _make_dense_query_coord(batch: int, h: int, w: int, device: torch.device) -> torch.Tensor:
    """Create dense 2D query coordinates in [-1, 1], order (y, x)."""
    ys = ((torch.arange(h, device=device, dtype=torch.float32) + 0.5) / max(float(h), 1.0)) * 2.0 - 1.0
    xs = ((torch.arange(w, device=device, dtype=torch.float32) + 0.5) / max(float(w), 1.0)) * 2.0 - 1.0
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    query = torch.stack([grid_y, grid_x], dim=-1).reshape(1, -1, 2)
    return query.expand(batch, -1, -1).contiguous()


class InfiniDepth(nn.Module):
    def __init__(
        self,
        model_path: Optional[str] = None,
        encoder: str = "vitl16",
    ):
        super().__init__()
        self.model_config = dinov3_model_configs[encoder]
        local_dinov3_repo = _resolve_local_dinov3_repo()
        self.pretrained = torch.hub.load(
            local_dinov3_repo,
            f"dinov3_{encoder}",
            source="local",
            pretrained=False,
        )
        self.patch_size = 16
        dim = self.pretrained.blocks[0].attn.qkv.in_features

        self.basic_encoder = BasicEncoder(
            input_dim=3,
            output_dim=128,
            stride=4,
        )
        self.depth_implicit_head = ImplicitHead(
            hidden_dim=dim,
            basic_dim=128,
            fusion_type="concat",
            out_dim=1,
            hidden_list=[1024, 256, 32],
        )
        self.register_buffer("_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        if model_path is not None:
            if os.path.exists(model_path):
                if model_path.endswith(".safetensors"):
                    checkpoint = load_file(model_path, device="cpu")
                else:
                    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
                if "state_dict" in checkpoint:
                    # Lightning checkpoint — strip wrapper prefix dynamically
                    state = checkpoint["state_dict"]
                    keys = list(state.keys())
                    prefix = keys[0]
                    for k in keys:
                        while prefix and not all(s.startswith(prefix) for s in keys):
                            prefix = prefix.rsplit(".", 1)[0]
                    clean = {k[len(prefix)+1:]: v for k, v in state.items()} if prefix else state
                else:
                    # Raw state_dict
                    clean = checkpoint
                self.load_state_dict(clean)
            else:
                raise FileNotFoundError(f"Model file {model_path} not found")

        self.eval()

    def _prepare_backbone_features(
        self,
        x: torch.Tensor,
        force_fp32: bool = False,
    ):
        h, w = x.shape[-2:]
        x_dino = (x - self._mean) / self._std

        act = _autocast_device_type(x.device)
        use_ac = act is not None and not torch.onnx.is_in_onnx_export() and not force_fp32
        dtype = _acc_dtype(x.device)

        # Only the last layer is consumed by ImplicitHead._encode_feat (features[-1][0]).
        # Extracting all 4 layer indices stores activations that are never read.
        last_layer_idx = self.model_config["layer_idxs"][-1]

        if use_ac:
            with torch.autocast(act, enabled=True, dtype=dtype):
                features = self.pretrained.get_intermediate_layers(
                    x_dino,
                    n=[last_layer_idx],
                    return_class_token=True,
                )
        else:
            features = self.pretrained.get_intermediate_layers(
                x_dino,
                n=[last_layer_idx],
                return_class_token=True,
            )

        patch_h, patch_w = h // self.patch_size, w // self.patch_size

        x_basic = 2.0 * x - 1.0
        basic_feat = self.basic_encoder(x_basic)  # fp32: InstanceNorm runs safely in fp32

        return features, basic_feat, patch_h, patch_w, h, w, use_ac, act, dtype

    def forward_dense(
        self,
        x: torch.Tensor,
        force_fp32: bool = False,
    ) -> torch.Tensor:
        (
            features,
            basic_feat,
            patch_h,
            patch_w,
            h,
            w,
            use_ac,
            act,
            dtype,
        ) = self._prepare_backbone_features(x, force_fp32=force_fp32)

        # ImplicitHead contains only Linear + ReLU/ELU + dense interpolation —
        # no normalization — so bf16 is safe and uses tensor cores.
        # Cast basic_feat to match dtype *before* the autocast block so that
        # feature fusion sees uniform input dtypes inside the autocast context.
        if use_ac:
            with torch.autocast(act, enabled=True, dtype=dtype):
                depth = self.depth_implicit_head.forward_dense(
                    features,
                    basic_feat.to(dtype=dtype),
                    patch_h,
                    patch_w,
                    h,
                    w,
                )
        else:
            depth = self.depth_implicit_head.forward_dense(
                features,
                basic_feat,
                patch_h,
                patch_w,
                h,
                w,
            )

        return depth

    def forward(
        self,
        x: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        force_fp32: bool = False,
    ) -> torch.Tensor:
        if coords is None:
            return self.forward_dense(x=x, force_fp32=force_fp32).flatten(2).permute(0, 2, 1)

        (
            features,
            basic_feat,
            patch_h,
            patch_w,
            _h,
            _w,
            use_ac,
            act,
            dtype,
        ) = self._prepare_backbone_features(x, force_fp32=force_fp32)

        # Keep the arbitrary-coordinate query path for callers that need it.
        if use_ac:
            with torch.autocast(act, enabled=True, dtype=dtype):
                depth = self.depth_implicit_head(
                    features,
                    basic_feat.to(dtype=dtype),
                    patch_h,
                    patch_w,
                    coords,
                )
        else:
            depth = self.depth_implicit_head(features, basic_feat, patch_h, patch_w, coords)

        return depth
