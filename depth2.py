import torch
torch.set_num_threads(1)  # Set to avoid high CPU usage caused by default full threads
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForDepthEstimation
import numpy as np
from threading import Lock
import cv2
from utils import DEVICE_ID, MODEL_ID, CACHE_PATH, FP16, DEPTH_RESOLUTION
import math
import types
import copy
import sys

# Model configuration
DTYPE = torch.float16 if FP16 else torch.float32  # Use float32 for DirectML compatibility

# Initialize DirectML Device
def get_device(index=0):
    """Returns a torch.device and a human-readable device info string."""
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

# Get the device and print information
DEVICE, DEVICE_INFO = get_device(DEVICE_ID)

# Enable cudnn benchmark if CUDA is available
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# Output Info
print(f"{DEVICE_INFO}")
print(f"Model: {MODEL_ID}")

# Load model with same configuration as example
model = AutoModelForDepthEstimation.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if FP16 else torch.float32,
    cache_dir=CACHE_PATH,
    weights_only=True
).to(DEVICE).eval()

if FP16:
    model.half()

MODEL_DTYPE = next(model.parameters()).dtype
# Normalization parameters (same as example)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

# Warm-up with dummy input
with torch.no_grad():
    dummy = torch.zeros(1, 3, DEPTH_RESOLUTION, DEPTH_RESOLUTION, device=DEVICE, dtype=MODEL_DTYPE)
    model(pixel_values=dummy)    

lock = Lock()

# ======================================================================
# RowFlowV3 Stereo Model Implementation
# ======================================================================
ROW_FLOW_V3_URL = "iw3_row_flow_v3.pth"
_tile_size_validators = {}
_models = {}


try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from contextlib import nullcontext

    def use_flash_attention(flag):
        if flag:
            return nullcontext()
        else:
            return sdpa_kernel([SDPBackend.MATH])

except ModuleNotFoundError:
    def use_flash_attention(flag):
        return torch.backends.cuda.sdp_kernel(enable_flash=flag, enable_math=True, enable_mem_efficient=flag)

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from contextlib import nullcontext

    def use_flash_attention(flag):
        if flag:
            return nullcontext()
        else:
            return sdpa_kernel([SDPBackend.MATH])

except ModuleNotFoundError:
    def use_flash_attention(flag):
        return torch.backends.cuda.sdp_kernel(enable_flash=flag, enable_math=True, enable_mem_efficient=flag)


def replication_pad2d_naive(x, padding, detach=False):
    assert x.ndim == 4 and len(padding) == 4
    left, right, top, bottom = padding

    detach_fn = lambda t: t.detach() if detach else t
    if left > 0 and right > 0:
        pad_l = (detach_fn(x[:, :, :, :1]),) * left
        pad_r = (detach_fn(x[:, :, :, -1:]),) * right
        x = torch.cat((*pad_l, x, *pad_r), dim=3)
    else:
        if left > 0:
            x = torch.cat((*((detach_fn(x[:, :, :, :1]),) * left), x), dim=3)
        elif left < 0:
            x = x[:, :, :, -left:]
        if right > 0:
            x = torch.cat((x, *((detach_fn(x[:, :, :, -1:]),) * right)), dim=3)
        elif right < 0:
            x = x[:, :, :, :right]
    if top > 0 and bottom > 0:
        pad_t = (detach_fn(x[:, :, :1, :]),) * top
        pad_b = (detach_fn(x[:, :, -1:, :]),) * bottom
        x = torch.cat((*pad_t, x, *pad_b), dim=2)
    else:
        if top > 0:
            x = torch.cat((*((detach_fn(x[:, :, :1, :]),) * top), x), dim=2)
        elif top < 0:
            x = x[:, :, -top:, :]
        if bottom > 0:
            x = torch.cat((x, *((detach_fn(x[:, :, -1:, :]),) * bottom)), dim=2)
        elif bottom < 0:
            x = x[:, :, :bottom, :]

    return x.contiguous()

def _basic_module_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    else:
        pass

def pad_shift_mask_token(x, mask_token, window_size, shift=(True, True)):
    mask_token = mask_token.to(x.dtype)

    if shift[1]:
        B, C, H, W = x.shape
        pad_w = mask_token.expand(B, C, H, window_size[1] // 2)
        x = torch.cat((pad_w, x, pad_w), dim=3)
    if shift[0]:
        B, C, H, W = x.shape
        pad_h = mask_token.expand(B, C, window_size[0] // 2, W)
        x = torch.cat((pad_h, x, pad_h), dim=2)
    return x

def basic_module_init(model):
    if isinstance(model, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.BatchNorm2d)):
        _basic_module_init(model)
    else:
        for m in model.modules():
            _basic_module_init(m)

def pixel_unshuffle(x, window_size):
    """ reference implementation of F.pixel_unshuffle + non-square window
    """
    B, C, H, W = x.shape
    SH, SW = window_size if isinstance(window_size, (list, tuple)) else [window_size, window_size]
    SS = SH * SW
    assert H % SH == 0 and W % SW == 0

    oc = C * SS
    oh = H // SH
    ow = W // SW
    x = x.reshape(B, C, oh, SH, ow, SW)
    # B, C, SH, SW, oh, ow
    x = x.permute(0, 1, 3, 5, 2, 4)
    # B, (C, SH, SW), oh, ow
    x = x.reshape(B, oc, oh, ow)

    return x

def pixel_shuffle(x, window_size):
    """ reference implementation of F.pixel_shuffle + non-square window
    """
    B, C, H, W = x.shape
    SH, SW = window_size if isinstance(window_size, (list, tuple)) else [window_size, window_size]
    SS = SH * SW
    assert C % SS == 0 and C % SS == 0

    oc = C // SS
    oh = H * SH
    ow = W * SW
    x = x.reshape(B, oc, SH, SW, H, W)
    # B, oc, H, SH, W, SW
    x = x.permute(0, 1, 4, 2, 5, 3)
    # B, oc, (H, SH), (W, SW)
    x = x.reshape(B, oc, oh, ow)

    return x

def _set(name, factory):
    global _models
    if name in _models:
        _models[name] = factory
    if isinstance(factory, types.FunctionType):
        ident = factory.__name__
    else:
        ident = repr(factory)

def register_model(cls):
    assert issubclass(cls, Model)
    _set(cls.name, cls)
    if hasattr(cls, "name_alias"):
        for alias in cls.name_alias:
            _set(alias, cls)
    return cls

def _find_valid_tile_size(name, base_tile_size):
    validator = _tile_size_validators.get(name, None)
    if validator is not None:
        tile_size = int(base_tile_size)
        while tile_size > 0:
            if validator(tile_size):
                return tile_size
            tile_size -= 1
        raise ValueError(f"Could not find valid tile size: tile_size={base_tile_size}")
    else:
        return int(base_tile_size)

def _register_tile_size_validator(name, func):
    _tile_size_validators[name] = func

def bchw_to_bnc(x, window_size):
    # For sequential model, e.g. transformer, lstm
    # b = B * (h // window_size) * (w // window_size)
    # n = window_size * window_size
    # c = c
    # aka. window_partition
    B, C, H, W = x.shape
    SH, SW = window_size if isinstance(window_size, (list, tuple)) else [window_size, window_size]
    assert H % SH == 0 and W % SW == 0

    oh = H // SH
    ow = W // SW
    x = x.reshape(B, C, oh, SH, ow, SW)
    # B, oh, ow, SH, SW, C
    x = x.permute(0, 2, 4, 3, 5, 1)
    # (B, SH, SW), (oh, ow), C
    x = x.reshape(B * oh * ow, SH * SW, C)

    return x

def bnc_to_bchw(x, out_shape, window_size):
    # reverse bchw_to_bnc
    B, N, C = x.shape
    OB, OC, OH, OW = out_shape
    SH, SW = window_size if isinstance(window_size, (list, tuple)) else [window_size, window_size]
    assert OH % SH == 0 and OW % SW == 0
    H = OH // SH
    W = OW // SW

    x = x.reshape(OB, H, W, SH, SW, C)
    # OB, C, H, SH, W, SW
    x = x.permute(0, 5, 1, 3, 2, 4)
    # OB, (H * SH), (W * SW), C
    x = x.reshape(OB, C, OH, OW)

    return x

def sliced_sdp(q, k, v, num_heads, attn_mask=None, dropout_p=0.0, is_causal=False):
    B, QN, C = q.shape  # batch, sequence, feature
    KN = k.shape[1]
    assert C % num_heads == 0
    qkv_dim = C // num_heads
    # B, H, N, C // H
    q = q.view(B, QN, num_heads, qkv_dim).permute(0, 2, 1, 3)
    k = k.view(B, KN, num_heads, qkv_dim).permute(0, 2, 1, 3)
    v = v.view(B, KN, num_heads, qkv_dim).permute(0, 2, 1, 3)
    use_flash = B <= 65535  # avoid CUDA error: invalid configuration argument.
    with use_flash_attention(use_flash):
        x = F.scaled_dot_product_attention(q, k, v,
                                            attn_mask=attn_mask, dropout_p=dropout_p,
                                            is_causal=is_causal)
    # B, N, (H, C // H)
    return x.permute(0, 2, 1, 3).reshape(B, QN, qkv_dim * num_heads)

class Model(nn.Module):
    name = "nunif.Model"

    def __init__(self, kwargs):
        super(Model, self).__init__()
        self.kwargs = {}
        self.updated_at = None
        self.register_kwargs(kwargs)

    def get_device(self):
        return next(self.parameters()).device

    def register_kwargs(self, kwargs):
        for name, value in kwargs.items():
            if name not in {"self", "__class__"}:
                self.kwargs[name] = value

    def get_kwargs(self):
        return self.kwargs

    def __repr__(self):
        return (f"name: {self.name}\nkwargs: {self.kwargs}\n" +
                super(Model, self).__repr__())

    def to_inference_model(self):
        net = copy.deepcopy(self)
        net.eval()
        return net

    def to_script_module(self):
        net = self.to_inference_model()
        return torch.jit.script(net)

    def export_onnx(self, f, **kwargs):
        raise NotImplementedError()
class ReplicationPad2d(nn.Module):
    def __init__(self, padding):
        super().__init__()
        if isinstance(padding, int):
            self.padding = (padding, padding, padding, padding)
        elif len(padding) == 2:
            # (left/right, top/bottom) to (left, right, top, bottom)
            self.padding = (padding[0], padding[0], padding[1], padding[1])
        else:
            self.padding = tuple(padding)
    
    def forward(self, x):
        return F.pad(x, self.padding, mode='replicate')
class MHA(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_dim=None):
        super().__init__()
        # require torch >= 2.0 (recommend torch >= 2.1.2)
        # nn.MultiheadAttention also has a bug with float attn_mask, so PyTorch 2.1 is required anyway.
        assert hasattr(F, "scaled_dot_product_attention"), "torch version does not support F.scaled_dot_product_attention"

        if qkv_dim is None:
            assert embed_dim % num_heads == 0
            qkv_dim = embed_dim // num_heads
        self.qkv_dim = qkv_dim
        self.num_heads = num_heads
        self.qkv_proj = nn.Linear(embed_dim, qkv_dim * num_heads * 3)
        self.head_proj = nn.Linear(qkv_dim * num_heads, embed_dim)
        basic_module_init(self)

    def forward(self, x, attn_mask=None, dropout_p=0.0, is_causal=False):
        # x.shape: batch, sequence, feature
        q, k, v = self.qkv_proj(x).split(self.qkv_dim * self.num_heads, dim=-1)
        x = sliced_sdp(q, k, v, self.num_heads, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
        x = self.head_proj(x)
        return x

@torch.no_grad()
def _gen_window_score_bias_input(window_size1, window_size2, reduction):
    N1 = window_size1[0] * window_size1[1]
    N2 = window_size2[0] * window_size2[1]

    positions1 = torch.stack(
        torch.meshgrid(torch.arange(0, window_size1[0]),
                       torch.arange(0, window_size1[1]), indexing="ij"), dim=2).reshape(N1, 2)

    positions2 = torch.stack(
        torch.meshgrid(torch.arange(0, window_size2[0]),
                       torch.arange(0, window_size2[1]), indexing="ij"), dim=2).reshape(N2, 2)
    positions2.mul_(reduction)

    delta = torch.zeros((N1, N2, 2), dtype=torch.long)
    for i in range(N1):
        for j in range(N2):
            delta[i][j] = positions1[i] - positions2[j]

    delta = delta.view(N1 * N2, 2)
    delta = [tuple(p) for p in delta.tolist()]
    unique_delta = sorted(list(set(delta)))
    index = [unique_delta.index(d) for d in delta]
    index = torch.tensor(index, dtype=torch.int64)
    unique_delta = torch.tensor(unique_delta, dtype=torch.float32)
    unique_delta = unique_delta / unique_delta.abs().max()
    return index, unique_delta

class WindowScoreBias(nn.Module):
    def __init__(self, window_size, hidden_dim=None, reduction=1, num_heads=None):
        super().__init__()
        if isinstance(window_size, int):
            window_size1 = [window_size, window_size]
        else:
            window_size1 = window_size

        assert window_size1[0] % reduction == 0 and window_size1[1] % reduction == 0

        window_size2 = [window_size1[0] // reduction, window_size1[1] // reduction]

        self.window_size1 = window_size1
        self.window_size2 = window_size2
        self.num_heads = num_heads

        index, unique_delta = _gen_window_score_bias_input(self.window_size1, self.window_size2, reduction)
        self.register_buffer("index", index)
        self.register_buffer("delta", unique_delta)
        if hidden_dim is None:
            hidden_dim = int((self.window_size1[0] * self.window_size1[1]) ** 0.5) * 2
        if self.num_heads is None:
            output_dim = 1
        else:
            output_dim = num_heads

        self.to_bias = nn.Sequential(
            nn.Linear(2, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim, bias=True))

        basic_module_init(self)

    def forward(self):
        N1 = self.window_size1[0] * self.window_size1[1]
        N2 = self.window_size2[0] * self.window_size2[1]
        bias = self.to_bias(self.delta)
        bias = bias[self.index]
        if self.num_heads is None:
            # (N,N) float attention score bias
            bias = bias.reshape(N1, N2)
        else:
            # (H,N,N) float attention score bias
            bias = bias.permute(1, 0).contiguous().reshape(self.num_heads, N1, N2)
        return bias

class WindowMHA2d(nn.Module):
    """ WindowMHA
    BCHW input/output
    """
    def __init__(self, in_channels, num_heads, window_size=(4, 4), qkv_dim=None, shift=False, shift_mask_token=False):
        super().__init__()
        self.window_size = (window_size if isinstance(window_size, (tuple, list))
                            else (window_size, window_size))
        self.shift = (shift if isinstance(shift, (tuple, list))
                      else (shift, shift))
        self.pad_h = self.pad_w = 0
        if self.shift[0] or self.shift[1]:
            if self.shift[0]:
                assert self.window_size[0] % 2 == 0
                self.pad_h = self.window_size[0] // 2
            if self.shift[1]:
                assert self.window_size[1] % 2 == 0
                self.pad_w = self.window_size[1] // 2
            if shift_mask_token:
                self.shift_mask_bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
                nn.init.trunc_normal_(self.shift_mask_bias, 0, 0.01)

        if not hasattr(self, "shift_mask_bias"):
            self.shift_mask_bias = None

        self.num_heads = num_heads
        self.mha = MHA(in_channels, num_heads, qkv_dim)
        basic_module_init(self)

    def forward(self, x, attn_mask=None, layer_norm=None):
        if self.shift[0] or self.shift[1]:
            if self.shift_mask_bias is not None:
                x = pad_shift_mask_token(x, self.shift_mask_bias, self.window_size, self.shift)
            else:
                x = F.pad(x, (self.pad_w, self.pad_w, self.pad_h, self.pad_h), mode="constant", value=0)
        out_shape = x.shape
        x = bchw_to_bnc(x, self.window_size)
        if layer_norm is not None:
            x = layer_norm(x)
        x = self.mha(x, attn_mask=attn_mask)
        x = bnc_to_bchw(x, out_shape, self.window_size)
        if self.shift[0] or self.shift[1]:
            x = F.pad(x, (-self.pad_w, -self.pad_w, -self.pad_h, -self.pad_h))
        return x
    
class I2IBaseModel(Model):
    name = "nunif.i2i_base_model"

    def __init__(self, kwargs, scale, offset, in_channels=None, in_size=None, blend_size=None,
                 default_tile_size=256, default_batch_size=4):
        super(I2IBaseModel, self).__init__(kwargs)
        self.i2i_scale = scale
        self.i2i_offset = offset
        self.i2i_in_channels = in_channels
        self.i2i_in_size = in_size
        self.i2i_blend_size = blend_size
        self.i2i_default_tile_size = default_tile_size
        self.i2i_default_batch_size = default_batch_size

    def register_tile_size_validator(self, validator):
        _register_tile_size_validator(self.name, validator)

    def find_valid_tile_size(self, base_tile_size):
        if base_tile_size is None:
            base_tile_size = self.i2i_default_tile_size
        tile_size = _find_valid_tile_size(self.name, base_tile_size)
        return tile_size

    def export_onnx(self, f, **kwargs):
        shape = [1, self.i2i_in_channels, self.i2i_default_tile_size, self.i2i_default_tile_size]
        x = torch.rand(shape, dtype=torch.float32)
        model = self.to_inference_model()
        torch.onnx.export(
            model,
            x,
            f,
            input_names=["x"],
            output_names=["y"],
            dynamic_axes={'x': {0: 'batch_size', 2: "input_height", 3: "input_width"},
                          'y': {0: 'batch_size', 2: "height", 3: "width"}},
            **kwargs
        )
        
OFFSET = 32

class WABlock(nn.Module):
    def __init__(self, in_channels, window_size, layer_norm=False):
        super(WABlock, self).__init__()
        self.mha = WindowMHA2d(in_channels, num_heads=2, window_size=window_size)
        self.conv_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0),
            nn.GELU(),
            ReplicationPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), padding=0),
            nn.LeakyReLU(0.1, inplace=True))
        self.bias = WindowScoreBias(window_size)

    def forward(self, x):
        x = x + self.mha(x, attn_mask=self.bias())
        x = x + self.conv_mlp(x)
        return x

@register_model
class RowFlowV3(I2IBaseModel):
    name = "sbs.row_flow_v3"

    def __init__(self):
        super(RowFlowV3, self).__init__(locals(), scale=1, offset=OFFSET, in_channels=8, blend_size=4)
        self.downscaling_factor = (1, 8)
        self.mod = 4 * 3
        pack = self.downscaling_factor[0] * self.downscaling_factor[1]
        C = 64
        assert C >= pack
        self.blocks = nn.Sequential(
            nn.Conv2d(3 * pack, C, kernel_size=1, stride=1, padding=0),
            WABlock(C, (4, 4)),
            WABlock(C, (3, 3)),
        )
        self.last_layer = nn.Sequential(
            ReplicationPad2d((1, 1, 1, 1)),
            nn.Conv2d(C // pack, 1, kernel_size=3, stride=1, padding=0)
        )
        self.register_buffer("delta_scale", torch.tensor(1.0 / 127.0))
        self.delta_output = False
        self.symmetric = False

    def _forward(self, x):
        input_height, input_width = x.shape[2:]
        pad1 = (self.mod * self.downscaling_factor[1]) - input_width % (self.mod * self.downscaling_factor[1])
        pad2 = (self.mod * self.downscaling_factor[0]) - input_height % (self.mod * self.downscaling_factor[0])
        x = replication_pad2d_naive(x, (0, pad1, 0, pad2), detach=True)
        x = pixel_unshuffle(x, self.downscaling_factor)
        x = self.blocks(x)
        x = pixel_shuffle(x, self.downscaling_factor)
        x = F.pad(x, (0, -pad1, 0, -pad2), mode="constant")
        x = self.last_layer(x)
        return x

    def _warp(self, rgb, grid, delta, delta_scale):
        output_dtype = rgb.dtype
        # Convert to float32 for precision during warping
        rgb = rgb.to(torch.float32)
        grid = grid.to(torch.float32)
        delta = delta.to(torch.float32)
        delta_scale = delta_scale.to(torch.float32)

        # Create full delta tensor (x and y components)
        delta = torch.cat([delta, torch.zeros_like(delta)], dim=1)
        
        # Apply delta to grid coordinates
        grid = grid + delta * delta_scale
        
        # Permute for grid_sample: (B, H, W, 2) format
        grid = grid.permute(0, 2, 3, 1)
        
        # Perform grid sampling with border padding
        z = F.grid_sample(
            rgb, 
            grid, 
            mode='bilinear', 
            padding_mode='border',
            align_corners=True
        )
        
        # Return to original dtype
        return z.to(output_dtype)

    def _forward_default(self, x):
        rgb = x[:, 0:3, :, :]
        grid = x[:, 6:8, :, :]
        x = x[:, 3:6, :, :]  # depth + divergence feature + convergence

        delta = self._forward(x)
        if self.symmetric:
            left = self._warp(rgb, grid, delta, self.delta_scale)
            right = self._warp(rgb, grid, -delta, self.delta_scale)
            left = F.pad(left, (-OFFSET,) * 4)
            right = F.pad(right, (-OFFSET,) * 4)
            z = torch.cat([left, right], dim=1)
        else:
            z = self._warp(rgb, grid, delta, self.delta_scale)
            z = F.pad(z, (-OFFSET,) * 4)

        if self.training:
            return z, ((grid[:, 0:1, :, :] / self.delta_scale).detach() + delta)
        else:
            return torch.clamp(z, 0., 1.)

    def _forward_delta_only(self, x):
        assert not self.training
        delta = self._forward(x)
        delta = delta.to(torch.float32)
        delta = torch.cat([delta, torch.zeros_like(delta)], dim=1)
        return delta

    def forward(self, x):
        if not self.delta_output:
            return self._forward_default(x)
        else:
            return self._forward_delta_only(x)

# ======================================================================
# Mapper Functions for Stereo Generation
# ======================================================================
def softplus01_legacy(depth, c=6):
    min_v = math.log(1 + math.exp(0 * 12.0 - c)) / (12 - c)
    max_v = math.log(1 + math.exp(1 * 12.0 - c)) / (12 - c)
    v = torch.log(1. + torch.exp(depth * 12.0 - c)) / (12 - c)
    return (v - min_v) / (max_v - min_v)

def resolve_mapper_function(name):
    if name == "pow2":
        return lambda x: x ** 2
    elif name == "none":
        return lambda x: x
    elif name == "softplus":
        return softplus01_legacy
    elif name == "softplus2":
        return lambda x: softplus01_legacy(x) ** 2
    else:
        raise NotImplementedError(f"mapper={name}")

def get_mapper(name):
    if ":" in name:
        names = name.split(":")
    else:
        names = [name]
    functions = []
    for name in names:
        if "+" in name:
            # weighted average (interpolation)
            name, weight = name.split("=")
            if not weight:
                weight = 0.5
            else:
                weight = float(weight)
                assert 0.0 <= weight <= 1.0
            mapper_a, mapper_b = name.split("+")
            mapper_a = resolve_mapper_function(mapper_a)
            mapper_b = resolve_mapper_function(mapper_b)
            functions.append(lambda x: mapper_a(x) * (1 - weight) + mapper_b(x) * weight)
        else:
            functions.append(resolve_mapper_function(name))

    return lambda x: x if not functions else functions[0](x)

# ======================================================================
# Stereo Generation Functions
# ======================================================================
def load_model(model_path, model=None, device_ids=None,
               strict=True, map_location="cpu", weights_only=False):
    if "mps" in str(map_location):
        map_location = "cpu"
    data = torch.load(model_path, map_location=map_location, weights_only=weights_only)

    assert ("nunif_model" in data)
    model_predefine = True
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(data["state_dict"], strict=strict)
    else:
        model.load_state_dict(data["state_dict"], strict=strict)
    if "updated_at" in data:
        model.updated_at = data["updated_at"]
    data.pop("state_dict")

    if not model_predefine and device_ids is not None:
        device = DEVICE
        model = model.to(device)

    return model, data

def load_row_flow_model():
    model = RowFlowV3()
    checkpoint = torch.load(ROW_FLOW_V3_URL, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(DEVICE).eval()
    model.symmetric = False
    model.delta_output = True
    return model

stereo_model = None

def get_stereo_model():
    global stereo_model
    if stereo_model is None:
        stereo_model = load_row_flow_model()
    return stereo_model

def make_grid(batch, width, height, device):
    mesh_y, mesh_x = torch.meshgrid(
        torch.linspace(-1, 1, height, device=device),
        torch.linspace(-1, 1, width, device=device),
        indexing="ij"
    )
    grid = torch.stack((mesh_x, mesh_y), dim=0)
    return grid.unsqueeze(0).expand(batch, 2, height, width)

def backward_warp(c, grid, delta, delta_scale):
    grid = grid + delta * delta_scale
    grid = grid.permute(0, 2, 3, 1)
    z = F.grid_sample(c, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return torch.clamp(z, 0, 1)

def make_divergence_feature_value(divergence, convergence, image_width):
    divergence_pix = divergence * 0.5 * 0.01 * image_width
    divergence_feature_value = divergence_pix / 32.0
    convergence_feature_value = (-divergence_pix * convergence) / 32.0
    return divergence_feature_value, convergence_feature_value

def apply_divergence_nn_delta(model, c, depth, divergence, convergence, steps,
                              mapper, shift, enable_amp):
    steps = 1 if steps is None else steps
    if shift > 0:
        c = torch.flip(c, (3,))
        depth = torch.flip(depth, (3,))
        
    # Ensure depth is 4D [B, C, H, W]
    if depth.ndim == 3:
        depth = depth.unsqueeze(1)  # Add channel dimension
    B, _, H, W = depth.shape
    
    divergence_step = divergence / steps
    grid = make_grid(B, W, H, c.device)
    delta_scale = torch.tensor(1.0 / (W // 2 - 1), dtype=c.dtype, device=c.device)
    depth_warp = depth
    delta_steps = []

    for j in range(steps):
        # Ensure depth_mapped is 4D [B, 1, H, W]
        depth_mapped = get_mapper(mapper)(depth_warp)
        if depth_mapped.ndim == 3:
            depth_mapped = depth_mapped.unsqueeze(1)
            
        div_val, conv_val = make_divergence_feature_value(divergence_step, convergence, W)
        div_feat = torch.full_like(depth_mapped, div_val)
        conv_feat = torch.full_like(depth_mapped, conv_val)
        x = torch.cat([depth_mapped, div_feat, conv_feat], dim=1)
        
        # Updated autocast usage
        if enable_amp:
            with torch.amp.autocast(device_type='cuda' if 'cuda' in str(DEVICE) else 'cpu', enabled=True):
                delta = model(x)
        else:
            delta = model(x)

        delta_steps.append(delta)
        if j + 1 < steps:
            depth_warp = backward_warp(depth_warp, grid, delta, delta_scale)

    c_warp = c
    for delta in delta_steps:
        c_warp = backward_warp(c_warp, grid, delta, delta_scale)
    z = c_warp

    if shift > 0:
        z = torch.flip(z, (3,))

    return z
def create_stereo_images(rgb_tensor, depth_tensor, divergence=2.0, convergence=0.0):
    """
    Generate left/right eye images from RGB and depth tensors
    """
    # Prepare arguments
    mapper = "pow2"
    warp_steps = 1
    enable_amp = True
    
    # Generate stereo views
    left_eye = apply_divergence_nn_delta(
        get_stereo_model(),
        rgb_tensor,
        depth_tensor,
        divergence=divergence,
        convergence=convergence,
        steps=warp_steps,
        mapper=mapper,
        shift=-1,
        enable_amp=enable_amp
    )
    
    right_eye = apply_divergence_nn_delta(
        get_stereo_model(),
        rgb_tensor,
        depth_tensor,
        divergence=divergence,
        convergence=convergence,
        steps=warp_steps,
        mapper=mapper,
        shift=1,
        enable_amp=enable_amp
    )
    
    return left_eye, right_eye

# ======================================================================
# Core Depth Processing Functions
# ======================================================================
def process_tensor(img_rgb: np.ndarray, height) -> np.ndarray:
    if height < img_rgb.shape[0]:
        width = int(img_rgb.shape[1] / img_rgb.shape[0] * height)
        img_rgb = cv2.resize(img_rgb, (width, height), interpolation=cv2.INTER_AREA)
    rgb_tensor = torch.from_numpy(img_rgb).to(DEVICE, dtype=DTYPE)
    return rgb_tensor

def process(img_rgb: np.ndarray, height) -> np.ndarray:
    if height < img_rgb.shape[0]:
        width = int(img_rgb.shape[1] / img_rgb.shape[0] * height)
        img_rgb = cv2.resize(img_rgb, (width, height), interpolation=cv2.INTER_AREA)
    return img_rgb

def predict_depth(image_rgb: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(image_rgb).to(DEVICE, dtype=DTYPE, non_blocking=True)
    
    with lock: 
        tensor = tensor.permute(2, 0, 1).contiguous()
        tensor = tensor.float() / 255
        tensor = tensor.unsqueeze(0)

        tensor = F.interpolate(tensor, (DEPTH_RESOLUTION, DEPTH_RESOLUTION), mode='bilinear', align_corners=False)
        tensor = ((tensor - MEAN) / STD).contiguous()
        with torch.no_grad():
            tensor = tensor.to(dtype=MODEL_DTYPE).contiguous()
            depth = model(pixel_values=tensor).predicted_depth

        h, w = image_rgb.shape[:2]
        depth = F.interpolate(depth.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False)[0, 0]
        depth = depth / depth.max().clamp(min=1e-6)
        return depth

def predict_depth_tensor(image_rgb: np.ndarray) -> np.ndarray:
    with lock: 
        tensor = torch.from_numpy(image_rgb).to(DEVICE, dtype=DTYPE, non_blocking=True)
        rgb_c = tensor.permute(2, 0, 1).contiguous()
        tensor = rgb_c / 255
        tensor = tensor.unsqueeze(0)

        tensor = F.interpolate(tensor, (DEPTH_RESOLUTION, DEPTH_RESOLUTION), mode='bilinear', align_corners=False)
        tensor = ((tensor - MEAN) / STD).contiguous()
        with torch.no_grad():
            tensor = tensor.to(dtype=MODEL_DTYPE).contiguous()
            depth = model(pixel_values=tensor).predicted_depth

        h, w = image_rgb.shape[:2]
        depth = F.interpolate(depth.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False)[0, 0]
        depth = depth / depth.max().clamp(min=1e-6)
        return depth, rgb_c
    
def make_sbs(rgb_c, depth, ipd_uv=0.064, depth_strength=1.0, display_mode="Half-SBS"):
    with lock:
        # Convert inputs to proper 4D tensors
        rgb_tensor = rgb_c.float() / 255.0
        if rgb_tensor.ndim == 3:
            rgb_tensor = rgb_tensor.unsqueeze(0)  # [1, C, H, W]
            
        depth_tensor = depth
        if depth_tensor.ndim == 2:
            depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        elif depth_tensor.ndim == 3:
            depth_tensor = depth_tensor.unsqueeze(1)  # [1, 1, H, W]
        
        # Calculate divergence
        divergence = ipd_uv * 100 * depth_strength
        
        # Generate stereo images
        left, right = create_stereo_images(
            rgb_tensor, 
            depth_tensor,
            divergence=divergence
        )
        
        def pad_to_aspect(img, target_ratio=(16,9)):
            """Pad image to target aspect ratio"""
            # Handle both 3D (C,H,W) and 4D (B,C,H,W) tensors
            if img.dim() == 4:
                _, C, H, W = img.shape
                is_batched = True
            else:
                C, H, W = img.shape
                is_batched = False
            
            t_w, t_h = target_ratio
            r_img = W / H
            r_t = t_w / t_h
            
            if abs(r_img - r_t) < 0.001:
                return img
            elif r_img > r_t:
                new_H = int(round(W / r_t))
                pad_top = (new_H - H) // 2
                pad_bottom = new_H - H - pad_top
                padding = (0, 0, pad_top, pad_bottom)
            else:
                new_W = int(round(H * r_t))
                pad_left = (new_W - W) // 2
                pad_right = new_W - W - pad_left
                padding = (pad_left, pad_right, 0, 0)
            
            # Apply padding based on tensor dimensions
            if is_batched:
                # For 4D: (B, C, H, W) - pad last two dimensions
                return F.pad(img, padding, value=0)
            else:
                # For 3D: (C, H, W) - pad last two dimensions
                return F.pad(img, padding, value=0)

        
        # Pad to target aspect ratio
        left = pad_to_aspect(left)
        right = pad_to_aspect(right)

        if display_mode=="TAB":
            out = torch.cat([left, right], dim=2)  # Stack vertically
        else:
            out = torch.cat([left, right], dim=3)   # Stack horizontally

        if display_mode != "Full-SBS":
            # Determine target size based on display mode
            if display_mode == "TAB":
                target_size = (left.shape[2] // 2, left.shape[3])
            else:
                target_size = (left.shape[2], left.shape[3] // 2)
            
            out = F.interpolate(out, size=target_size, mode="area")

        # Convert to uint8
        out = (out * 255).clamp(0, 255).to(torch.uint8)
        
        # Convert to numpy (H, W, C)
        if out.dim() == 4:
            out = out[0]  # Remove batch dimension
        return out.permute(1, 2, 0).contiguous().cpu().numpy()
# Preload stereo model at startup
get_stereo_model()