import torch, math
import torch.nn as nn
import torch.nn.functional as F
import copy, types, sys
from depth import DEVICE

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
        if sys.platform == "darwin":
            # macOS, detach=False for compatibility
            self.pad = ReplicationPad2d(padding, detach=False)
        else:
            self.pad = nn.ReplicationPad2d(padding)

    def forward(self, x):
        return self.pad(x)


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
        output_dtye = rgb.dtype
        rgb = rgb.to(torch.float32)
        grid = grid.to(torch.float32)
        delta = delta.to(torch.float32)
        delta_scale = delta_scale.to(torch.float32)

        delta = torch.cat([delta, torch.zeros_like(delta)], dim=1)
        grid = grid + delta * delta_scale
        grid = grid.permute(0, 2, 3, 1)
        z = F.grid_sample(rgb, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return z.to(output_dtye)

    def _forward_default(self, x):
        rgb = x[:, 0:3, :, ]
        grid = x[:, 6:8, :, ]
        x = x[:, 3:6, :, ]  # depth + diverdence feature + convergence

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

def load_row_flow_model(device_id):
    
    model = load_model(ROW_FLOW_V3_URL, weights_only=True, device_ids=[device_id])[0].eval()
    model.symmetric = False
    model.delta_output = True

    return model


def create_stereo_model(
        device_id,
):

    return load_row_flow_model(device_id=device_id)


def _bench(name):
    from nunif.models import create_model
    import time
    device = "cuda:0"
    B = 4
    N = 100

    model = create_model(name).to(device).eval()
    x = torch.zeros((B, 8, 512, 512)).to(device)
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        z, *_ = model(x)
        print(z.shape)
        params = sum([p.numel() for p in model.parameters()])
        print(model.name, model.i2i_offset, model.i2i_scale, f"{params}")

    # benchmark
    torch.cuda.synchronize()
    t = time.time()
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        for _ in range(N):
            z = model(x)
    torch.cuda.synchronize()
    print(1 / ((time.time() - t) / (B * N)), "FPS")

def softplus01_legacy(depth, c=6):
    min_v = math.log(1 + math.exp(0 * 12.0 - c)) / (12 - c)
    max_v = math.log(1 + math.exp(1 * 12.0 - c)) / (12 - c)
    v = torch.log(1. + torch.exp(depth * 12.0 - c)) / (12 - c)
    return (v - min_v) / (max_v - min_v)


def softplus01(x, bias, scale):
    # x: 0-1 normalized
    min_v = math.log(1 + math.exp((0 - bias) * scale))
    max_v = math.log(1 + math.exp((1 - bias) * scale))
    v = torch.log(1. + torch.exp((x - bias) * scale))
    return (v - min_v) / (max_v - min_v)


def inv_softplus01(x, bias, scale):
    min_v = ((torch.zeros(1, dtype=x.dtype, device=x.device) - bias) * scale).expm1().clamp(min=1e-6).log()
    max_v = ((torch.ones(1, dtype=x.dtype, device=x.device) - bias) * scale).expm1().clamp(min=1e-6).log()
    v = ((x - bias) * scale).expm1().clamp(min=1e-6).log()
    return (v - min_v) / (max_v - min_v)


def distance_to_disparity(x, c):
    c1 = 1.0 + c
    min_v = c / c1
    return ((c / (c1 - x)) - min_v) / (1.0 - min_v)


def inv_distance_to_disparity(x, c):
    return ((c + 1) * x) / (x + c)


def shift_relative_depth(x, min_distance, max_distance=16):
    # convert x from dispariy space to distance space
    # reference: https://github.com/LiheYoung/Depth-Anything/issues/72#issuecomment-1937892879
    provisional_max_distance = min_distance + max_distance
    A = 1.0 / provisional_max_distance
    B = (1.0 / min_distance) - (1.0 / provisional_max_distance)
    distance = 1 / (A + B * x)

    # shift distance in distance space. old min_distance -> 1
    new_min_distance = 1.0
    distance = (new_min_distance - min_distance) + distance

    # back to disparity space
    new_x = 1.0 / distance

    # scale output to 0-1 range
    # NOTE: Do not use new_x.amin()/new_x.amax() to avoid re-normalization.
    #       This condition is required when using EMA normalization with look-ahead buffers.
    min_value = 1.0 / (max_distance + 1)
    value_range = 1.0 - 1.0 / (max_distance + 1)
    new_x = (new_x - min_value) / value_range

    return new_x


    
    
    
def backward_warp(c, grid, delta, delta_scale):
    grid = grid + delta * delta_scale
    if c.shape[2] != grid.shape[2] or c.shape[3] != grid.shape[3]:
        grid = F.interpolate(grid, size=c.shape[-2:],
                            mode="bilinear", align_corners=True, antialias=False)
    grid = grid.permute(0, 2, 3, 1)
    if DEVICE == "MPS":
        # MPS does not support bicubic and border
        mode = "bilinear"
        padding_mode = "reflection"
    else:
        mode = "bilinear"
        padding_mode = "border"

    z = F.grid_sample(c, grid, mode=mode, padding_mode=padding_mode, align_corners=True)
    z = torch.clamp(z, 0, 1)
    return z

def make_grid(batch, width, height, device):
    # TODO: xpu: torch.meshgrid causes fallback from XPU to CPU, but it is faster to simply do nothing
    mesh_y, mesh_x = torch.meshgrid(torch.linspace(-1, 1, height, device=device),
                                    torch.linspace(-1, 1, width, device=device), indexing="ij")
    mesh_y = mesh_y.reshape(1, 1, height, width).expand(batch, 1, height, width)
    mesh_x = mesh_x.reshape(1, 1, height, width).expand(batch, 1, height, width)
    grid = torch.cat((mesh_x, mesh_y), dim=1)
    return grid

def resolve_mapper_function(name):
    # https://github.com/nagadomi/nunif/assets/287255/0071a65a-62ff-4928-850c-0ad22bceba41
    if name == "pow2":
        return lambda x: x ** 2
    elif name == "none":
        return lambda x: x
    elif name == "softplus":
        return softplus01_legacy
    elif name == "softplus2":
        return lambda x: softplus01_legacy(x) ** 2
    elif name in {"mul_1", "mul_2", "mul_3"}:
        # for Relative Depth
        # https://github.com/nagadomi/nunif/assets/287255/2be5c0de-cb72-4c9c-9e95-4855c0730e5c
        param = {
            # none 1x
            "mul_1": {"bias": 0.343, "scale": 12},  # smooth 1.5x
            "mul_2": {"bias": 0.515, "scale": 12},  # smooth 2x
            "mul_3": {"bias": 0.687, "scale": 12},  # smooth 3x
        }[name]
        return lambda x: softplus01(x, **param)
    elif name in {"inv_mul_1", "inv_mul_2", "inv_mul_3"}:
        # for Relative Depth
        # https://github.com/nagadomi/nunif/assets/287255/f580b405-b0bf-4c6a-8362-66372b2ed930
        param = {
            # none 1x
            "inv_mul_1": {"bias": -0.002102, "scale": 7.8788},  # inverse smooth 1.5x
            "inv_mul_2": {"bias": -0.0003, "scale": 6.2626},    # inverse smooth 2x
            "inv_mul_3": {"bias": -0.0001, "scale": 3.4343},    # inverse smooth 3x
        }[name]
        return lambda x: inv_softplus01(x, **param)
    elif name in {"shift_30", "shift_20", "shift_14", "shift_08", "shift_06", "shift_045"}:
        # for Relative Depth
        # https://github.com/user-attachments/assets/7c953aae-101e-4337-82b4-10a073863d47
        param = {
            "shift_30": {"min_distance": 3.0},
            "shift_20": {"min_distance": 2.0},
            "shift_14": {"min_distance": 1.4},
            "shift_08": {"min_distance": 0.8},
            "shift_06": {"min_distance": 0.6},
            "shift_045": {"min_distance": 0.45},
        }[name]
        return lambda x: shift_relative_depth(x, **param)
    elif name in {"div_25", "div_10", "div_6", "div_4", "div_2", "div_1"}:
        # for Metric Depth (inverse distance)
        # TODO: There is no good reason for this parameter step
        # https://github.com/nagadomi/nunif/assets/287255/46c6b292-040f-4820-93fc-9e001cd53375
        param = {
            "div_25": 2.5,
            "div_10": 1,
            "div_6": 0.6,
            "div_4": 0.4,
            "div_2": 0.2,
            "div_1": 0.1,
        }[name]
        return lambda x: distance_to_disparity(x, param)
    else:
        raise NotImplementedError(f"mapper={name}")

def chain(x, functions):
    for f in functions:
        x = f(x)
    return x

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

    return lambda x: chain(x, functions)

def make_divergence_feature_value(divergence, convergence, image_width):
    # assert image_width <= 2048
    divergence_pix = divergence * 0.5 * 0.01 * image_width
    divergence_feature_value = divergence_pix / 32.0
    convergence_feature_value = (-divergence_pix * convergence) / 32.0

    return divergence_feature_value, convergence_feature_value

def make_input_tensor(c, depth, divergence, convergence,
                    image_width, mapper="pow2", preserve_screen_border=False):
    depth = depth.squeeze(0)  # CHW -> HW
    depth = get_mapper(mapper)(depth)
    divergence_value, convergence_value = make_divergence_feature_value(divergence, convergence, image_width)
    divergence_feat = torch.full_like(depth, divergence_value, device=depth.device)
    convergence_feat = torch.full_like(depth, convergence_value, device=depth.device)

    if preserve_screen_border:
        # Force set screen border parallax to zero.
        # Note that this does not work with tiled rendering (training code)
        border_pix = round(divergence * 0.75 * 0.01 * image_width * (depth.shape[-1] / image_width))
        if border_pix > 0:
            border_weight_l = torch.linspace(0.0, 1.0, border_pix, device=depth.device)
            border_weight_r = torch.linspace(1.0, 0.0, border_pix, device=depth.device)
            divergence_feat[:, :border_pix] = (border_weight_l[None, :].expand_as(divergence_feat[:, :border_pix]) *
                                            divergence_feat[:, :border_pix])
            divergence_feat[:, -border_pix:] = (border_weight_r[None, :].expand_as(divergence_feat[:, -border_pix:]) *
                                                divergence_feat[:, -border_pix:])
            convergence_feat[:, :border_pix] = (border_weight_l[None, :].expand_as(convergence_feat[:, :border_pix]) *
                                                convergence_feat[:, :border_pix])
            convergence_feat[:, -border_pix:] = (border_weight_r[None, :].expand_as(convergence_feat[:, -border_pix:]) *
                                                convergence_feat[:, -border_pix:])

def autocast(device, dtype=None, enabled=True):
    if DEVICE == "cpu":
        # autocast on cpu is extremely slow for unknown reasons
        # disabled
        amp_device_type = "cpu"
        amp_dtype = torch.bfloat16
        if enabled:
            enabled = False
    elif DEVICE == "mps":  # TODO: xpu work or not
        # currently pytorch does not support mps autocast
        # disabled
        amp_device_type = "cpu"
        amp_dtype = torch.bfloat16
        if enabled:
            enabled = False
    elif torch.cuda.is_avaliable():
        amp_device_type = device.split(":")[0] if isinstance(device, str) else device.type
        amp_dtype = dtype
        if False:
            # TODO: I think better to do this, but leave it to the user (use --disable-amp option)
            cuda_capability = torch.cuda.get_device_capability(device)
            if enabled and cuda_capability < (7, 0):
                enabled = False
    else:
        # Unknown device
        amp_device_type = device.split(":")[0] if isinstance(device, str) else device.type
        amp_dtype = dtype

    return torch.autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=enabled)

def apply_divergence_nn_delta(model, c, depth, divergence, convergence, steps,
                            mapper, shift, preserve_screen_border, enable_amp):
    steps = 1 if steps is None else steps
    # BCHW
    # assert model.delta_output
    if shift > 0:
        c = torch.flip(c, (3,))
        depth = torch.flip(depth, (3,))

    B, _, H, W = depth.shape
    divergence_step = divergence / steps
    grid = make_grid(B, W, H, c.device)
    delta_scale = torch.tensor(1.0 / (W // 2 - 1), dtype=c.dtype, device=c.device)
    depth_warp = depth
    delta_steps = []

    for j in range(steps):
        x = torch.stack([make_input_tensor(None, depth_warp[i],
                                        divergence=divergence_step,
                                        convergence=convergence,
                                        image_width=W,
                                        mapper=mapper,
                                        preserve_screen_border=preserve_screen_border)
                        for i in range(depth_warp.shape[0])])
        with autocast(device=depth.device, enabled=enable_amp):
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

def apply_divergence_nn_LR(model, c, depth, divergence, convergence, steps,
                        mapper, synthetic_view, preserve_screen_border, enable_amp):
    assert synthetic_view in {"both", "right", "left"}
    steps = 1 if steps is None else steps


    left_eye = apply_divergence_nn_delta(model, c, depth, divergence, convergence, steps,
                                mapper=mapper, shift=-1,
                                preserve_screen_border=preserve_screen_border,
                                enable_amp=enable_amp)
    right_eye = apply_divergence_nn_delta(model, c, depth, divergence, convergence, steps,
                                    mapper=mapper, shift=1,
                                    preserve_screen_border=preserve_screen_border,
                                    enable_amp=enable_amp)

    return left_eye, right_eye
def apply_divergence(depth, im, args, side_model):
    batch = True
    if depth.ndim != 4:
        # CHW
        depth = depth.unsqueeze(0)
        im = im.unsqueeze(0)
        batch = False
    else:
        # BCHW
        pass

    if args.stereo_width is not None:
        # NOTE: use src aspect ratio instead of depth aspect ratio
        H, W = im.shape[2:]
        stereo_width = min(W, args.stereo_width)
        if depth.shape[3] != stereo_width:
            new_w = stereo_width
            new_h = int(H * (stereo_width / W))
            depth = F.interpolate(depth, size=(new_h, new_w),
                                    mode="bilinear", align_corners=True, antialias=True)
            depth = torch.clamp(depth, 0, 1)
    left_eye, right_eye = apply_divergence_nn_LR(
        side_model, im, depth,
        args.divergence, args.convergence, args.warp_steps,
        mapper=args.mapper,
        synthetic_view=args.synthetic_view,
        preserve_screen_border=args.preserve_screen_border,
        enable_amp=not args.disable_amp)

    if not batch:
        left_eye = left_eye.squeeze(0)
        right_eye = right_eye.squeeze(0)

    return left_eye, right_eye

        
    
