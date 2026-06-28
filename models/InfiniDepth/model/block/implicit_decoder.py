import enum
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def _grid_sample_bilinear(input, grid, align_corners=False, padding_mode="zeros"):
    """DirectML-friendly bilinear grid_sample that stays on the GPU.

    torch-directml has no aten::grid_sampler_2d kernel and silently falls back
    to CPU (slow). This reimplements the same math with gather + arithmetic,
    all of which run on the DML device.

    Matches F.grid_sample(input, grid, mode='bilinear', align_corners=...,
    padding_mode='zeros'|'border'). grid last dim is (x, y) in [-1, 1].

    Args:
        input: [B, C, H, W]
        grid:  [B, Hg, Wg, 2]
    Returns:
        [B, C, Hg, Wg]
    """
    B, C, H, W = input.shape
    _, Hg, Wg, _ = grid.shape

    x = grid[..., 0]
    y = grid[..., 1]

    # Normalized [-1, 1] -> pixel coordinates.
    if align_corners:
        ix = (x + 1) * 0.5 * (W - 1)
        iy = (y + 1) * 0.5 * (H - 1)
    else:
        ix = ((x + 1) * W - 1) * 0.5
        iy = ((y + 1) * H - 1) * 0.5

    x0 = torch.floor(ix)
    y0 = torch.floor(iy)
    x1 = x0 + 1
    y1 = y0 + 1

    # Bilinear weights.
    wx1 = ix - x0
    wx0 = 1.0 - wx1
    wy1 = iy - y0
    wy0 = 1.0 - wy1
    w00 = wx0 * wy0
    w01 = wx1 * wy0
    w10 = wx0 * wy1
    w11 = wx1 * wy1

    if padding_mode == "zeros":
        # Out-of-bounds corners contribute nothing.
        def _mask(xc, yc):
            return ((xc >= 0) & (xc <= W - 1) & (yc >= 0) & (yc <= H - 1)).to(input.dtype)
        w00 = w00 * _mask(x0, y0)
        w01 = w01 * _mask(x1, y0)
        w10 = w10 * _mask(x0, y1)
        w11 = w11 * _mask(x1, y1)

    # Clamp indices so gather is always in range (border behaviour for the read;
    # the zeros mask above zeroes the weight where it mattered).
    x0c = x0.clamp(0, W - 1).long()
    x1c = x1.clamp(0, W - 1).long()
    y0c = y0.clamp(0, H - 1).long()
    y1c = y1.clamp(0, H - 1).long()

    inp_flat = input.reshape(B, C, H * W)

    def _gather(xc, yc):
        idx = (yc * W + xc).reshape(B, 1, Hg * Wg).expand(B, C, Hg * Wg)
        return torch.gather(inp_flat, 2, idx).reshape(B, C, Hg, Wg)

    v00 = _gather(x0c, y0c)
    v01 = _gather(x1c, y0c)
    v10 = _gather(x0c, y1c)
    v11 = _gather(x1c, y1c)

    return (v00 * w00.unsqueeze(1) + v01 * w01.unsqueeze(1)
            + v10 * w10.unsqueeze(1) + v11 * w11.unsqueeze(1))


def _grid_sample(input, grid):
    """Bilinear sample with align_corners=False, zeros padding.

    Uses the GPU gather fallback on DirectML; native F.grid_sample elsewhere."""
    if input.device.type == "privateuseone":
        return _grid_sample_bilinear(input, grid, align_corners=False, padding_mode="zeros")
    return F.grid_sample(input, grid, mode="bilinear", align_corners=False, padding_mode="zeros")


class ELU(nn.Module):
    """ELU activation that stays on the GPU on DirectML.

    torch-directml has no aten::elu kernel and falls back to CPU. The
    where/exp form is identical (clamp(max=0) keeps exp from overflowing on the
    unused positive branch) and runs on-device. Other backends use native F.elu.
    """
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        if x.device.type == "privateuseone":
            return torch.where(x > 0, x, self.alpha * (torch.exp(torch.clamp(x, max=0.0)) - 1.0))
        return F.elu(x, self.alpha)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list, output_act='elu'):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers += [nn.Linear(lastv, hidden), nn.ReLU()]
            lastv = hidden

        if out_dim is not None:
            layers.append(nn.Linear(lastv, out_dim))
            act = {
                "sigmoid": nn.Sigmoid(),
                "relu": nn.ReLU(),
                "elu": ELU(),
            }.get(output_act, nn.Identity())
            layers.append(act)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    

class ImplicitHead(nn.Module):
    """
    Implicit head that fuses DINOv3 semantic features and BasicEncoder low-level features.

    Args:
        hidden_dim: DINOv3 feature dimension (e.g., 1024)
        basic_dim: BasicEncoder feature dimension (e.g., 128)
        fusion_type: Feature fusion strategy
            - "concat": Simple concatenation
            - "cross_attn": Cross-attention between features
            - "gated": Gated fusion with learnable weights
        out_dim: Output dimension (1 for depth)
        hidden_list: MLP hidden layer dimensions
    """
    def __init__(
            self,
            hidden_dim,  # 1024 for DINOv3
            basic_dim=128,  # BasicEncoder output dim
            fusion_type="gated",  # concat, gated
            out_dim=1,
            hidden_list=[1024, 256, 32],
            ):

        super().__init__()
        self.hidden_dim = hidden_dim
        self.basic_dim = basic_dim
        self.fusion_type = fusion_type

        # Determine input dimension based on fusion type
        if fusion_type == "concat":
            # Simple concatenation
            in_channels = hidden_dim + basic_dim
        elif fusion_type == "gated":
            # Gated fusion with learnable weights
            self.gate_proj = nn.Linear(basic_dim, hidden_dim)
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            )
            in_channels = hidden_dim
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

        self.out_layer = MLP(
            in_dim=in_channels,
            out_dim=out_dim,
            hidden_list=hidden_list,
            output_act='elu'
        )

    def _encode_feat(self, features, patch_h, patch_w):
        """Extract DINOv3 feature map."""
        x = features[-1][0]
        out_feat = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
        return out_feat

    def _decode_dpt(self, feat, basic_feat, coord):
        """
        Query features at given coordinates and fuse them.

        Args:
            feat: DINOv3 feature map [B, hidden_dim, H_dino, W_dino]
            basic_feat: BasicEncoder feature map [B, basic_dim, H_basic, W_basic]
            coord: Query coordinates [B, N, 2] in range [-1, 1]

        Returns:
            pred: Predicted depth [B, N, 1]
        """
        coord_ = coord.clamp(-1 + 1e-6, 1 - 1e-6)

        # Reverse the (y, x) last dim to grid_sample's (x, y) order. Use fancy
        # indexing instead of .flip(-1): aten::flip has no DirectML kernel and
        # segfaults (0xC0000005); [..., [1, 0]] is identical for a size-2 dim
        # and works on every backend.
        grid = coord_[..., [1, 0]].unsqueeze(1)

        # Sample DINOv3 features at query coordinates
        q_feat_dino = _grid_sample(
            feat, grid
        )[:, :, 0, :].permute(0, 2, 1)  # [B, N, hidden_dim]

        # Sample BasicEncoder features at query coordinates (if available)
        if basic_feat is not None:
            q_feat_basic = _grid_sample(
                basic_feat, grid
            )[:, :, 0, :].permute(0, 2, 1)  # [B, N, basic_dim]

            # Fuse features based on fusion type
            q_feat_fused = self._fuse_features(q_feat_dino, q_feat_basic)
        else:
            # If no basic features, use only DINOv3
            q_feat_fused = q_feat_dino

        # Predict depth
        pred = self.out_layer(q_feat_fused)
        return pred

    def _fuse_features(self, feat_dino, feat_basic):
        """
        Fuse DINOv3 and BasicEncoder features.

        Args:
            feat_dino: [B, N, hidden_dim]
            feat_basic: [B, N, basic_dim]

        Returns:
            fused_feat: [B, N, fused_dim]
        """
        if self.fusion_type == "concat":
            # Simple concatenation
            return torch.cat([feat_dino, feat_basic], dim=-1)

        elif self.fusion_type == "gated":
            # Gated fusion with learnable weights
            feat_basic_proj = self.gate_proj(feat_basic)  # [B, N, hidden_dim]
            gate_input = torch.cat([feat_dino, feat_basic_proj], dim=-1)
            gate_weights = self.gate(gate_input)  # [B, N, hidden_dim]
            return gate_weights * feat_dino + (1 - gate_weights) * feat_basic_proj

    @staticmethod
    def _dense_zero_padding_mask(input_h, input_w, output_h, output_w, device, dtype):
        """Mask F.interpolate edges so dense upsample matches grid_sample zero padding."""
        # ONNX export cannot constant-fold Half Range reliably. Build the
        # coordinate mask in fp32, then cast back so FP16 model outputs stay FP16.
        work_dtype = torch.float32 if dtype == torch.float16 else dtype
        yy = (torch.arange(output_h, device=device, dtype=work_dtype) + 0.5) * (
            float(input_h) / float(output_h)
        ) - 0.5
        xx = (torch.arange(output_w, device=device, dtype=work_dtype) + 0.5) * (
            float(input_w) / float(output_w)
        ) - 0.5
        one_y = torch.ones_like(yy)
        one_x = torch.ones_like(xx)
        wy = torch.where(
            yy < 0,
            yy + 1.0,
            torch.where(yy > float(input_h - 1), float(input_h) - yy, one_y),
        ).clamp(0.0, 1.0)
        wx = torch.where(
            xx < 0,
            xx + 1.0,
            torch.where(xx > float(input_w - 1), float(input_w) - xx, one_x),
        ).clamp(0.0, 1.0)
        return (wy.view(1, 1, output_h, 1) * wx.view(1, 1, 1, output_w)).to(dtype=dtype)

    def _dense_sample(self, feat, output_h, output_w):
        sampled = F.interpolate(
            feat,
            size=(output_h, output_w),
            mode="bilinear",
            align_corners=False,
        )
        mask = self._dense_zero_padding_mask(
            feat.shape[-2],
            feat.shape[-1],
            output_h,
            output_w,
            feat.device,
            feat.dtype,
        )
        return sampled * mask

    def _decode_dense_dpt(self, feat, basic_feat, output_h, output_w):
        """
        Dense full-image decoder.

        This is mathematically equivalent to querying _make_dense_query_coord at
        every output pixel, but avoids exporting GridSample for MIGraphX.
        """
        feat_dino = self._dense_sample(feat, output_h, output_w).flatten(2).permute(0, 2, 1)
        if basic_feat is not None:
            feat_basic = self._dense_sample(basic_feat, output_h, output_w).flatten(2).permute(0, 2, 1)
            feat_fused = self._fuse_features(feat_dino, feat_basic)
        else:
            feat_fused = feat_dino

        pred = self.out_layer(feat_fused)
        return pred.permute(0, 2, 1).reshape(feat.shape[0], -1, output_h, output_w)

    def forward_dense(self, features, basic_feat, patch_h, patch_w, output_h, output_w):
        feat = self._encode_feat(features, patch_h, patch_w)
        return self._decode_dense_dpt(feat, basic_feat, output_h, output_w)

    def forward(self, features, basic_feat, patch_h, patch_w, coords):
        """
        Forward pass.

        Args:
            features: DINOv3 features from backbone
            basic_feat: BasicEncoder features [B, basic_dim, H/4, W/4]
            patch_h, patch_w: DINOv3 feature map spatial size
            coords: Query coordinates [B, N, 2]

        Returns:
            dpt_pred: Predicted depth [B, N, 1]
        """
        # Extract DINOv3 feature map
        feat = self._encode_feat(features, patch_h, patch_w)  # [B, hidden_dim, H/14, W/14]

        # Query and fuse features at coordinates
        dpt_pred = self._decode_dpt(feat, basic_feat, coords)

        return dpt_pred
