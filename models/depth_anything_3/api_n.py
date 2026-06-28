# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates. Modified by LC700X
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Depth Anything 3 simplified API for dense depth estimation.

Provides a clean predict_depth(image) interface similar to InfiniDepth,
handling the multi-view input format internally for single-image inference.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import contextlib
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from .cfg import create_object, load_config
from .registry import MODEL_REGISTRY


def maybe_autocast(device, enabled=True):
    return (
        torch.autocast(device_type=device.type, enabled=enabled)
        if device.type != "privateuseone"  # privateuseone is DirectML
        else contextlib.nullcontext()
    )


class DepthAnything3(nn.Module, PyTorchModelHubMixin):
    """Depth Anything 3 — simplified API for dense depth estimation.

    Wraps the multi-view DA3 model so callers can pass a single image tensor
    and receive a dense depth map directly.

    Supports CUDA, MPS, XPU, DirectML, and CPU via runtime autocast.
    """

    _commit_hash: str | None = None

    def __init__(self, model_name: str = "da3-large", **kwargs):
        super().__init__()
        self.model_name = model_name

        self.config = load_config(MODEL_REGISTRY[self.model_name])
        self.model = create_object(self.config)
        self.model.eval()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Dense depth prediction.

        Args:
            pixel_values: (B, 3, H, W) float tensor, ImageNet-normalized.

        Returns:
            depth: (B, H, W) depth map.
        """
        if pixel_values.dim() != 4:
            raise ValueError("Expected input shape (B, 3, H, W)")

        input_dtype = pixel_values.dtype
        param_dtype = next(
            (p.dtype for p in self.model.parameters() if p.is_floating_point()),
            pixel_values.dtype,
        )

        # DA3 internal expects (B, N, 3, H, W), N=1 for single image
        x = pixel_values.to(dtype=param_dtype).unsqueeze(1)

        # Patch: make resize_layers dtype-safe for export
        for layer in getattr(self.model, "resize_layers", []):
            if hasattr(layer, "weight") and x.dtype != layer.weight.dtype:
                x = x.to(dtype=layer.weight.dtype)

        out = self.model(x)

        if isinstance(out, dict):
            if "depth" in out:
                depth = out["depth"]
            elif "predicted_depth" in out:
                depth = out["predicted_depth"]
            else:
                raise RuntimeError("Depth key not found in model output")
        else:
            depth = out

        if depth.dim() == 4:
            depth = depth.squeeze(1)

        if depth.dtype == torch.float16:
            depth = depth.clamp(-65504.0, 65504.0)
        if not torch.onnx.is_in_onnx_export():
            depth = torch.nan_to_num(depth, nan=0.0, posinf=65504.0, neginf=-65504.0)
        return depth.to(dtype=input_dtype)

    def predict_depth(self, pixel_values: torch.Tensor, fp32: bool = False) -> torch.Tensor:
        """High-level inference API with autocast.

        Args:
            pixel_values: (B, 3, H, W) float tensor, ImageNet-normalized.
            fp32: If True, disable autocast (full FP32 inference).

        Returns:
            depth: (B, H, W) depth map.
        """
        with torch.no_grad():
            with maybe_autocast(pixel_values.device, enabled=not fp32):
                return self.forward(pixel_values)
