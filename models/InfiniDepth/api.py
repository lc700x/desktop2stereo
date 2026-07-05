# Copyright (c) 2025. Modified by LC700X
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
InfiniDepth simplified API for dense depth estimation.

Provides a clean predict_depth(image) interface similar to Depth Anything 3,
handling dense query-coordinate generation and batched inference internally.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from models.InfiniDepth.model.model import InfiniDepth


class InfiniDepthModel(nn.Module):
    """InfiniDepth — simplified API for dense relative depth estimation.

    Wraps InfiniDepth with a dense full-image path so callers can pass an
    image tensor and receive a dense depth map directly.

    CUDA is required. The model loads onto CUDA during __init__; move it
    elsewhere via ``model.to(device)`` after construction.
    """

    def __init__(self, model_path: str, encoder: str = "vitl16"):
        super().__init__()
        self.model = InfiniDepth(model_path=model_path, encoder=encoder)

    def forward(self, pixel_values: torch.Tensor, fp32: bool = False) -> torch.Tensor:
        """Dense depth prediction.

        Args:
            pixel_values: (B, 3, H, W) float tensor, RGB in [0, 1].
            fp32: If True, force fp32 inference even on MPS/XPU (skips
                   internal autocast that would otherwise override to fp16).

        Returns:
            depth: (B, H, W) relative depth (1 / disparity).
        """
        if pixel_values.dim() != 4:
            raise ValueError("Expected input shape (B, 3, H, W)")

        B, _, _, _ = pixel_values.shape
        input_dtype = pixel_values.dtype
        param_dtype = next(
            (p.dtype for p in self.model.parameters() if p.is_floating_point()),
            pixel_values.dtype,
        )
        run_fp32 = fp32 and param_dtype == torch.float32

        # Keep activations consistent with the loaded parameter dtype. This
        # avoids FP32 input / FP16 bias mismatches during ONNX/MIGraphX export
        # while preserving true FP32 inference when the model is loaded as FP32.
        x = pixel_values.float() if run_fp32 else pixel_values.to(dtype=param_dtype)

        with torch.no_grad():
            pred = self.model.forward_dense(x=x, force_fp32=run_fp32)

        depth = pred.reshape(B, 1, pred.shape[-2], pred.shape[-1]).squeeze(1)
        if depth.dtype == torch.float16:
            depth = depth.clamp(-65504.0, 65504.0)
        if not torch.onnx.is_in_onnx_export():
            depth = torch.nan_to_num(depth, nan=0.0, posinf=65504.0, neginf=-65504.0)
        return depth.to(input_dtype)
    
    def predict_depth(self, pixel_values: torch.Tensor, fp32: bool = False) -> torch.Tensor:
        """High-level inference API with autocast.

        Uses dynamic device_type so it works on CUDA, MPS, XPU, etc.
        DirectML (privateuseone) skips autocast.

        Args:
            pixel_values: (B, 3, H, W) float tensor, RGB in [0, 1].
            fp32: If True, disable autocast (full FP32 inference).

        Returns:
            depth: (B, H, W) relative depth map.
        """
        with torch.no_grad():
            if pixel_values.device.type != "privateuseone":
                use_autocast = (not fp32) and (not torch.onnx.is_in_onnx_export())
                with torch.autocast(device_type=pixel_values.device.type, enabled=use_autocast):
                    return self.forward(pixel_values, fp32=fp32)
            else:
                return self.forward(pixel_values, fp32=fp32)
