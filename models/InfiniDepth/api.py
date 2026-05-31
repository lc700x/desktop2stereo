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
from models.InfiniDepth.model.model import InfiniDepth, _make_dense_query_coord


class InfiniDepthModel(nn.Module):
    """InfiniDepth — simplified API for dense relative depth estimation.

    Wraps the coordinate-query InfiniDepth model with automatic dense
    query-coordinate generation so callers can pass an image tensor and
    receive a dense depth map directly.

    CUDA is required. The model loads onto CUDA during __init__; move it
    elsewhere via ``model.to(device)`` after construction.
    """

    def __init__(self, model_path: str, encoder: str = "vitl16"):
        super().__init__()
        self.model = InfiniDepth(model_path=model_path, encoder=encoder)

    # Keyed by (B, H, W, device_str) — same key → reuse coord tensor.
    _coord_cache: dict = {}

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Dense depth prediction.

        Args:
            pixel_values: (B, 3, H, W) float tensor, RGB in [0, 1].

        Returns:
            depth: (B, H, W) relative depth (1 / disparity).
        """
        if pixel_values.dim() != 4:
            raise ValueError("Expected input shape (B, 3, H, W)")

        B, _, H, W = pixel_values.shape
        device = pixel_values.device
        input_dtype = pixel_values.dtype

        # The model's internal autocast is tuned for float32 input.
        # Running the implicit head / basic encoder in pure float16 loses
        # precision and produces blank depth maps. Cast to float32 for
        # inference, then restore the caller's dtype on output.
        x = pixel_values.float()

        cache_key = (B, H, W, str(device))
        if cache_key not in InfiniDepthModel._coord_cache:
            InfiniDepthModel._coord_cache[cache_key] = _make_dense_query_coord(B, H, W, device)
        query = InfiniDepthModel._coord_cache[cache_key]

        with torch.no_grad():
            # Use non-batched forward: processes all query points in a single
            # grid_sample + MLP pass.  The while-loop in batch_forward /
            # inference cannot be traced by ONNX / TensorRT.  For 512x512
            # input (262k points) this fits comfortably in GPU memory.
            pred = self.model.forward(x=x, coords=query)
            # depth = 1.0 / torch.clamp(pred, min=5e-3)

        depth = pred.reshape(B, 1, H, W).squeeze(1)
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
                with torch.autocast(device_type=pixel_values.device.type, enabled=not fp32):
                    return self.forward(pixel_values)
            else:
                return self.forward(pixel_values)
