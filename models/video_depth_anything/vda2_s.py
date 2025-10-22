# Modified depth estimation process for Desktop2Stereo by lc700x
#####################################################
# Copyright (2025) Bytedance Ltd. and/or its affiliates 

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import Compose
import cv2
import numpy as np

from .dinov2 import DINOv2
from .dpt_temporal import DPTHeadTemporal
from .util.transform import Resize, NormalizeImage, PrepareForNet

# check CUDA
CUDA = True if torch.cuda.is_available() else False


# infer settings, do not change
INFER_LEN = 32
OVERLAP = 10
INTERP_LEN = 8

class VideoDepthAnything(nn.Module):
    def __init__(
        self,
        encoder='vitl',
        features=256, 
        out_channels=[256, 512, 1024, 1024], 
        use_bn=False, 
        use_clstoken=False,
        num_frames=32,
        pe='ape'
    ):
        super(VideoDepthAnything, self).__init__()

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            "vitb": [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23]
        }
        self.predicted_depth = None
        self.hidden_cache = None
        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)

        self.head = DPTHeadTemporal(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken, num_frames=num_frames, pe=pe)
        self.transform = None
        self.frame_id_list = []
        self.frame_cache_list = []
        self.gap = (INFER_LEN - OVERLAP) * 2 - 1 - (OVERLAP - INTERP_LEN)
        assert self.gap == 41
        self.id = -1

    # def forward(self, x):
    #     return self.forward_depth(self.forward_features(x), x.shape)[0]
    
    def forward_features(self, x):
        features = self.pretrained.get_intermediate_layers(x.flatten(0,1), self.intermediate_layer_idx[self.encoder], return_class_token=True)
        return features

    def forward_depth(self, features, x_shape, cached_hidden_state_list=None):
        B, T, C, H, W = x_shape
        patch_h, patch_w = H // 14, W // 14
        depth, cur_cached_hidden_state_list = self.head(features, patch_h, patch_w, T, cached_hidden_state_list=cached_hidden_state_list)
        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=True)
        depth = F.relu(depth)
        return depth.squeeze(1).unflatten(0, (B, T)), cur_cached_hidden_state_list # return shape [B, T, H, W]
    
    # def predict_depth_vda2(self, frame, input_size: int = 518, device='cuda', dtype=torch.float16):
    #     self.id += 1

    #     if self.transform is None:  # first frame
    #         # Initialize the transform
    #         frame_height, frame_width = frame.shape[:2]
    #         self.frame_height = frame_height
    #         self.frame_width = frame_width
    #         # ratio = max(frame_height, frame_width) / min(frame_height, frame_width)
    #         # if ratio > 1.78:  # we recommend to process video with ratio smaller than 16:9 due to memory limitation
    #         #     input_size = int(input_size * 1.777 / ratio)
    #         #     input_size = round(input_size / 14) * 14

    #         self.transform = Compose([
    #             Resize(
    #                 width=input_size,
    #                 height=input_size,
    #                 resize_target=False,
    #                 keep_aspect_ratio=True,
    #                 ensure_multiple_of=14,
    #                 resize_method='lower_bound',
    #                 image_interpolation_method=cv2.INTER_CUBIC,
    #             ),
    #             NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #             PrepareForNet(),
    #         ])

    #         # Inference the first frame
    #         cur_list = [torch.from_numpy(self.transform({'image': frame.astype(np.float32) / 255.0})['image']).unsqueeze(0).unsqueeze(0)]
    #         cur_input = torch.cat(cur_list, dim=1).to(device, dtype=dtype)

    #         with torch.no_grad():
    #         # with torch.amp.autocast('cuda'):
    #             cur_feature = self.forward_features(cur_input)
    #             x_shape = cur_input.shape
    #             depth, cached_hidden_state_list = self.forward_depth(cur_feature, x_shape)

    #         # depth = depth.to(cur_input.dtype)
    #         depth = F.interpolate(depth.flatten(0,1).unsqueeze(1), size=(frame_height, frame_width), mode='bilinear', align_corners=True)

    #         # Copy multiple cache to simulate the windows
    #         self.frame_cache_list = [cached_hidden_state_list] * INFER_LEN
    #         self.frame_id_list.extend([0] * (INFER_LEN - 1))

    #         new_depth = depth[0][0]
    #     else:
    #         frame_height, frame_width = frame.shape[:2]
    #         assert frame_height == self.frame_height
    #         assert frame_width == self.frame_width

    #         # infer feature
    #         cur_input = torch.from_numpy(self.transform({'image': frame.astype(np.float32) / 255.0})['image']).unsqueeze(0).unsqueeze(0).to(device, dtype=dtype)

    #         with torch.no_grad():
    #         # with torch.amp.autocast('cuda'):
    #             cur_feature = self.forward_features(cur_input)
    #             x_shape = cur_input.shape

    #         cur_list = self.frame_cache_list[0:2] + self.frame_cache_list[-INFER_LEN+3:]
    #         '''
    #         cur_id = self.frame_id_list[0:2] + self.frame_id_list[-INFER_LEN+3:]
    #         print(f"cur_id: {cur_id}")
    #         '''
    #         assert len(cur_list) == INFER_LEN - 1
    #         cur_cache = [torch.cat([h[i] for h in cur_list], dim=1) for i in range(len(cur_list[0]))]

    #         # infer depth
    #         with torch.no_grad():
    #         # with torch.amp.autocast('cuda'):
    #             depth, new_cache = self.forward_depth(cur_feature, x_shape, cached_hidden_state_list=cur_cache)

    #         # depth = depth.to(cur_input.dtype)
    #         depth = F.interpolate(depth.flatten(0,1).unsqueeze(1), size=(frame_height, frame_width), mode='bilinear', align_corners=True)
    #         new_depth = depth[-1, 0]

    #         self.frame_cache_list.append(new_cache)

    #     # adjust the sliding window
    #     self.frame_id_list.append(self.id)
    #     if self.id + INFER_LEN > self.gap + 1:
    #         del self.frame_id_list[1]
    #         del self.frame_cache_list[1]

    #     return new_depth
    
    def update_cache(self, new_cache):
        for i in range(len(self.hidden_cache)):
            # Drop oldest slice, append new one
            old = self.hidden_cache[i]  # shape: (B, C*(INFER_LEN-1), H, W)
            new = new_cache[i]          # shape: (B, C, H, W)

            # Shift left: [C, …, C*(N-2)] <- [C, …, C*(N-1)]
            old[:, :-new.shape[1]] = old[:, new.shape[1]:].clone()

            # Insert newest at the end
            old[:, -new.shape[1]:] = new
    
    def forward(self, pixel_values, device='cuda', fp32=False):
        self.id += 1
        cur_input = pixel_values.unsqueeze(0)

        if not self.transform:  # first frame
            # Inference the first frame
            cur_list = [cur_input]

            if not CUDA:
                with torch.no_grad():
                    cur_feature = self.forward_features(cur_input)
                    x_shape = cur_input.shape
                    self.predicted_depth, cached_hidden_state_list = self.forward_depth(cur_feature, x_shape)
            else:
                with torch.amp.autocast('cuda'):
                    with torch.no_grad():
                        cur_feature = self.forward_features(cur_input)
                        x_shape = cur_input.shape
                        self.predicted_depth, cached_hidden_state_list = self.forward_depth(cur_feature, x_shape)
        
            
            # Build pre-concatenated cache
            self.hidden_cache = [
                torch.cat([cached_hidden_state_list[i]] * (INFER_LEN  - 1), dim=1).contiguous()
                for i in range(len(cached_hidden_state_list))
            ]

            self.frame_id_list.extend([0] * (INFER_LEN  - 1))
            self.transform = True
        else:
            if not CUDA:
                with torch.no_grad():
                    cur_feature = self.forward_features(cur_input)
                    x_shape = cur_input.shape
            else:
                with torch.amp.autocast('cuda'):
                    with torch.no_grad():
                        cur_feature = self.forward_features(cur_input)
                        x_shape = cur_input.shape
            '''
            cur_id = self.frame_id_list[0:2] + self.frame_id_list[-INFER_LEN+3:]
            print(f"cur_id: {cur_id}")
            '''
            cur_cache = self.hidden_cache

            # infer depth
            if not CUDA:
                with torch.no_grad():
                    self.predicted_depth, new_cache = self.forward_depth(cur_feature, x_shape, cached_hidden_state_list=cur_cache)
            else:
                with torch.amp.autocast('cuda'):
                    with torch.no_grad():
                        self.predicted_depth, new_cache = self.forward_depth(cur_feature, x_shape, cached_hidden_state_list=cur_cache)

            self.update_cache(new_cache)
            # Sliding window housekeeping
            self.frame_id_list.append(self.id)
            if self.id + INFER_LEN  > self.gap + 1:
                del self.frame_id_list[1]
        return self.predicted_depth.squeeze(1)  # return shape [T, H, W]
