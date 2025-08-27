import math
import torch
import torch.nn.functional as F

def pad_to_aspect_torch(img: torch.Tensor, target_aspect=(16, 9), fill=0):
    """
    Pad a torch tensor image to the target aspect ratio (width:height).
    - img: tensor with shape (C, H, W) or (H, W)
    - target_aspect: tuple (W_ratio, H_ratio) e.g. (16,9)
    - fill: pad value (0 = black)
    Returns: padded tensor same dtype as input, shape (C, new_H, new_W)
    """
    # normalize shape to (C, H, W)
    was_2d = False
    if img.dim() == 2:
        was_2d = True
        img = img.unsqueeze(0)
    if img.dim() != 3:
        raise ValueError("img must be 2D (H,W) or 3D (C,H,W)")

    C, H, W = img.shape
    target_w_ratio, target_h_ratio = target_aspect
    target_ar = float(target_w_ratio) / float(target_h_ratio)  # width / height

    current_ar = float(W) / float(H)

    if abs(current_ar - target_ar) < 1e-9:
        # already exactly the target aspect
        return img if not was_2d else img.squeeze(0)

    if current_ar > target_ar:
        # image too wide -> pad height
        new_H = math.ceil(W / target_ar)
        new_W = W
    else:
        # image too tall (or narrow) -> pad width
        new_H = H
        new_W = math.ceil(H * target_ar)

    pad_h = new_H - H
    pad_w = new_W - W

    # split equally, if odd put extra pixel on bottom/right
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # F.pad expects pad=(pad_left, pad_right, pad_top, pad_bottom) for 3D (C,H,W)
    padded = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=fill)

    return padded if not was_2d else padded.squeeze(0)

# Example:
x = torch.randn(3, 2, 4)   # C=3, H=100, W=200
y = pad_to_aspect_torch(x)     # result will have aspect 16:9
print(x)
print(y)
