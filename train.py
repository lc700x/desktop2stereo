# train_refiner_kitti.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import vgg16
import cv2
import glob
import os
import numpy as np




# KITTI Stereo Dataset

class KITTIStereoDataset(Dataset):
    def __init__(self, root_dir, split="training", resize=(256,512)):
        self.left_images = sorted(glob.glob(os.path.join(root_dir, split, "image_2", "*.png")))
        self.right_images = sorted(glob.glob(os.path.join(root_dir, split, "image_3", "*.png")))
        self.disp_images = sorted(glob.glob(os.path.join(root_dir, split, "disp_occ_0", "*.png")))
        self.resize = resize

    def __len__(self):
        return len(self.left_images)

    def __getitem__(self, idx):
        left = cv2.imread(self.left_images[idx])[:, :, ::-1]  # BGR -> RGB
        right = cv2.imread(self.right_images[idx])[:, :, ::-1]
        disp = cv2.imread(self.disp_images[idx], cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0  # KITTI stores disparity*256

        if self.resize is not None:
            h, w = self.resize
            left = cv2.resize(left, (w, h), interpolation=cv2.INTER_AREA)
            right = cv2.resize(right, (w, h), interpolation=cv2.INTER_AREA)
            disp = cv2.resize(disp, (w, h), interpolation=cv2.INTER_NEAREST)

        # Convert to tensor
        to_tensor = T.ToTensor()
        left_tensor = to_tensor(left)
        right_tensor = to_tensor(right)
        disp_tensor = torch.from_numpy(disp).unsqueeze(0)  # 1,H,W

        # Input to TinyRefiner = [left, left, right, depth] -> 3+3+3+1=10 channels
        inp = torch.cat([left_tensor, left_tensor, right_tensor, disp_tensor], dim=0)

        return inp, right_tensor  # target is ground-truth right image


# Perceptual Loss

class VGGPerceptualLoss(nn.Module):
    def __init__(self, layer_ids=[3,8,15]):
        super().__init__()
        vgg = vgg16(pretrained=True).features.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.layers = nn.ModuleList([vgg[i] for i in range(max(layer_ids)+1)])
        self.layer_ids = set(layer_ids)

    def forward(self, x, y):
        loss = 0.0
        for i, layer in enumerate(self.layers):
            x = layer(x)
            y = layer(y)
            if i in self.layer_ids:
                loss += nn.functional.l1_loss(x, y)
        return loss


# Training Loop

def train_kitti(data_root, save_path="refiner_kitti.pt", epochs=20, batch_size=8, lr=1e-4, device="cuda"):
    dataset = KITTIStereoDataset(data_root, split="training", resize=(256,512))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = TinyRefiner().to(device)
    opt = optim.AdamW(model.parameters(), lr=lr)
    l1_loss = nn.L1Loss()
    perceptual_loss = VGGPerceptualLoss().to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inp, target in loader:
            inp = inp.to(device)
            target = target.to(device)

            pred = model(inp)

            loss_l1 = l1_loss(pred, target)
            loss_perc = perceptual_loss(pred, target)
            loss = loss_l1 + 0.1 * loss_perc

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(loader):.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Saved trained weights to {save_path}")


# Run training
if __name__ == "__main__":
    train_kitti(data_root="datasets/KITTI", epochs=10, batch_size=8, device="cuda")
