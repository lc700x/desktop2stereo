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
import matplotlib.pyplot as plt
from torch.utils.data import random_split

class TinyRefiner(nn.Module):
    def __init__(self, in_channels=10, out_channels=3, features=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, features, 3, padding=1)
        self.conv2 = nn.Conv2d(features, features, 3, padding=1)
        self.conv3 = nn.Conv2d(features, features, 3, padding=1)
        self.conv4 = nn.Conv2d(features, out_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        return torch.sigmoid(x)  # output normalized 0-1



# KITTI Stereo Dataset

class KITTIStereoDataset(Dataset):
    def __init__(self, root_dir, split="training", resize=(256,512)):
        left_images = sorted(glob.glob(os.path.join(root_dir, split, "image_2", "*.png")))
        right_images = sorted(glob.glob(os.path.join(root_dir, split, "image_3", "*.png")))
        disp_images = sorted(glob.glob(os.path.join(root_dir, split, "disp_occ_0", "*.png")))

        # Keep only files that exist in all three lists
        self.samples = []
        for l,r,d in zip(left_images, right_images, disp_images):
            if os.path.exists(l) and os.path.exists(r) and os.path.exists(d):
                self.samples.append((l,r,d))

        self.resize = resize

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        left_path, right_path, disp_path = self.samples[idx]
        left = cv2.imread(left_path)[:, :, ::-1] # H,W,3 RGB
        right = cv2.imread(right_path)[:, :, ::-1]
        disp = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0

        if self.resize is not None:
            h, w = self.resize
            left = cv2.resize(left, (w, h), interpolation=cv2.INTER_AREA)
            right = cv2.resize(right, (w, h), interpolation=cv2.INTER_AREA)
            disp = cv2.resize(disp, (w, h), interpolation=cv2.INTER_NEAREST)

        left_tensor = torch.from_numpy(left).permute(2,0,1).float() / 255.0 # C, H, W
        right_tensor = torch.from_numpy(right).permute(2,0,1).float() / 255.0 
        disp_tensor = torch.from_numpy(disp).unsqueeze(0)

        inp = torch.cat([left_tensor, left_tensor, right_tensor, disp_tensor], dim=0)
        return inp, right_tensor

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
def train_kitti(data_root, save_path="refiner_kitti.pt",
                epochs=20, batch_size=8, lr=1e-4, device="cuda", test_dataset=None):
    
    full_dataset = KITTIStereoDataset(data_root, split="training", resize=(256,512))
    val_size = int(0.1 * len(full_dataset))  # 10% for validation
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    else:
        test_loader = []

    model = TinyRefiner().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    l1_loss = nn.L1Loss()
    perceptual_loss = VGGPerceptualLoss().to(device)

    train_losses, val_losses, test_losses = [], [], []

    # plt.ion()
    fig, ax = plt.subplots(figsize=(8,5))

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inp, target in train_loader:
            inp = inp.to(device)
            target = target.to(device)

            pred = model(inp)
            loss_l1 = l1_loss(pred, target)
            loss_perc = perceptual_loss(pred, target)
            loss = loss_l1 + 0.1*loss_perc

            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader) if len(train_loader) > 0 else float('nan')
        train_losses.append(train_loss)

        # Validation
        if len(val_loader) > 0:
            val_loss_accum = 0.0
            model.eval()
            with torch.no_grad():
                for inp, target in val_loader:
                    inp = inp.to(device)
                    target = target.to(device)
                    pred = model(inp)
                    loss_l1 = l1_loss(pred, target)
                    loss_perc = perceptual_loss(pred, target)
                    val_loss_accum += (loss_l1 + 0.1*loss_perc).item()
            val_loss = val_loss_accum / len(val_loader)
        else:
            val_loss = float('nan')
        val_losses.append(val_loss)

        # Test (optional)
        if len(test_loader) > 0:
            test_loss_accum = 0.0
            model.eval()
            with torch.no_grad():
                for inp, target in test_loader:
                    inp = inp.to(device)
                    target = target.to(device)
                    pred = model(inp)
                    loss_l1 = l1_loss(pred, target)
                    loss_perc = perceptual_loss(pred, target)
                    test_loss_accum += (loss_l1 + 0.1*loss_perc).item()
            test_loss = test_loss_accum / len(test_loader)
        else:
            test_loss = float('nan')
        test_losses.append(test_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Test: {test_loss:.4f}")

        # Live plotting
        ax.clear()
        ax.plot(train_losses, label="Train Loss")
        if not np.isnan(val_loss):
            ax.plot(val_losses, label="Val Loss")
        if not np.isnan(test_loss):
            ax.plot(test_losses, label="Test Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("TinyRefiner KITTI Training Curve")
        ax.legend()
        # plt.pause(0.01)

    # plt.ioff()
    plt.savefig("training_curve.png")
    print("Saved training curve to training_curve.png")

    torch.save(model.state_dict(), save_path)
    print(f"Saved trained weights to {save_path}")

# Run training
if __name__ == "__main__":
    train_kitti(data_root="datasets/KITTI", epochs=5, batch_size=16, device="cuda")
