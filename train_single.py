import os
import argparse
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from model import UNet  


class MembraneSegDataset(Dataset):
    def __init__(self, root_dir, split, patch_size=256, stride=128, transform=None):
        self.raw_dir = os.path.join(root_dir, split, "raw")
        self.mask_dir = os.path.join(root_dir, split, "membranes")
        self.filenames = sorted(os.listdir(self.raw_dir))
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.patches = []
        for img_idx, fname in enumerate(self.filenames):
            img = Image.open(os.path.join(self.raw_dir, fname))
            w, h = img.size
            for y in range(0, h - patch_size + 1, stride):
                for x in range(0, w - patch_size + 1, stride):
                    self.patches.append((img_idx, y, x))
        print(f"[{split}] Total patches: {len(self.patches)}")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img_idx, y, x = self.patches[idx]
        fname = self.filenames[img_idx]
        img = Image.open(os.path.join(self.raw_dir, fname)).convert("L")
        mask = Image.open(os.path.join(self.mask_dir, fname)).convert("L")
        img_patch = img.crop((x, y, x + self.patch_size, y + self.patch_size))
        mask_patch = mask.crop((x, y, x + self.patch_size, y + self.patch_size))
        if self.transform:
            seed = torch.seed()
            torch.manual_seed(seed)
            img_patch = self.transform(img_patch)
            torch.manual_seed(seed)
            mask_patch = T.ToTensor()(mask_patch)
        else:
            img_patch = T.ToTensor()(img_patch)
            mask_patch = T.ToTensor()(mask_patch)

        mask_patch = (mask_patch > 0.5).float()

        return img_patch, mask_patch



class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE
        return F_loss.mean()
    
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersect = (pred * target).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (2. * intersect + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()



def dice_coef(pred, target, smooth=1.):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersect = (pred * target).sum()
    return (2. * intersect + smooth) / (pred.sum() + target.sum() + smooth)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for imgs, masks in tqdm(loader, desc="Train", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)

def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_dice = 0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Val", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)
            total_loss += loss.item() * imgs.size(0)
            total_dice += dice_coef(preds, masks).item() * imgs.size(0)
    n = len(loader.dataset)
    return total_loss / n, total_dice / n



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data', help='Dataset root')
    parser.add_argument('--domain', required=True, help='Domain/folder name (e.g. c-elegans-dauer-stage)')
    parser.add_argument('--out_path', required=True, help='Path to save the trained model')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    domain_path = os.path.join(args.data_root, args.domain)
    print(f"Training on: {domain_path}")


    train_ds = MembraneSegDataset(domain_path, "train", patch_size=256, stride=128, transform=None)
    val_ds = MembraneSegDataset(domain_path, "val", patch_size=256, stride=128, transform=None)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    dice_loss = DiceLoss()
    focal_loss = FocalLoss(alpha=0.8, gamma=2)
    bce_loss = nn.BCEWithLogitsLoss()

    def combined_loss(pred, target):
        return dice_loss(pred, target) + focal_loss(pred, target) + bce_loss(pred, target)


    model = UNet(in_ch=1, out_ch=1).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = combined_loss

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, args.device)
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} ")
        if (epoch + 1) % 10 == 0:
            ckpt_path = args.out_path.replace(".pth", f"_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Model checkpoint saved at {ckpt_path}")
    final_ckpt = args.out_path.replace(".pth", f"_final.pth")
    torch.save(model.state_dict(), final_ckpt)
    print(f"Final model saved at {final_ckpt}")


if __name__ == '__main__':
    main()




