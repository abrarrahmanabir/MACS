import os
import argparse
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, recall_score, jaccard_score
import numpy as np
from scipy.stats import entropy
from model import UNet  
from tqdm import tqdm


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

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img_idx, y, x = self.patches[idx]
        fname = self.filenames[img_idx]
        img = Image.open(os.path.join(self.raw_dir, fname)).convert("L")
        mask = Image.open(os.path.join(self.mask_dir, fname)).convert("L")
        img_patch = img.crop((x, y, x + self.patch_size, y + self.patch_size))
        mask_patch = mask.crop((x, y, x + self.patch_size, y + self.patch_size))
        img_patch = self.transform(img_patch) if self.transform else T.ToTensor()(img_patch)
        mask_patch = T.ToTensor()(mask_patch)
        return img_patch, mask_patch


def dice_coef(pred, target, smooth=1.):
    pred = (pred > 0.5).float()
    intersect = (pred * target).sum()
    return (2. * intersect + smooth) / (pred.sum() + target.sum() + smooth)

def variation_of_information(y_true, y_pred, eps=1e-10):
    y_true = y_true.flatten().astype(np.int32)
    y_pred = y_pred.flatten().astype(np.int32)
    contingency = np.histogram2d(y_true, y_pred, bins=(2, 2))[0]
    Pxy = contingency / contingency.sum()
    Px = Pxy.sum(axis=1)
    Py = Pxy.sum(axis=0)
    Hx = entropy(Px + eps)
    Hy = entropy(Py + eps)
    Ixy = np.nansum(Pxy * np.log((Pxy + eps) / ((Px[:, None] * Py[None, :]) + eps)))
    return Hx + Hy - 2 * Ixy

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    dices, ious, f1s, recalls, vis = [], [], [], [], []

    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)
        preds = model(imgs)
        preds = torch.sigmoid(preds)
        preds_bin = (preds > 0.5).float()

        for p, m in zip(preds_bin.cpu(), masks.cpu()):
            p_np = p.squeeze().numpy()
            m_np = m.squeeze().numpy()
            dices.append(dice_coef(torch.tensor(p_np), torch.tensor(m_np)).item())
            ious.append(jaccard_score(m_np.flatten(), p_np.flatten(), zero_division=1))
            f1s.append(f1_score(m_np.flatten(), p_np.flatten(), zero_division=1))
            recalls.append(recall_score(m_np.flatten(), p_np.flatten(), zero_division=1))
            vis.append(variation_of_information(m_np, p_np))

    return {
        "dice": np.mean(dices),
        "iou": np.mean(ious),
        "f1": np.mean(f1s),
        "recall": np.mean(recalls),
        "vi": np.mean(vis)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data', help='Dataset root')
    parser.add_argument('--domain', required=True, help='Domain name (e.g. fly)')
    parser.add_argument('--model_path', required=True, help='Path to trained .pth model')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    domain_path = os.path.join(args.data_root, args.domain)
    test_ds = MembraneSegDataset(domain_path, "test", patch_size=256, stride=128, transform=T.ToTensor())
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    model = UNet(in_ch=1, out_ch=1).to(args.device)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    metrics = evaluate(model, test_loader, args.device)
    percent = args.model_path.split("_")[-1].replace(".pth", "")
    print(",".join([args.domain, percent] + [f"{v:.4f}" for v in metrics.values()]))

if __name__ == '__main__':
    main()


