import os
import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import numpy as np
from tqdm import tqdm
from model import UNet
import torchvision.transforms as T
from sklearn.metrics.pairwise import pairwise_kernels
from scipy.optimize import minimize


def extract_features(model, dataloader, device):
    model.eval()
    features = []
    with torch.no_grad():
        for imgs, _ in tqdm(dataloader, desc="Extract Features"):
            imgs = imgs.to(device)
            # Use UNet encoder up to final down layer
            x = imgs
            skips = []
            for down in model.downs:
                x = down(x)
                skips.append(x)
                x = model.pool(x)
            # Bottleneck layer
            x = model.bottleneck(x)
            # Global max pooling
            pooled = torch.amax(x, dim=(2,3))
            features.append(pooled.cpu())
    return torch.cat(features, dim=0).numpy() # shape: [N, C]



def compute_mk_mmd(Xs, Y, kernels='rbf', lambda_entropy=0.1):
    K = len(Xs)
    k_vals = []
    for i in range(K):
        XX = pairwise_kernels(Xs[i], Xs[i], metric=kernels)
        YY = pairwise_kernels(Y, Y, metric=kernels)
        XY = pairwise_kernels(Xs[i], Y, metric=kernels)
        k_vals.append((XX, YY, XY))

    def mk_mmd_objective(alpha):
        # alpha: (K,)
        k_sum = sum(alpha[i] * k_vals[i][0] for i in range(K))
        k_sum_YY = sum(alpha[i] * k_vals[i][1] for i in range(K))
        k_sum_XY = sum(alpha[i] * k_vals[i][2] for i in range(K))
        mmd = k_sum.mean() + k_sum_YY.mean() - 2 * k_sum_XY.mean()
        
        # Entropy regularization
        epsilon = 1e-8
        entropy_reg = np.sum(alpha * np.log(alpha + epsilon))
        return mmd + lambda_entropy * entropy_reg

    return mk_mmd_objective


def solve_mk_weights(Xs, Y, kernel='rbf', lambda_entropy=0.01):

    K = len(Xs)
    min_n = min([x.shape[0] for x in Xs] + [Y.shape[0]])
    Xs = [x[:min_n] for x in Xs]
    Y = Y[:min_n]

    mk_mmd_obj = compute_mk_mmd(Xs, Y, kernels=kernel, lambda_entropy=lambda_entropy)

    cons = [
        {'type': 'eq', 'fun': lambda a: np.sum(a) - 1},
        {'type': 'ineq', 'fun': lambda a: a}  # a >= 0
    ]

    alpha0 = np.ones(K) / K
    res = minimize(mk_mmd_obj, alpha0, constraints=cons, method='SLSQP')
    return res.x  # optimal alpha*


def get_ensemble_soft_predictions(models, alphas, loader, device):
    # models: list of source models, alphas: weights
    all_soft_preds = []
    for model in models:
        model.eval()
        preds = []
        with torch.no_grad():
            for imgs, _ in tqdm(loader, desc="Soft Label:"):
                imgs = imgs.to(device)
                out = torch.sigmoid(model(imgs))
                preds.append(out.cpu())
        all_soft_preds.append(torch.cat(preds, dim=0))  # [N, 1, H, W]
    # Weighted average of all sources (broadcast along batch)
    all_soft_preds = torch.stack(all_soft_preds, dim=0)  # [K, N, 1, H, W]
    alphas = torch.tensor(alphas, dtype=all_soft_preds.dtype).view(-1, 1, 1, 1, 1)
    ensemble_soft = (all_soft_preds * alphas).sum(dim=0)  # [N, 1, H, W]
    return ensemble_soft



def kd_train(student_model, loader, pseudo_labels, optimizer, device, epochs=5, batch_size=8, alpha=1e-4):

    student_model.train()
    dataset = loader.dataset
    eps = 1e-4  

    for epoch in range(epochs):
        idxs = np.random.permutation(len(dataset))
        total_loss = 0

        for i in tqdm(range(0, len(dataset), batch_size), desc=f"KD-Train epoch {epoch+1}"):
            batch_idx = idxs[i:i+batch_size]
            imgs, soft_labels = [], []

            for j in batch_idx:
                img, _ = dataset[j]
                imgs.append(img)
                soft_labels.append(pseudo_labels[j])

            imgs = torch.stack(imgs).to(device)                     # [B, 1, H, W]
            soft_labels = torch.stack(soft_labels).to(device)      # [B, 1, H, W]
            soft_labels = torch.clamp(soft_labels, eps, 1 - eps)   

            optimizer.zero_grad()
            student_logits = student_model(imgs)            # raw logits
            student_probs = torch.sigmoid(student_logits)
            student_probs = torch.clamp(student_probs, eps, 1 - eps)

            # Losses
            loss_bce = F.binary_cross_entropy_with_logits(student_logits, soft_labels)
            loss_kl  = F.kl_div(student_probs.log(), soft_labels, reduction='batchmean')
            loss = alpha * loss_kl + (1 - alpha) * loss_bce

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}: KD Loss = {avg_loss:.4f}")


def get_batch_ensemble_predictions(models, alphas, batch_imgs, device):

    batch_imgs = batch_imgs.to(device)
    ensemble_pred = None
    
    for model, alpha in zip(models, alphas):
        model.eval()
        with torch.no_grad():
            pred = torch.sigmoid(model(batch_imgs))  # [B, 1, H, W]
            
            if ensemble_pred is None:
                ensemble_pred = alpha * pred
            else:
                ensemble_pred += alpha * pred
    
    return ensemble_pred


def kd_train_memory_efficient(student_model, models, alphas, loader, optimizer, device, 
                            epochs=5, alpha_loss=1e-4):

    student_model.train()
    
    # Set all teacher models to eval mode
    for model in models:
        model.eval()
    
    eps = 1e-6  # Small epsilon to avoid numerical issues
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(loader, desc=f"KD Epoch {epoch+1}/{epochs}")
        
        for batch_imgs, _ in pbar:
            batch_size = batch_imgs.size(0)
            
            # Generate soft labels for this batch
            soft_labels = get_batch_ensemble_predictions(models, alphas, batch_imgs, device)
            soft_labels = torch.clamp(soft_labels, eps, 1 - eps)
            batch_imgs = batch_imgs.to(device)
            
            # Forward pass through student
            optimizer.zero_grad()
            student_logits = student_model(batch_imgs)
            student_probs = torch.sigmoid(student_logits)
            student_probs = torch.clamp(student_probs, eps, 1 - eps)
            
            # Compute losses
            loss_bce = F.binary_cross_entropy_with_logits(student_logits, soft_labels)
            loss_kl = F.kl_div(student_probs.log(), soft_labels, reduction='batchmean')
            loss = alpha_loss * loss_kl + (1 - alpha_loss) * loss_bce
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item() * batch_size
            num_batches += 1
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(loader.dataset)
        print(f"Epoch {epoch+1}: Average KD Loss = {avg_loss:.4f}")


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

        # Load grayscale image and mask
        img = Image.open(os.path.join(self.raw_dir, fname)).convert("L")
        mask = Image.open(os.path.join(self.mask_dir, fname)).convert("L")

        # Crop patch
        img_patch = img.crop((x, y, x + self.patch_size, y + self.patch_size))
        mask_patch = mask.crop((x, y, x + self.patch_size, y + self.patch_size))

        # Apply transforms and ToTensor
        if self.transform:
            seed = torch.seed()
            torch.manual_seed(seed)
            img_patch = self.transform(img_patch)
            torch.manual_seed(seed)
            mask_patch = T.ToTensor()(mask_patch)
        else:
            img_patch = T.ToTensor()(img_patch)
            mask_patch = T.ToTensor()(mask_patch)

        # Binarize mask: ensure it's 0 or 1
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


# ---------- Laplace Uncertainty with Global Hessian ----------
def laplace_pixel_uncertainty(student_model, loader, device, global_hess_diag):
    student_model.eval()
    all_uncertainties = []

    w_var = 1.0 / global_hess_diag
    w_var = w_var.view(-1, 1)

    for img, _ in tqdm(loader, desc="Laplace Uncertainty"):
        img = img.to(device)
        with torch.no_grad():
            x = img
            skips = []
            for down in student_model.downs:
                x = down(x)
                skips.append(x)
                x = student_model.pool(x)
            x = student_model.bottleneck(x)

            i = 0
            while i < len(student_model.ups):
                x = student_model.ups[i](x)
                skip = skips[-(i // 2 + 1)]
                if x.shape != skip.shape:
                    x = T.CenterCrop([skip.shape[2], skip.shape[3]])(x)
                x = torch.cat([skip, x], dim=1)
                x = student_model.ups[i + 1](x)
                i += 2

            out = student_model.final_conv(x)
            probs = torch.sigmoid(out)

        z = x.detach().cpu().permute(0, 2, 3, 1).reshape(-1, x.shape[1])
        p = probs.detach().cpu().permute(0, 2, 3, 1).reshape(-1)
        var_logits = (z.pow(2) @ w_var).reshape(-1)
        entropy = - (p * torch.log(p + 1e-6) + (1 - p) * torch.log(1 - p + 1e-6))
        pixel_entropy = 0.5 * var_logits + entropy
        all_uncertainties.append(pixel_entropy)

    return torch.cat(all_uncertainties).numpy()


def precompute_laplace_hessian(student_model, loader, device):
    student_model.eval()
    z_squares_sum = None

    for img, _ in tqdm(loader, desc="Precomputing Laplace Hessian"):
        img = img.to(device)
        with torch.no_grad():
            x = img
            skips = []
            for down in student_model.downs:
                x = down(x)
                skips.append(x)
                x = student_model.pool(x)
            x = student_model.bottleneck(x)

            i = 0
            while i < len(student_model.ups):
                x = student_model.ups[i](x)
                skip = skips[-(i // 2 + 1)]
                if x.shape != skip.shape:
                    x = T.CenterCrop([skip.shape[2], skip.shape[3]])(x)
                x = torch.cat([skip, x], dim=1)
                x = student_model.ups[i + 1](x)
                i += 2

            out = student_model.final_conv(x)
            probs = torch.sigmoid(out)

        z = x.detach().cpu().permute(0, 2, 3, 1).reshape(-1, x.shape[1])  # [B*H*W, D]
        p = probs.detach().cpu().permute(0, 2, 3, 1).reshape(-1)         # [B*H*W]
        pp = p * (1 - p)

        if z_squares_sum is None:
            z_squares_sum = (pp.unsqueeze(1) * z.pow(2)).sum(dim=0)
        else:
            z_squares_sum += (pp.unsqueeze(1) * z.pow(2)).sum(dim=0)

    global_hess_diag = z_squares_sum + 1e-6
    return global_hess_diag


def combined_loss(pred, target):
    dice_loss = DiceLoss()
    focal_loss = FocalLoss(alpha=0.8, gamma=2)
    bce_loss = nn.BCEWithLogitsLoss()
    return dice_loss(pred, target) + focal_loss(pred, target) + bce_loss(pred, target)


# ---------- AL Train Loop ----------

def train_on_subset(model, subset, val_loader, optimizer, device, epochs, batch_size):
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=2)
    criterion = combined_loss
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, masks in tqdm(loader, desc=f"AL-train ep {epoch+1}/{epochs}", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
        print(f"AL Epoch {epoch+1}/{epochs} | Train Loss: {total_loss / len(loader.dataset):.4f}")



# ---------- Laplace Uncertainty with Global Hessian----------
def laplace_patch_uncertainty_scores(student_model, loader, device, global_hess_diag):

    student_model.eval()
    w_var = 1.0 / global_hess_diag
    w_var = w_var.view(-1, 1)
    
    patch_scores = []

    for img, _ in tqdm(loader, desc="Computing Patch Uncertainty Scores"):
        img = img.to(device)
        with torch.no_grad():
            x = img
            skips = []
            for down in student_model.downs:
                x = down(x)
                skips.append(x)
                x = student_model.pool(x)
            x = student_model.bottleneck(x)

            i = 0
            while i < len(student_model.ups):
                x = student_model.ups[i](x)
                skip = skips[-(i // 2 + 1)]
                if x.shape != skip.shape:
                    x = T.CenterCrop([skip.shape[2], skip.shape[3]])(x)
                x = torch.cat([skip, x], dim=1)
                x = student_model.ups[i + 1](x)
                i += 2

            out = student_model.final_conv(x)
            probs = torch.sigmoid(out)

        # Process each sample in the batch
        batch_size = img.shape[0]
        for b in range(batch_size):
            # Extract features and probabilities for this sample
            z_sample = x[b].detach().cpu().permute(1, 2, 0).reshape(-1, x.shape[1])  # [H*W, D]
            p_sample = probs[b].detach().cpu().permute(1, 2, 0).reshape(-1)         # [H*W]
            
            # Compute uncertainty for this sample
            var_logits = (z_sample.pow(2) @ w_var).reshape(-1)
            entropy = -(p_sample * torch.log(p_sample + 1e-6) + 
                       (1 - p_sample) * torch.log(1 - p_sample + 1e-6))
            pixel_entropy = 0.5 * var_logits + entropy
            
            # Store only the mean uncertainty score for this patch
            patch_scores.append(pixel_entropy.mean().item())
    
    return patch_scores


def active_learning_memory_efficient(student_model, target_train_ds, unlabeled_idxs, 
                                   global_hess_diag, device, K, batch_size=8):

    # Create subset and loader for unlabeled samples
    unlabeled_subset = Subset(target_train_ds, unlabeled_idxs)
    unl_loader = DataLoader(unlabeled_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Compute patch-level uncertainty scores 
    patch_scores = laplace_patch_uncertainty_scores(student_model, unl_loader, device, global_hess_diag)
    
    # Select top-K samples based on uncertainty scores
    patch_scores = np.array(patch_scores)
    topk_idx = np.argsort(-patch_scores)[:K]
    selected = [unlabeled_idxs[i] for i in topk_idx]
    
    return selected



# ---------- Main Script ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data', help='Dataset root')
    parser.add_argument('--source_domains', nargs='+', required=True, help='List of source domains')
    parser.add_argument('--source_ckpts', nargs='+', required=True, help='List of pretrained model checkpoints')
    parser.add_argument('--target_domain', required=True, help='Target/folder name')
    parser.add_argument('--out_path', required=True, help='Path to save the final model')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--active_iters', type=int, default=10)
    parser.add_argument('--annot_budget', type=int, default=100)
    parser.add_argument('--train_epochs_per_iter', type=int, default=5)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    log_filename = f"multi_{args.target_domain.replace('/', '-')}_budget{args.annot_budget}.txt"
    log_file = open(log_filename, "w")
    sys.stdout = log_file


    # Feature extraction for all source domains and target and obtaining optimal kernel weight for each source domain (α*) by minimizing MK-MMD
    # Equation 1, 2, 3, 4

    print("Extracting features for MK-MMD...")
    target_path = os.path.join(args.data_root, args.target_domain)
    target_val_ds = MembraneSegDataset(target_path, "val", patch_size=256, stride=128, transform=None)
    target_val_loader = DataLoader(target_val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    tgt_feats = []
    src_feats = []
    models = []
    for src_domain, ckpt in zip(args.source_domains, args.source_ckpts):
        src_path = os.path.join(args.data_root, src_domain)
        src_val_ds = MembraneSegDataset(src_path, "val", patch_size=256, stride=128, transform=None)
        src_val_loader = DataLoader(src_val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
        model = UNet(in_ch=1, out_ch=1).to(args.device)
        model.load_state_dict(torch.load(ckpt, map_location=args.device))
        models.append(model)
        src_feats.append(extract_features(model, src_val_loader, args.device))
    tgt_feats = extract_features(models[0], target_val_loader, args.device)
    print("Solving for optimal MK-MMD weights...")
    alphas = solve_mk_weights(src_feats, tgt_feats)  # optimal α*
    print(f"Optimal alpha*: {alphas}")



    # Student model training via Knowledge Distillation to mimic the ensemble predictions of the source models
    # Equation 5, 6

    target_train_ds = MembraneSegDataset(target_path, "train", patch_size=256, stride=128, transform=None)
    target_train_loader = DataLoader(target_train_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print("Training student model with memory-efficient KD...")
    student_model = UNet(in_ch=1, out_ch=1).to(args.device)
    optimizer = optim.Adam(student_model.parameters(), lr=1e-4)

    kd_train_memory_efficient(
        student_model=student_model,
        models=models,
        alphas=alphas,
        loader=target_train_loader,
        optimizer=optimizer,
        device=args.device,
        epochs=5,
        alpha_loss=1e-4
    )
    
    

    #  Active learning with analytic Laplace uncertainty
    # Equation 11, 12

    print("Starting active learning loop...")
    unlabeled_idxs = np.arange(len(target_train_ds)).tolist()
    labeled_idxs = []
    val_loader = DataLoader(target_val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    patch_counts = {
        'c-elegans-dauer-stage': 5641,
        'katz-lab-berghia-connective': 436,
        'octopus-vulgaris-vertical-lobe-sfltract': 3072,
        'dhanyasi-P14-mouse-cerebellum': 7688,
        'octopus-vulgaris-vertical-lobe-glia-deep-neuropil': 7514,
        'snemi': 3430,
        'micron': 42238,
        'fly': 35133,
        'whole-mouse-brain': 5075
    }
    
    target_total_patches = patch_counts[args.target_domain]
    absolute_budget = int((args.annot_budget / 100.0) * target_total_patches)
    K = absolute_budget // args.active_iters  # samples per active iteration


    print("Precomputing global Hessian from KD-trained model...")
    global_hess_diag = precompute_laplace_hessian(student_model, target_train_loader, args.device)

    for t in range(args.active_iters):
        print(f"Active learning iteration {t+1}/{args.active_iters}")
        
        # Memory-efficient active learning selection
        selected = active_learning_memory_efficient(
            student_model=student_model,
            target_train_ds=target_train_ds,
            unlabeled_idxs=unlabeled_idxs,
            global_hess_diag=global_hess_diag,
            device=args.device,
            K=K,
            batch_size=args.batch_size
        )

        print(f"Selected {len(selected)} samples for annotation.")
        # Update labeled/unlabeled pools
        labeled_idxs += selected
        unlabeled_idxs = [i for i in unlabeled_idxs if i not in selected]
        labeled_subset = Subset(target_train_ds, labeled_idxs)
        # Fine-tune for train_epochs_per_iter epochs on labeled set
        train_on_subset(student_model, labeled_subset, val_loader, optimizer, args.device, args.train_epochs_per_iter, args.batch_size)
        print(f"Total labeled: {len(labeled_idxs)}; Unlabeled: {len(unlabeled_idxs)}")

 
    torch.save(student_model.state_dict(), args.out_path)
    print(f"Final Multi-Domain model saved at {args.out_path}")



if __name__ == '__main__':
    main()



