"""
train.py — Paradox Genesis Core Training Pipeline
==================================================
Trains the LatentGenesisCore VAE for neural image compression using:
    - L1 + SSIM perceptual loss  (sharp edges + structural fidelity)
    - KLD annealing warmup       (prevents posterior collapse)
    - AdamW + Cosine LR decay    (better generalisation)
    - Gradient clipping          (stable early training)

Usage:
    python src/train.py --epochs 80 --batch_size 128 --latent_channels 8
"""

import os
import argparse
import logging
from typing import Tuple, Optional, Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm

from data import get_dataloaders
from model import LatentGenesisCore

# ─── Perceptual Loss (VGG) ──────────────────────────────────────────────────
class PerceptualLoss(nn.Module):
    """
    Uses a pre-trained VGG16 to compare deep features of images.
    Forces the model to ensure 'textures' and 'meaningful features' match
    the original HD images, preventing the common 'plastic/blurry' look.
    """
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential(*vgg[:4])   # Relu1_2
        self.slice2 = nn.Sequential(*vgg[4:9])  # Relu2_2
        self.slice3 = nn.Sequential(*vgg[9:16]) # Relu3_3
        for param in self.parameters():
            param.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        # Normalize from [-1, 1] (Tanh) to ImageNet stats
        x = (x * 0.5 + 0.5 - self.mean) / self.std
        y = (y * 0.5 + 0.5 - self.mean) / self.std
        
        # Chain the features through the VGG slices
        x_f1, y_f1 = self.slice1(x), self.slice1(y)
        x_f2, y_f2 = self.slice2(x_f1), self.slice2(y_f1)
        x_f3, y_f3 = self.slice3(x_f2), self.slice3(y_f2)
        
        return F.mse_loss(x_f1, y_f1) + F.mse_loss(x_f2, y_f2) + F.mse_loss(x_f3, y_f3)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── SSIM Loss ────────────────────────────────────────────────────────────────

def _gaussian_window(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Builds a normalised 2-D Gaussian kernel for SSIM computation."""
    coords = torch.arange(size, dtype=torch.float) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    window = g.unsqueeze(1) * g.unsqueeze(0)
    return window.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)


def ssim_loss(x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """
    Computes 1 − SSIM (Structural Similarity Index), averaged over all channels.

    SSIM measures luminance, contrast, and structure simultaneously, making it
    a far better perceptual proxy than MSE alone.

    Args:
        x: Predicted images, shape (B, C, H, W), values in [-1, 1].
        y: Target images,    shape (B, C, H, W), values in [-1, 1].
        window_size: Size of the Gaussian kernel (must be odd).

    Returns:
        Scalar loss in [0, 2].
    """
    C_ch = x.shape[1]
    window = (
        _gaussian_window(window_size)
        .to(x.device)
        .expand(C_ch, 1, window_size, window_size)
        .contiguous()
    )
    pad = window_size // 2

    mu_x  = F.conv2d(x, window, padding=pad, groups=C_ch)
    mu_y  = F.conv2d(y, window, padding=pad, groups=C_ch)
    mu_xx = mu_x * mu_x
    mu_yy = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sig_xx = F.conv2d(x * x, window, padding=pad, groups=C_ch) - mu_xx
    sig_yy = F.conv2d(y * y, window, padding=pad, groups=C_ch) - mu_yy
    sig_xy = F.conv2d(x * y, window, padding=pad, groups=C_ch) - mu_xy

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu_xy + C1) * (2 * sig_xy + C2)) / (
        (mu_xx + mu_yy + C1) * (sig_xx + sig_yy + C2)
    )
    return 1.0 - ssim_map.mean()


# ── Loss Function ─────────────────────────────────────────────────────────────

def compression_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kld_weight: float = 0.01,
    perc_model: Optional[PerceptualLoss] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Paradox Generative Loss — fuses pixel fidelity, perceptual structure,
    conceptual texture matching, and information-theoretic entropy pressure.

    Components:
        L1    — per-pixel absolute error; sparse gradients preserve sharp edges.
        SSIM  — structural similarity; penalises blur and texture loss.
        PERC  — perceptual feature matching; ensures realistic high-res texture.
        KLD   — KL divergence; regularises the latent space toward N(0, I).
    """
    l1_l   = F.l1_loss(recon_x, x)
    ssim_l = ssim_loss(recon_x, x)
    kld_l  = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    perc_l = torch.tensor(0.0, device=x.device)
    if perc_model is not None:
        perc_l = perc_model(recon_x, x)

    # Balanced weighting: pixel + structure + texture + entropy
    total = l1_l + (0.5 * ssim_l) + (0.1 * perc_l) + (kld_weight * kld_l)
    return total, l1_l, ssim_l, perc_l, kld_l


# ── Training Loop ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    """
    Main training entry-point.

    Trains LatentGenesisCore for `args.epochs` epochs, saves the best
    checkpoint to `args.checkpoint_dir/best_genesis_core.pth`.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Paradox Genesis Core — initialising on %s", device)

    # cuDNN auto-tuner: ~15-20% speedup for fixed input sizes
    torch.backends.cudnn.benchmark = True

    trainloader, testloader = get_dataloaders(
        batch_size=args.batch_size,
        root="./data",
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    model = LatentGenesisCore(latent_channels=args.latent_channels).to(device)
    
    # Initialize Perceptual Model for HD-Ready training
    perc_model = PerceptualLoss().to(device)
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model parameters: %s", f"{param_count:,}")

    # AdamW: weight decay acts as L2 regularisation without polluting adaptive moments
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Cosine annealing: smooth, predictable LR decay → avoids plateau oscillation
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        # KLD annealing: ramp weight from 0 → kld_max over kld_warmup epochs.
        # Starting at 0 prevents posterior collapse before the reconstruction
        # loss has established a meaningful gradient signal.
        kld_weight = min(1.0, epoch / max(1, args.kld_warmup)) * args.kld_max

        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        pbar = tqdm(trainloader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False)

        for images, _ in pbar:
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)  # Slightly faster than zero_grad()

            outputs, mu, logvar = model(images)
            loss, l1_l, ssim_l, perc_l, kld_l = compression_loss(
                outputs, images, mu, logvar, kld_weight, perc_model
            )
            loss.backward()

            # Gradient clipping: prevents exploding gradients in early epochs
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                ssim=f"{ssim_l.item():.4f}",
                perc=f"{perc_l.item():.4f}",
                kld_w=f"{kld_weight:.3f}",
            )

        epoch_loss = running_loss / len(trainloader)
        scheduler.step()

        # ── Validate ─────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_images, _ in testloader:
                val_images = val_images.to(device, non_blocking=True)
                val_outputs, val_mu, val_logvar = model(val_images)
                v_loss, *_ = compression_loss(
                    val_outputs, val_images, val_mu, val_logvar, kld_weight, perc_model
                )
                val_loss += v_loss.item()

        epoch_val_loss = val_loss / len(testloader)
        current_lr = scheduler.get_last_lr()[0]

        log.info(
            "Epoch [%d/%d] → Train: %.4f | Val: %.4f | LR: %.6f",
            epoch + 1, args.epochs, epoch_loss, epoch_val_loss, current_lr,
        )

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            log.info("[*] Core upgraded! Best Genesis Loss: %.4f", best_val_loss)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_loss": best_val_loss,
                    "latent_channels": args.latent_channels,
                    "args": vars(args),
                },
                os.path.join(args.checkpoint_dir, "best_genesis_core.pth"),
            )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Paradox Genesis Core — VAE Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--batch_size",      type=int,   default=128, help="Training batch size")
    parser.add_argument("--epochs",          type=int,   default=80,  help="Total training epochs")
    parser.add_argument("--lr",              type=float, default=2e-3, help="AdamW base learning rate")
    parser.add_argument("--latent_channels", type=int,   default=8,   help="Bottleneck channel depth")
    parser.add_argument("--kld_warmup",      type=int,   default=15,  help="Epochs to ramp KLD weight")
    parser.add_argument("--kld_max",         type=float, default=0.005, help="Max KLD loss weight")
    parser.add_argument("--checkpoint_dir",  type=str,   default="checkpoints", help="Checkpoint directory")
    args = parser.parse_args()
    train(args)
