import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from data import get_dataloaders
from model import LatentGenesisCore
from tqdm import tqdm
import torch.nn.functional as F

# ─── SSIM Loss ───────────────────────────────────────────────────────────────
def gaussian_window(size=11, sigma=1.5):
    coords = torch.arange(size, dtype=torch.float) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    window = g.unsqueeze(1) * g.unsqueeze(0)
    return window.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

def ssim_loss(x, y, window_size=11):
    """Structural Similarity loss (1 - SSIM), averaged over channels."""
    C = x.shape[1]
    window = gaussian_window(window_size).to(x.device).expand(C, 1, window_size, window_size)
    mu_x   = F.conv2d(x, window, padding=window_size//2, groups=C)
    mu_y   = F.conv2d(y, window, padding=window_size//2, groups=C)
    mu_xx  = mu_x * mu_x
    mu_yy  = mu_y * mu_y
    mu_xy  = mu_x * mu_y
    sig_xx = F.conv2d(x * x, window, padding=window_size//2, groups=C) - mu_xx
    sig_yy = F.conv2d(y * y, window, padding=window_size//2, groups=C) - mu_yy
    sig_xy = F.conv2d(x * y, window, padding=window_size//2, groups=C) - mu_xy
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim  = ((2 * mu_xy + C1) * (2 * sig_xy + C2)) / \
            ((mu_xx + mu_yy + C1) * (sig_xx + sig_yy + C2))
    return 1.0 - ssim.mean()

# ─── Loss Function ───────────────────────────────────────────────────────────
def compression_loss(recon_x, x, mu, logvar, kld_weight: float = 0.01):
    """
    Paradox Generative Loss — Upgraded with SSIM perceptual term.
    L1  : per-pixel fidelity (sparse gradient → sharp edges)
    SSIM: structural / perceptual fidelity (keeps textures intact)
    KLD : information pressure (compact latent manifold)
    """
    l1_loss   = F.l1_loss(recon_x, x)
    ssim_l    = ssim_loss(recon_x, x)
    kld_loss  = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # Balanced weighting: perception + structure + entropy pressure
    return l1_loss + (0.5 * ssim_l) + (kld_weight * kld_loss)

# ─── Training Loop ───────────────────────────────────────────────────────────
def train(args: argparse.Namespace):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Paradox Genesis Core: Initializing on {device}")

    # Enable cuDNN auto-tuner for fixed-size input → ~20% speedup
    torch.backends.cudnn.benchmark = True

    trainloader, testloader = get_dataloaders(
        batch_size=args.batch_size,
        root='./data',
        num_workers=4,           # More workers for GPU feeding
        pin_memory=(device.type == 'cuda')
    )

    model = LatentGenesisCore(latent_channels=args.latent_channels).to(device)

    # AdamW → weight decay regularization (better generalization than Adam)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Cosine annealing: smooth LR decay → avoids plateau oscillation
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_loss = float('inf')

    for epoch in range(args.epochs):
        # KLD annealing: start at 0 → full weight by epoch kld_warmup
        # Prevents "posterior collapse" in early training
        kld_weight = min(1.0, epoch / max(1, args.kld_warmup)) * args.kld_max

        model.train()
        running_loss = 0.0
        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for images, _ in pbar:
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad()

            outputs, mu, logvar = model(images)
            loss = compression_loss(outputs, images, mu, logvar, kld_weight)

            loss.backward()

            # Gradient clipping → prevents exploding gradients during early epochs
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'batch_loss': f"{loss.item():.4f}", 'kld_w': f"{kld_weight:.3f}"})

        epoch_loss = running_loss / len(trainloader)
        scheduler.step()

        # ── Validation ──────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_images, _ in testloader:
                val_images = val_images.to(device, non_blocking=True)
                val_outputs, mu, logvar = model(val_images)
                val_loss += compression_loss(val_outputs, val_images, mu, logvar, kld_weight).item()

        epoch_val_loss = val_loss / len(testloader)
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch [{epoch+1}/{args.epochs}] -> Genesis Loss: {epoch_loss:.4f} | "
              f"Validation Fidelity: {epoch_val_loss:.4f} | LR: {current_lr:.6f}")

        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            print(f"[*] Core upgraded! New best Paradox Genesis Loss: {best_loss:.4f}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'latent_channels': args.latent_channels,
            }, os.path.join(args.checkpoint_dir, 'best_genesis_core.pth'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paradox Genesis Core Training")
    parser.add_argument('--batch_size',     type=int,   default=128)
    parser.add_argument('--epochs',         type=int,   default=50,   help='More epochs → sharper reconstructions')
    parser.add_argument('--lr',             type=float, default=1e-3, help='AdamW base learning rate')
    parser.add_argument('--latent_channels',type=int,   default=4,    help='Spatial channels in bottleneck')
    parser.add_argument('--kld_warmup',     type=int,   default=10,   help='Epochs to ramp KLD from 0 to max')
    parser.add_argument('--kld_max',        type=float, default=0.01, help='Maximum KLD loss weight')
    parser.add_argument('--checkpoint_dir', type=str,   default='checkpoints')
    args = parser.parse_args()
    train(args)
