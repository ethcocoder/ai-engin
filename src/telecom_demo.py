import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
from model import LatentGenesisCore
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
import argparse

# Advanced Pathing Protocol
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

def unnorm(img):
    """Denormalize from [-1,1] to [0,1]."""
    img = img * 0.5 + 0.5
    return torch.clamp(img, 0, 1)

def psnr(original, reconstructed):
    """Peak Signal-to-Noise Ratio in dB. Higher = better."""
    mse = torch.mean((original - reconstructed) ** 2).item()
    if mse == 0:
        return float('inf')
    return 10 * (torch.log10(torch.tensor(1.0)) - torch.log10(torch.tensor(mse))).item()

def run_bandwidth_simulation(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Initializing Telecom Neural Compression Simulator on {device}\n")

    # 1. Initialize Paradox Genesis Layer
    model = LatentGenesisCore(latent_channels=args.latent_channels).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. Get sample data
    # CRITICAL FIX: Do NOT resize to 256 here.
    # The model was trained on native 32x32 CIFAR images.
    # Resizing to 256 at inference causes catastrophic distribution shift.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    images, _ = next(iter(loader))
    images = images.to(device)

    print("[USER A: SENDER MOBILE DEVICE]")
    original_bytes = images[0].element_size() * images[0].nelement()
    print(f"[-] Image Original Dimensions: {list(images[0].shape)}")
    print(f"[-] Bytes sent typically (Uncompressed): {original_bytes:,} bytes")

    # Encode: compress locally on Sender's device
    encoded_latents = []
    with torch.no_grad():
        for i in range(4):
            mu, _ = model.encoder(images[i].unsqueeze(0))
            # Clamp before quantization (matches upgraded model.py)
            mu_clamped = torch.clamp(mu, -1.0, 1.0)
            q_latent = torch.round(mu_clamped * 127.5) / 127.5
            encoded_latents.append(q_latent)

    print("\n[TELECOM GATEWAY: TRANSMISSION]")
    payload_bytes = encoded_latents[0].nelement() * 1   # 8-bit = 1 byte per element
    compression_ratio = original_bytes / payload_bytes
    latent_shape = list(encoded_latents[0].squeeze(0).shape)
    print(f"[-] Transmitted Neural Payload Dimensions: {latent_shape}")
    print(f"[-] Server Bandwidth Usage (Compressed & Quantized): {payload_bytes:,} bytes")
    print(f"[-] *** PROFIT ACHIEVED: {compression_ratio:.1f}X REDUCTION IN BANDWIDTH COSTS! ***")

    print("\n[USER B: RECEIVER MOBILE DEVICE]")
    print("[-] Collapsing & Decoding Payload locally...")
    decoded_images = []
    psnr_scores = []
    with torch.no_grad():
        for i in range(4):
            reconstructed = model.decoder(encoded_latents[i]).cpu().squeeze(0)
            decoded_images.append(reconstructed)
            # Compute PSNR in [0,1] space
            orig_01 = unnorm(images[i].cpu())
            recon_01 = unnorm(reconstructed)
            psnr_scores.append(psnr(orig_01, recon_01))

    avg_psnr = sum(psnr_scores) / len(psnr_scores)
    print(f"[-] Average PSNR: {avg_psnr:.2f} dB  (>25 dB = good quality)")

    # 3. Visualization — display at 4x native size for clarity
    scale = 4
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(
        f"Future Telecom Network  |  {compression_ratio:.1f}x Compression  |  "
        f"Avg PSNR: {avg_psnr:.2f} dB",
        fontsize=14, fontweight='bold'
    )

    interp = 'nearest'   # crisp pixels — no blur from bilinear upscaling
    for i in range(4):
        orig_img  = unnorm(images[i].cpu()).permute(1, 2, 0).numpy()
        recon_img = unnorm(decoded_images[i]).permute(1, 2, 0).numpy()

        axes[0, i].imshow(orig_img, interpolation=interp)
        axes[0, i].set_title(f"Sender Original\n[{original_bytes:,} bytes]", fontsize=9)
        axes[0, i].axis('off')

        axes[1, i].imshow(recon_img, interpolation=interp)
        axes[1, i].set_title(
            f"Receiver Final\n({payload_bytes:,} bytes | PSNR: {psnr_scores[i]:.1f} dB)",
            fontsize=9
        )
        axes[1, i].axis('off')

    plt.tight_layout()
    file_out = 'telecom_simulation_result.png'
    plt.savefig(file_out, dpi=150, bbox_inches='tight')
    print(f"\n[*] SUCCESS: View the transmission result in '{file_out}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Telecom AI Bandwidth Simulator")
    parser.add_argument('--model_path',      type=str, required=True, help="Path to best_genesis_core.pth")
    parser.add_argument('--latent_channels', type=int, default=4,     help="Must match training config")
    args = parser.parse_args()

    run_bandwidth_simulation(args)
