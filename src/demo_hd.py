"""
demo_hd.py — Paradox Aether Mesh HD Visualization
==================================================
Demonstrates High-Definition (256x256+) compression on real internet images.
Uses the 4-stage Genesis Core to show structural fidelity at high scales.
"""

import os
import torch
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from model import LatentGenesisCore
from hd_data import get_hd_dataloaders
import argparse

# Advanced Pathing Protocol
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

def unnorm(img):
    return torch.clamp(img * 0.5 + 0.5, 0, 1)

def psnr(original, reconstructed):
    mse = torch.mean((original - reconstructed) ** 2).item()
    if mse == 0: return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(torch.tensor(mse))).item()

def run_hd_simulation(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Paradox HD Engine: Initializing on {device}")
    
    # Load Model
    model = LatentGenesisCore(latent_channels=args.latent_channels).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load HD Data (Automatically pulls from web if empty)
    loader = get_hd_dataloaders(image_dir=args.image_dir, batch_size=4)
    if loader is None: return
    images, _ = next(iter(loader))
    images = images.to(device)

    # Calculate Compression Math
    # Original: 256x256x3 (1 byte per channel)
    original_bytes = images[0].shape[1] * images[0].shape[2] * images[0].shape[0]
    
    encoded_latents = []
    with torch.no_grad():
        for i in range(len(images)):
            # Use real quantization logic
            mu, _ = model.encoder(images[i].unsqueeze(0))
            mu_q = torch.round(torch.clamp(mu, -1, 1) * 127.5) / 127.5
            encoded_latents.append(mu_q)
            
    # Payload: 16x16 x latent_channels (1 byte per value)
    payload_bytes = encoded_latents[0].nelement() * 1
    compression_ratio = original_bytes / payload_bytes
    
    print("\n--- HD TRANSMISSION METRICS ---")
    print(f"[-] Image Resolution:  {images[0].shape[1]}x{images[0].shape[2]}")
    print(f"[-] Original Payload:  {original_bytes:,} bytes")
    print(f"[-] Compressed Payload: {payload_bytes:,} bytes")
    print(f"[-] Reduction Factor:   {compression_ratio:.1f}X")
    print("--------------------------------\n")

    decoded_images = []
    psnr_scores = []
    with torch.no_grad():
        for i in range(len(images)):
            recon = model.decoder(encoded_latents[i]).cpu().squeeze(0)
            decoded_images.append(recon)
            psnr_scores.append(psnr(unnorm(images[i].cpu()), unnorm(recon)))

    avg_psnr = sum(psnr_scores) / len(psnr_scores)
    print(f"[*] Average HD PSNR: {avg_psnr:.2f} dB")

    fig, axes = plt.subplots(2, len(images), figsize=(16, 8))
    fig.suptitle(f"Sovereign HD Engine: {compression_ratio:.1f}x Bandwidth reduction | Avg PSNR: {avg_psnr:.2f}dB", fontsize=18)

    for i in range(len(images)):
        ax_top = axes[0, i]
        ax_bot = axes[1, i]
            
        ax_top.imshow(unnorm(images[i].cpu()).permute(1, 2, 0).numpy())
        ax_top.set_title(f"Sender HD Original")
        ax_top.axis('off')

        ax_bot.imshow(unnorm(decoded_images[i]).permute(1, 2, 0).numpy())
        ax_bot.set_title(f"Receiver HD Final\nPSNR: {psnr_scores[i]:.1f}dB")
        ax_bot.axis('off')

    plt.tight_layout()
    plt.savefig('hd_telecom_result.png', dpi=300)
    print(f"\n[*] Rendered output to 'hd_telecom_result.png'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--image_dir', type=str, default='hd_images')
    parser.add_argument('--latent_channels', type=int, default=16)
    args = parser.parse_args()
    run_hd_simulation(args)
