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

def run_hd_simulation(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LatentGenesisCore(latent_channels=args.latent_channels).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    loader = get_hd_dataloaders(image_dir=args.image_dir, batch_size=4)
    if loader is None: return
    images, _ = next(iter(loader))
    images = images.to(device)

    # 1080p -> Compressed dimensions
    original_bytes = images[0].element_size() * images[0].nelement()
    
    encoded_latents = []
    with torch.no_grad():
        for i in range(len(images)):
            encoded_latents.append(model.encoder(images[i].unsqueeze(0))[0])
            
    payload_bytes = encoded_latents[0].element_size() * encoded_latents[0].nelement()
    compression_ratio = original_bytes / payload_bytes
    
    print("\n--- RESULTS ---")
    print(f"Original Sender Bytes: {original_bytes:,} bytes")
    print(f"Transmitted V-Payload: {payload_bytes:,} bytes")
    print(f"Total Bandwidth Saving: {compression_ratio:.1f}X REDUCTION!")
    print("-----------------\n")

    decoded_images = []
    with torch.no_grad():
        for i in range(len(images)):
            decoded_images.append(model.decoder(encoded_latents[i]).cpu().squeeze(0))

    fig, axes = plt.subplots(2, len(images), figsize=(16, 8))
    fig.suptitle(f"Real HD Telecom Engine: {compression_ratio:.1f}x Bandwidth Reduction", fontsize=18)

    for i in range(len(images)):
        if len(images) > 1:
            ax_top = axes[0, i]
            ax_bot = axes[1, i]
        else:
            ax_top = axes[0]
            ax_bot = axes[1]
            
        ax_top.imshow(unnorm(images[i].cpu()).permute(1, 2, 0).numpy())
        ax_top.set_title(f"Sender HD")
        ax_top.axis('off')

        ax_bot.imshow(unnorm(decoded_images[i]).permute(1, 2, 0).numpy())
        ax_bot.set_title(f"Receiver HD")
        ax_bot.axis('off')

    plt.tight_layout()
    plt.savefig('hd_telecom_result.png', dpi=400)
    print("[*] Rendered output to hd_telecom_result.png. Open it to see HD fidelity!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--image_dir', type=str, default='hd_images')
    parser.add_argument('--latent_channels', type=int, default=4)
    args = parser.parse_args()
    run_hd_simulation(args)
