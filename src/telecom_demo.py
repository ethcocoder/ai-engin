import torch
import matplotlib.pyplot as plt
from model import NeuralCompressor
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
import argparse
import sys

def unnorm(img):
    img = img * 0.5 + 0.5
    return torch.clamp(img, 0, 1)

def run_bandwidth_simulation(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Initializing Telecom Neural Compression Simulator on {device}\n")

    # 1. Initialize High-Fidelity Telecom Layer
    model = NeuralCompressor(latent_channels=args.latent_channels).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. Get sample data (simulating User A's photo gallery)
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
    print(f"[-] Bytes sent typically (Uncompressed): {original_bytes} bytes")

    # Encode (Compressing locally on Mobile A)
    encoded_latents = []
    with torch.no_grad():
        for i in range(4):
            latent_map = model.encoder(images[i].unsqueeze(0))
            encoded_latents.append(latent_map)
            
    print("\n[TELECOM GATEWAY: TRANSMISSION]")
    payload_bytes = encoded_latents[0].element_size() * encoded_latents[0].nelement()
    compression_ratio = original_bytes / payload_bytes
    print(f"[-] Transmitted Neural Payload Dimensions: {list(encoded_latents[0].squeeze(0).shape)}")
    print(f"[-] Server Bandwidth Usage (Compressed): {payload_bytes} bytes")
    print(f"[-] *** PROFIT ACHIEVED: {compression_ratio:.1f}X REDUCTION IN BANDWIDTH COSTS! ***")

    print("\n[USER B: RECEIVER MOBILE DEVICE]")
    print("[-] Decoding Payload locally...")
    decoded_images = []
    with torch.no_grad():
        for i in range(4):
            reconstructed = model.decoder(encoded_latents[i]).cpu().squeeze(0)
            decoded_images.append(reconstructed)

    # 3. Visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"Future Telecom Network (Compression Factor: {compression_ratio:.1f}x reduction)", fontsize=18)

    for i in range(4):
        # Top Row: SENDER
        axes[0, i].imshow(unnorm(images[i].cpu()).permute(1, 2, 0).numpy())
        axes[0, i].set_title(f"Sender Original\n[{original_bytes} bytes]")
        axes[0, i].axis('off')

        # Bottom Row: RECEIVER
        axes[1, i].imshow(unnorm(decoded_images[i]).permute(1, 2, 0).numpy())
        axes[1, i].set_title(f"Receiver Final\n(Decoded from {payload_bytes} bytes)")
        axes[1, i].axis('off')

    plt.tight_layout()
    file_out = 'telecom_simulation_result.png'
    plt.savefig(file_out, dpi=300)
    print(f"\n[*] SUCCESS: View the transmission result in '{file_out}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Telecom AI Bandwidth Simulator")
    parser.add_argument('--model_path', type=str, required=True, help="Path to best_compressor.pth weights")
    parser.add_argument('--latent_channels', type=int, default=4, help="Number of spatial channels in bottlenecks")
    args = parser.parse_args()
    
    run_bandwidth_simulation(args)
