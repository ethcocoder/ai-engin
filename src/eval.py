import torch
import torchvision
import matplotlib.pyplot as plt
from data import get_dataloaders
from model import Autoencoder
import argparse
import os

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Evaluating on device: {device}")
    
    # Load test data
    _, testloader = get_dataloaders(batch_size=args.num_images, root='./data')
    dataiter = iter(testloader)
    images, _ = next(dataiter)
    images = images.to(device)

    # Load Model
    model = Autoencoder(latent_dim=args.latent_dim).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Reconstruct
    with torch.no_grad():
        reconstructed = model(images)
    
    images = images.cpu()
    reconstructed = reconstructed.cpu()

    # Visualization code
    fig, axes = plt.subplots(nrows=2, ncols=args.num_images, figsize=(15, 4))
    
    # Unnormalize for visualization
    def unnorm(img):
        img = img * 0.5 + 0.5
        return torch.clamp(img, 0, 1)

    for i in range(args.num_images):
        axes[0, i].imshow(unnorm(images[i]).permute(1, 2, 0).numpy())
        axes[0, i].axis('off')
        if i == 0: axes[0, i].set_title("Original Data")

        axes[1, i].imshow(unnorm(reconstructed[i]).permute(1, 2, 0).numpy())
        axes[1, i].axis('off')
        if i == 0: axes[1, i].set_title("Reconstructed from Latent Code")

    out_file = "reconstruction_comparison.png"
    plt.savefig(out_file)
    print(f"[*] Evaluation completed. Image saved as: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate visual quality of the Ai Engine")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the Autoencoder checkpoint')
    parser.add_argument('--latent_dim', type=int, default=128, help='Size of the latent vector')
    parser.add_argument('--num_images', type=int, default=8, help='Number of test images to visualize')
    
    args = parser.parse_args()
    evaluate(args)
