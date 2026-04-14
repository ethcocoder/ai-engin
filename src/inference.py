import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from model import Autoencoder
import os
import argparse

def load_image(image_path: str) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((32, 32)), # Resize to match CIFAR-10 requirements
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0)

def save_image(tensor: torch.Tensor, output_path: str):
    tensor = tensor.squeeze(0).detach().cpu()
    tensor = tensor * 0.5 + 0.5 # Unnormalize
    tensor = torch.clamp(tensor, 0, 1)
    img = transforms.ToPILImage()(tensor)
    img.save(output_path)

def encode_data(args: argparse.Namespace):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Autoencoder(latent_dim=args.latent_dim).to(device)
    
    model.load_state_dict(torch.load(args.model_path, map_location=device)['model_state_dict'])
    model.eval()

    img_tensor = load_image(args.input_image).to(device)
    with torch.no_grad():
        latent = model.encoder(img_tensor)
    
    # Save the compressed space array to a file to simulate transmission
    latent_np = latent.cpu().numpy()
    np.save(args.output_latent, latent_np)
    
    original_size = os.path.getsize(args.input_image)
    latent_size = os.path.getsize(args.output_latent + '.npy')
    
    print(f"[*] Data Encoded: {args.input_image}")
    print(f"[-] Original File Size: {original_size / 1024:.2f} KB")
    print(f"[-] Latent Vector Size: {latent_size / 1024:.2f} KB")
    print(f"[*] Compression Ratio: {original_size/latent_size:.2f}x")
    print(f"[>] Send '{args.output_latent}.npy' over the network instead of the original image.")

def decode_data(args: argparse.Namespace):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Autoencoder(latent_dim=args.latent_dim).to(device)
    
    model.load_state_dict(torch.load(args.model_path, map_location=device)['model_state_dict'])
    model.eval()

    # Provide the path precisely with .npy extension
    latent_file = args.input_latent if args.input_latent.endswith('.npy') else args.input_latent + '.npy'
    latent_np = np.load(latent_file)
    latent_tensor = torch.tensor(latent_np).to(device)

    with torch.no_grad():
        reconstructed = model.decoder(latent_tensor)

    save_image(reconstructed, args.output_image)
    print(f"[*] Data received and decoded successfully.")
    print(f"[-] Reconstructed image saved to: {args.output_image}")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent Communication Transmission Simulator")
    subparsers = parser.add_subparsers(dest='mode', required=True)

    # Encode setup
    parser_enc = subparsers.add_parser('encode', help="Encode image into a latent file for transmission")
    parser_enc.add_argument('--input_image', type=str, required=True, help="Original image file")
    parser_enc.add_argument('--output_latent', type=str, required=True, help="Latent output file name")
    parser_enc.add_argument('--model_path', type=str, required=True, help="Path to best_autoencoder.pth")
    parser_enc.add_argument('--latent_dim', type=int, default=128, help="Size of the latent vector")

    # Decode setup
    parser_dec = subparsers.add_parser('decode', help="Decode latent file back into an image")
    parser_dec.add_argument('--input_latent', type=str, required=True, help="Received latent file (.npy)")
    parser_dec.add_argument('--output_image', type=str, required=True, help="Output reconstructed image file")
    parser_dec.add_argument('--model_path', type=str, required=True, help="Path to best_autoencoder.pth")
    parser_dec.add_argument('--latent_dim', type=int, default=128, help="Size of the latent vector")

    args = parser.parse_args()
    
    if args.mode == 'encode':
        encode_data(args)
    elif args.mode == 'decode':
        decode_data(args)
