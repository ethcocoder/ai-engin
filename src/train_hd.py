import os
import torch
import torch.optim as optim
import sys
from pathlib import Path
from hd_data import get_hd_dataloaders
from model import LatentGenesisCore
from train import compression_loss
import argparse

# Advanced Pathing Protocol
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

def train_hd(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Telecom HD Engine: Initializing on {device}")
    
    loader = get_hd_dataloaders(image_dir=args.image_dir, batch_size=args.batch_size)
    if loader is None: return

    # Our NeuralCompressor is Fully Convolutional! It scales dynamically to HD!
    model = LatentGenesisCore(latent_channels=args.latent_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Fast Overfit loop to rapidly test HD capacity
    print("\n[*] Rapidly encoding your custom HD images to measure fidelity...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        for images, _ in loader:
            images = images.to(device)
            optimizer.zero_grad()
            
            outputs, mu, logvar = model(images)
            loss = compression_loss(outputs, images, mu, logvar)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{args.epochs}] -> HD Genesis Error: {(running_loss/len(loader)):.4f}")

    print("[*] HD Genesis Complete. Saving deployment architecture.")
    torch.save({'model_state_dict': model.state_dict()}, os.path.join(args.checkpoint_dir, 'hd_genesis_core.pth'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='hd_images')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100) # Fast epochs to lock in the HD fidelity over small data
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--latent_channels', type=int, default=4)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    args = parser.parse_args()
    train_hd(args)
