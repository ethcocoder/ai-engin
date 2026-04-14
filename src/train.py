import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from data import get_dataloaders
from model import Autoencoder
from tqdm import tqdm

def train(args: argparse.Namespace):
    """
    Trains the Latent Communication Engine (Autoencoder).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Platform: Using device: {device}")
    print(f"[*] Hyperparameters: Batch Size={args.batch_size}, LR={args.lr}, Latent Dim={args.latent_dim}")

    trainloader, testloader = get_dataloaders(batch_size=args.batch_size, root='./data')

    model = Autoencoder(latent_dim=args.latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5) # Added weight decay

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    best_loss = float('inf')

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for data in pbar:
            images, _ = data
            images = images.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, images) 
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'batch_loss': f"{loss.item():.4f}"})
            
        epoch_loss = running_loss / len(trainloader)
        print(f"Epoch [{epoch+1}/{args.epochs}] Average Loss: {epoch_loss:.4f}")
        
        # Save checkpoints conditionally
        is_best = epoch_loss < best_loss
        if is_best:
            best_loss = epoch_loss
            print(f"[*] Checkpoint saved! New best loss: {best_loss:.4f}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, os.path.join(args.checkpoint_dir, 'best_autoencoder.pth'))
            
        # Regular save
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, os.path.join(args.checkpoint_dir, f'autoencoder_epoch_{epoch+1}.pth'))
            
    print("[*] Training routine completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AI Latent Communication Engine")
    parser.add_argument('--batch_size', type=int, default=64, help='Input batch size for training/testing')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--latent_dim', type=int, default=128, help='Size of the latent vector (compression level)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save model weights')
    
    args = parser.parse_args()
    train(args)
