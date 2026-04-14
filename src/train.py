import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from data import get_dataloaders
from model import NeuralCompressor
from tqdm import tqdm

def compression_loss(recon_x, x):
    """
    Perceptual Sharpness Loss.
    Standard MSE guarantees a blurry output. By fusing L1 Loss (which optimizes median error) 
    with MSE, the Neural Compressor learns to preserve sharp lines, textures, and exact contrast!
    """
    l1_loss = nn.functional.l1_loss(recon_x, x)
    mse_loss = nn.functional.mse_loss(recon_x, x)
    return l1_loss + (mse_loss * 0.5)

def train(args: argparse.Namespace):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Telecom AI Core: Initializing on {device}")
    print(f"[*] Target Compression Channels: {args.latent_channels}")
    
    trainloader, testloader = get_dataloaders(batch_size=args.batch_size, root='./data')

    model = NeuralCompressor(latent_channels=args.latent_channels).to(device)
    # Removing weight decay natively so the model can dedicate 100% capacity to exact pixels
    optimizer = optim.Adam(model.parameters(), lr=args.lr) 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for data in pbar:
            images, _ = data
            images = images.to(device)
            optimizer.zero_grad()
            
            # Neural Compression Pipeline
            outputs, _ = model(images)
            loss = compression_loss(outputs, images)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'batch_loss': f"{loss.item():.4f}"})
            
        epoch_loss = running_loss / len(trainloader)
        
        # Validation Pipeline
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_data in testloader:
                val_images, _ = val_data
                val_images = val_images.to(device)
                
                val_outputs, _ = model(val_images)
                loss = compression_loss(val_outputs, val_images)
                val_loss += loss.item()
                
        epoch_val_loss = val_loss / len(testloader)
        
        print(f"Epoch [{epoch+1}/{args.epochs}] -> Train Layer: {epoch_loss:.4f} | Validation Accuracy: {epoch_val_loss:.4f}")
        scheduler.step(epoch_val_loss)
        
        is_best = epoch_val_loss < best_loss
        if is_best:
            best_loss = epoch_val_loss
            print(f"[*] Model upgraded! New best Telecom Loss: {best_loss:.4f}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.checkpoint_dir, 'best_compressor.pth'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=2e-3) # Higher learning rate for L1 convergence
    parser.add_argument('--latent_channels', type=int, default=4, help='Size of spatial compressed payload')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    args = parser.parse_args()
    train(args)
