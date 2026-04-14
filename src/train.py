import torch
import torch.nn as nn
import torch.optim as optim
from data import get_dataloaders
from model import Autoencoder
from tqdm import tqdm
import os

def train():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set parameters
    batch_size = 64
    epochs = 10
    learning_rate = 1e-3
    latent_dim = 128

    # Load data
    trainloader, testloader = get_dataloaders(batch_size=batch_size, root='./data')

    # Initialize model, loss and optimizer
    model = Autoencoder(latent_dim=latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Ensure checkpoints directory exists
    os.makedirs('checkpoints', exist_ok=True)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for data in pbar:
            images, _ = data
            images = images.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, images) # compare reconstructed to original
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        epoch_loss = running_loss / len(trainloader)
        print(f"Epoch [{epoch+1}/{epochs}] Average Loss: {epoch_loss:.4f}")
        
        # Save model checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, f'checkpoints/autoencoder_epoch_{epoch+1}.pth')
        
    print("Training finished.")

if __name__ == "__main__":
    train()
