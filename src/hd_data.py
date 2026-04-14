import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class CustomHDDataset(Dataset):
    """Loads any real HD images uploaded by the user."""
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, 0 # Dummy label since we don't need classification

def get_hd_dataloaders(image_dir='./hd_images', batch_size=4):
    """
    Creates a dataloader for HD 256x256 images.
    """
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
        print(f"[!] Please upload some .jpg images into the '{image_dir}' folder!")
        return None

    # We resize to a standard crisp multiple of 8 for the ResNet architecture
    transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = CustomHDDataset(image_dir, transform=transform)
    
    if len(dataset) == 0:
        print(f"[!] No images found in '{image_dir}'. Please upload some!")
        return None

    # For this testing scenario, train & test are the same to verify exact capacity
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
