import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from typing import Tuple

def get_dataloaders(
    batch_size: int = 64,
    root: str = './data',
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Downloads CIFAR10, applies train-time augmentation, and returns dataloaders.
    
    Augmentation improves generalization and results in sharper decoded images
    because the model learns invariance to horizontal flips and slight color shifts.
    """
    # Training: augment to improve VAE generalization
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Validation / test: no augmentation (deterministic)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=train_transform
    )
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers > 0)
    )

    testset = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=test_transform
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers > 0)
    )

    return trainloader, testloader
