import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from typing import Tuple

def get_dataloaders(batch_size: int = 64, root: str = './data', num_workers: int = 2) -> Tuple[DataLoader, DataLoader]:
    """
    Downloads CIFAR10 dataset, applies preprocessing, 
    and returns train and test dataloaders.
    """
    # Preprocess images
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Normalize to [-1, 1] for better autoencoder training
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Download and load training data
    trainset = torchvision.datasets.CIFAR10(root=root, train=True,
                                            download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers)

    # Download and load test data
    testset = torchvision.datasets.CIFAR10(root=root, train=False,
                                           download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)

    return trainloader, testloader
