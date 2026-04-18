"""
data.py — Paradox Genesis Core Data Pipeline
=============================================
Downloads and prepares the CIFAR-10 dataset for VAE training.

Train split augmentation (horizontal flip, colour jitter) improves the
decoder's ability to generalise structural patterns regardless of minor
photometric or geometric variation in real-world mobile imagery.
"""

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from typing import Tuple


def get_dataloaders(
    batch_size: int = 128,
    root: str = "./data",
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build and return CIFAR-10 train and test DataLoaders.

    Images are normalised to [-1, 1] (matching the encoder's Tanh output range).
    Train split applies light augmentation; test split is deterministic.

    Args:
        batch_size:  Number of images per mini-batch.
        root:        Directory where the dataset is downloaded/cached.
        num_workers: Parallel data-loading workers (set 0 for debugging).
        pin_memory:  Pin host memory for faster CPU→GPU transfers (True on CUDA).

    Returns:
        (trainloader, testloader): Both are PyTorch DataLoader instances.
    """
    # Training: augment to improve VAE generalisation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Validation / test: no augmentation — deterministic, reproducible
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=train_transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=test_transform
    )

    # persistent_workers=True avoids worker re-spawn overhead between epochs
    _persistent = num_workers > 0
    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=_persistent,
    )
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=_persistent,
    )

    return trainloader, testloader
