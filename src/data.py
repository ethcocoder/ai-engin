import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=64, root='./data'):
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
                             shuffle=True, num_workers=2)

    # Download and load test data
    testset = torchvision.datasets.CIFAR10(root=root, train=False,
                                           download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size,
                            shuffle=False, num_workers=2)

    return trainloader, testloader
