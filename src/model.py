import torch
import torch.nn as nn
import numpy as np
import os
import sys
from pathlib import Path

# Advanced Pathing Protocol: 
# Dynamically appending the source root to ensure Sovereign Substrate resolution.
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

try:
    # Attempt absolute package import first
    from qau_qvs.core.qvs import QVS
    from qau_qvs.core.asc import ASC
except ImportError:
    # Fallback to explicit relative if strictly within a package context
    from .qau_qvs.core.qvs import QVS
    from .qau_qvs.core.asc import ASC

class ResBlock(nn.Module):
    """
    Paradox Residual Block: The topological anchor.
    """
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.conv(x))

class SemanticEncoder(nn.Module):
    """
    Paradox Semantic Encoder: Resolves images into Superpositions.
    """
    def __init__(self, latent_channels: int = 4):
        super(SemanticEncoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResBlock(32),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResBlock(64),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResBlock(128)
        )
        self.mu = nn.Conv2d(128, latent_channels, kernel_size=3, padding=1)
        self.logvar = nn.Conv2d(128, latent_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.layers(x)
        return self.mu(x), self.logvar(x)

class GenesisDecoder(nn.Module):
    """
    Paradox Genesis Decoder: The Collapse Mechanism.
    """
    def __init__(self, latent_channels: int = 4):
        super(GenesisDecoder, self).__init__()
        self.expand = nn.Sequential(
            nn.Conv2d(latent_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResBlock(128)
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResBlock(32)
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResBlock(32)
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
            nn.Conv2d(12, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.expand(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        return x

class LatentGenesisCore(nn.Module):
    """
    The Soul of Paradox: Quantum-Neural Genesis.
    Fuses Classical Silicon Learning with Quantum Virtual Substrate Logic.
    """
    def __init__(self, latent_channels: int = 4):
        super(LatentGenesisCore, self).__init__()
        self.encoder = SemanticEncoder(latent_channels)
        self.decoder = GenesisDecoder(latent_channels)
        self.qvs = QVS() # The Quantum Engine living inside the Neural Core

    def quantum_superposition(self, mu, logvar):
        """
        Active Quantum Integration:
        1. Encodes the 'Latent Energy' (mean) into the QVS.
        2. Performs a Superposition + Collapse cycle to determine Phase Bias.
        3. Modulates the neural manifold with the resulting Quantum Outcome.
        """
        batch_size = mu.shape[0]
        std = torch.exp(0.5 * logvar)
        
        if self.training:
            # We perform a Symbolic Quantum Measurement for each batch item
            # to determine the 'Phase Weave' of the entire manifold.
            phase_biases = []
            for i in range(batch_size):
                # Map the mean signal to a symbolic 4-state quantum basis
                # (Representing 4 quadrants of the complex Hilbert space)
                asc_id = self.qvs.create_asc(size=2)
                self.qvs.SUPERPOSE(asc_id, [(0,0), (0,1), (1,0), (1,1)])
                
                # Use the mean latent intensity to 'WEAVE' a phase shift
                intensity = torch.mean(mu[i]).item()
                self.qvs.WEAVE(asc_id, phase_angle=intensity * np.pi)
                
                # COLLAPSE the state to get a physical phase-outcome
                outcome = self.qvs.COLLAPSE(asc_id)
                
                # Map the binary outcome (e.g. (1,0)) back to a scalar bias
                bias = 1.0 if sum(outcome) % 2 == 0 else -1.0
                phase_biases.append(bias)
                
                # Cleanup quantum resources
                self.qvs.delete_asc(asc_id)
            
            bias_tensor = torch.tensor(phase_biases, device=mu.device).view(batch_size, 1, 1, 1)
            eps = torch.randn_like(std) * bias_tensor
        else:
            eps = torch.zeros_like(std)
            
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        
        # Transmuting Normal Latents into Quantum Superpositions
        z = self.quantum_superposition(mu, logvar)
        
        # Neural Bit-Depth Quantization (Information Pressure)
        # This simulates the transfer across the Aether Mesh
        z = z + (torch.round(z * 127.5) / 127.5 - z).detach()
        
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar
