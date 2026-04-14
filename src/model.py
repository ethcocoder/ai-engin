import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    CNN-based Encoder that maps input images into a compact latent vector.
    """
    def __init__(self, latent_dim: int = 128):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Input size: 3 x 32 x 32 (CIFAR-10)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), # 32 x 16 x 16
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 64 x 8 x 8
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 128 x 4 x 4
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 4 * 4, latent_dim)
        
        self._init_weights()

    def _init_weights(self):
        """Applies Kaiming Initialization to CNN layers for optimal ReLU routing."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.flatten(x)
        latent_vector = self.fc(x)
        return latent_vector


class Decoder(nn.Module):
    """
    CNN-based Decoder that reconstructs the image from a latent vector.
    """
    def __init__(self, latent_dim: int = 128):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.unflatten = nn.Unflatten(1, (128, 4, 4))
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # 64 x 8 x 8
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # 32 x 16 x 16
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1), # 3 x 32 x 32
            nn.Tanh() # Output in range [-1, 1] to match the normalization
        )
        
        self._init_weights()

    def _init_weights(self):
        """Applies Kaiming Initialization for intermediate layers."""
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.unflatten(x)
        reconstructed_image = self.deconv(x)
        return reconstructed_image


class Autoencoder(nn.Module):
    """
    Full AI Engine Autoencoder wrapper combining Encoder and Decoder.
    """
    def __init__(self, latent_dim: int = 128):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
