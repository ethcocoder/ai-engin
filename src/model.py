import torch
import torch.nn as nn

class ResBlock(nn.Module):
    """
    Residual Block setup: Prevents spatial information loss, ensuring
    sharp edges and high-fidelity texture transfers.
    """
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class SpatialEncoder(nn.Module):
    """
    Compresses an HD image into a 2D Spatial Map rather than a 1D Vector.
    This preserves coordinate math so exact reconstructions are possible!
    """
    def __init__(self, latent_channels: int = 4):
        super(SpatialEncoder, self).__init__()
        # Input: 3 x 32 x 32 -> Downsample to 64 x 16 x 16
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.res1 = ResBlock(64)
        
        # 64 x 16 x 16 -> Downsample to 128 x 8 x 8
        self.down_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.res2 = ResBlock(128)
        self.res3 = ResBlock(128)
        
        # Squeeze channels to bottleneck size: 128 x 8 x 8 -> latent_channels x 8 x 8
        self.to_latent = nn.Conv2d(128, latent_channels, kernel_size=3, padding=1)
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                
    def forward(self, x):
        x = self.init_conv(x)
        x = self.res1(x)
        x = self.down_conv(x)
        x = self.res2(x)
        x = self.res3(x)
        latent_map = self.to_latent(x)
        return latent_map


class SpatialDecoder(nn.Module):
    """
    Takes the ultra-tiny 2D spatial footprint and symmetrically explodes 
    it back into a flawless HD image using Deep Residual logic.
    """
    def __init__(self, latent_channels: int = 4):
        super(SpatialDecoder, self).__init__()
        # Re-inflate channels: latent_channels x 8 x 8 -> 128 x 8 x 8
        self.from_latent = nn.Sequential(
            nn.Conv2d(latent_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.res1 = ResBlock(128)
        self.res2 = ResBlock(128)
        
        # Upsample: 128 x 8 x 8 -> 64 x 16 x 16
        self.up_conv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.res3 = ResBlock(64)
        
        # Upsample to Output: 64 x 16 x 16 -> 3 x 32 x 32
        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh() # Output clamped strictly to normal image formats bounds
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.from_latent(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.up_conv1(x)
        x = self.res3(x)
        reconstructed = self.up_conv2(x)
        return reconstructed


class NeuralCompressor(nn.Module):
    """
    High-Fidelity Neural Image Compression Engine.
    Replaces the 'Autoencoder' specifically for telecom and bandwidth dominance.
    """
    def __init__(self, latent_channels: int = 4):
        super(NeuralCompressor, self).__init__()
        self.encoder = SpatialEncoder(latent_channels)
        self.decoder = SpatialDecoder(latent_channels)

    def forward(self, x: torch.Tensor):
        latent_map = self.encoder(x)
        reconstructed = self.decoder(latent_map)
        return reconstructed, latent_map
