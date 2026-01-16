import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """
    Kelas blok konvolusi
    Adaptif bisa untuk downsampling atau upsampling
    Adaptif untuk ReLU atau pakai Identity
    """
    def __init__(self, in_channels, out_channels, down = True, use_idty = True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode='reflect', **kwargs)
            if down else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_idty else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)
    

class ResidualBlock(nn.Module):
    """
    Kelas blok residual untuk jaringan dalam.
    Tidak merubah dimensi input
    """
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, down=True, use_idty=True, kernel_size = 3, padding = 1, stride = 1),
            ConvBlock(channels, channels, down=True, use_idty=False, kernel_size = 3, padding = 1, stride = 1)
        )

    def forward(self, x):
        return x + self.block(x)
    

class Generator(nn.Module):
    """
    Kelas generator
    Downsampling -> ResBlock -> Upsampling
        - 9 ResBlock untuk gambar 256x256
        - 6 ResBlock untuk gambar 128x128
    """
    def __init__(self, img_channels, num_features = 64, num_resblock = 9):
        super().__init__()
        # first phase
        self.first = nn.Sequential(
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.ReLU(inplace=True)
        )
        # downsampling phase
        self.downsampling = nn.ModuleList(
            [
                ConvBlock(num_features, num_features * 2, down=True, use_idty=True, kernel_size=3, stride=2, padding=1),
                ConvBlock(num_features * 2, num_features * 4, down=True, use_idty=True, kernel_size=3, stride=2, padding=1)
            ]
        )
        # residual blocks/phase
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_resblock)]
        )
        # upsampling phase
        self.upsampling = nn.ModuleList(
            [
                ConvBlock(num_features * 4, num_features * 2, down=False, use_idty=True, kernel_size=3, stride=2, padding=1, output_padding=1),
                ConvBlock(num_features * 2, num_features, down=False, use_idty=True, kernel_size=3, stride=2, padding=1, output_padding=1)
            ]
        )
        # last phase, mengembalikan ke 3 channel
        self.last = nn.Conv2d(num_features, img_channels, kernel_size=7, stride=1, padding=3, padding_mode='reflect')

    def forward(self, x):
        x = self.first(x)
        for layer in self.downsampling:
            x = layer(x)
        x = self.residual_blocks(x)
        for layer in self.upsampling:
            x = layer(x)
        # tanh
        return torch.tanh(self.last(x))