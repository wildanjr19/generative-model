import torch
import torch.nn as nn

class Block(nn.Module):
    """
    Block konvolusi untuk discriminator
    """
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=True, padding_mode='reflect'),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)
    
class Discriminator(nn.Module):
    """
    Kelas discriminator
    Agar mudah pakai looping di input dan output channels
    """
    def __init__(self, in_channels = 3, features = [64, 128, 256, 512]):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        )

        layers = []
        in_channels = features[0]

        for feature in features[1:]:
            layers.append(Block(in_channels, feature, stride=1 if feature == features[-1] else 2)) # last layer paddingnya 1
            in_channels = feature

        # layer terakhir agar output skalar
        layers.append(
            nn.Conv2d(in_channels, out_channels=1, kernel_size=4, stride=1, padding=1, padding_mode='reflect')
        )

        # unwrap
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # masuk ke first layer
        x = self.first(x)
        # masuk ke model, karena BCE pakai sigmoid
        return torch.sigmoid(self.model(x))