import torch
import torch.nn as nn

# DISCRIMINATOR
class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        """
        Args:
            channels_img (int) : channel dari gambar
            features_d (int) : banyak fitur discriminator
        """
        super(Discriminator, self).__init__()
        # layer sekuensial
        self.disc = nn.Sequential(
            # input layer konvolusi pertama -> N x channels_img x 64 x 64
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # tiga blok konvolusi + BN + LeakyReLU -> output semakin meningkat
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # setelah melewati blok, ukuran gambar jadi 4x4, konvolusi terakhir untuk jadi 1x1
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
            # seperti GAN, akhiri dengan sigmoid
            nn.Sigmoid(),
        )
    
    # helper untuk buat blok
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        # langsung kembalikan lapisan sekuensial
        return nn.Sequential(
            # conv
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            # BN
            nn.BatchNorm2d(out_channels),
            # LeakyReLU
            nn.LeakyReLU(0.2),
        )
    
    # forward pass
    def forward(self, x):
        return self.disc(x)
