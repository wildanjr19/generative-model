import torch
import torch.nn as nn

# DISCRIMINATOR
class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        """
        Args:
        Discriminator akan melakukan konvolusi (downsampling) pada gambar input
        Kemudian akan berusaha membedakan apakah gambar tersebut nyata
            channels_img (int) : channel dari gambar
            features_d (int) : banyak fitur discriminator
        """
        super(Discriminator, self).__init__()
        # layer sekuensial
        self.disc = nn.Sequential(
            # input layer konvolusi pertama -> N x channels_img x 64 x 64
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # blok pertama -> N x features_d*2 x 32 x 32
            self._block(features_d, features_d * 2, 4, 2, 1),
            # blok kedua -> N x features_d*4 x 16 x 16
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            # blok ketiga -> N x features_d*8 x 8 x 8
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # setelah melewati blok, ukuran gambar jadi 4x4, konvolusi terakhir untuk jadi 1x1
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
            # seperti GAN, akhiri dengan sigmoid (untuk membedakan nyata/palsu)
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
    

# GENERATOR
class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        """
        Args:
            channels_noise (int) : channel dari noise input, dilatih untuk jadi gambar seperti input nyata
            channels_img (int) : channel dari gambar output
            faetures_g (int) : banyak fitur generator
        """
        super(Generator, self).__init__()
        # layer sekuensial (blok - blok kemudian layer output)
        # Input -> N x channels_noise x 1 x 1
        # upsampling proses
        self.gen = nn.Sequential(
            # blok pertama -> N x features_g*16 x 4 x 4
            self._block(channels_noise, features_g * 16, 4, 1, 0),
            # blok kedua -> N x features_g*8 x 8 x 8
            self._block(features_g * 16, features_g * 8, 4, 2, 1),
            # blok ketiga -> N x features_g*4 x 16 x 16
            self._block(features_g * 8, features_g * 4, 4, 2, 1),
            # blok keempat -> N x features_g*2 x 32 x 32
            self._block(features_g * 4, features_g * 2, 4, 2, 1),
            # layer output -> N x channels_img x 64 x 64
            nn.ConvTranspose2d(
                features_g * 2,
                channels_img,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            # akhiri dengan tanh untuk output antara -1 dan 1
            nn.Tanh(),
        )

    # helper untuk buat blok
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        # ConvTranspose2d + BN + ReLu
        nn.Sequential(
            # ConvTranspose2d
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            # BN
            nn.BatchNorm2d(out_channels),
            # ReLU
            nn.ReLU(),
        )

    # forward pass
    def forward(self, x):
        return self.gen(x)
