import torch
import torch.nn as nn
import torch.nn.functional as F

# Convolutional Block
class DoubleConv(nn.Module):
    """
    Layer konvolusi pada umumnya, dan terdiri dari dua lapisan konvolusi disertai dengan GruopNorm
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, residual = False):
        super(DoubleConv, self).__init__()
        self.residual = residual
        if not mid_channels:                                # cek apakah nilai mid_channels diisi
            mid_channels = out_channels

        # dua lyer konvolusi
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels)
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)
        

# DownBlock -- DownSampling
class Down(nn.Module):
    """
    Downsampling layer -- mereduksi ukuran gambar
    Flow : 
        Input -> Maxpool -> DoubleConv -> Output
        return output + embedding
    Notes : 
        - emb bukan nn.Embedding, tapi dari linear layer
    """
    def __init__(self, in_channels, out_channels, emb_dim = 256):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels)
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels)
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:,:, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb
    
