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
        # apakah residual connection digunakan
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
            # proyeksikan dimensi embedding waktu (t)
            # ke jumlah channel output dari maxpool_conv
            nn.Linear(emb_dim, out_channels)
        )

    def forward(self, x, t): # t = [batch_size, emb_dim], x = [batch_size, in_channels, height, width]
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:,:, None, None].repeat(1, 1, x.shape[-2], x.shape[-1]) # emb = [batch_size, out_channels, 1, 1] -> [batch_size, out_channels, height, width]
        # add element-wise
        return x + emb
    
# attention block
class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, num_heads=4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )

    def forward(self, x):
        # ubah bentuk agar bisa diproses oleh MHA
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        # lewatkan layer norm
        x_ln = self.ln(x)
        # MHA
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        # skip connection (dengan x awal)
        attention_value = attention_value + x
        # lewatakan ke ffn dan skip connection (dengan attention_value)
        attention_value = self.ff_self(attention_value) + attention_value
        # ubah kembali bentuknya
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)
