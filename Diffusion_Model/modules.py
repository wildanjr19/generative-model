import torch
import torch.nn as nn
import torch.nn.functional as F

# Convolutional Block
class DoubleConv(nn.Module):
    """
    Layer konvolusi pada umumnya, dan terdiri dari dua lapisan konvolusi disertai dengan GruopNorm
    Args : 
        - in_channels : jumlah channel input
        - out_channels : jumlah channel output
        - mid_channels : jumlah channel di layer tengah (default = out_channels)
        - residual (bool): apakah menggunakan residual connection (default = False)
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, residual = False):
        super(DoubleConv, self).__init__()
        self.residual = residual
        # cek apakah nilai mid_channels diisi
        # jika tidak, gunakan out_channels sebagai mid_channels
        if not mid_channels:                                
            mid_channels = out_channels

        # dua layer konvolusi
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
        # output [batch_size, out_channels, height, width]

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
        # output -> [batch_size, out_channels, height/2, width/2]

# UpBlock -- UpSampling
class Up(nn.Module):
    """
    Upsapling layer -- memperbesar/merekonstruksi ukuran gambar
    Flow : 
        Input -> Upsample -> DoubleConv -> Output
        return output + embedding
    """
    def __init__(self, in_channels, out_channels, emb_dim = 256):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2)
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels)
        )

    def forward(self, x, skip_x, t):
        # upsample (dikali 2)
        x = self.up(x) # [batch_size, in_channels, height * 2, width * 2]
        # gabungkan dengan hasil dari encoder(downsampling) pada dimensi channel
        x = torch.cat([skip_x, x], dim=1)
        # lewatkan ke double conv
        x = self.conv(x) # [batch_size, out_channels, height * 2, width * 2]
        # embedding dari waktu (t)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1]) # emb = [batch_size, out_channels, 1, 1] -> [batch_size, out_channels, height * 2, width * 2]
        return x + emb
        # output -> [batch_size, out_channels, height * 2, width * 2]

# attention block
class SelfAttention(nn.Module):
    """
    Flow :
        Input (noisy) -> Down -> SelfAttention -> Up -> Output
    """
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
    
# UNet Block -- Backbone of DDPM
class UNet(nn.Module):
    def __init__(self, c_in = 3, c_out = 3, time_dim = 256, device = 'cuda'):
        super(UNet, self).__init__()
        self.device = device
        self.time_dim = time_dim

        # PHASE 1 - reduksi
        """
        Flow : 
            Input -> Konvolusi -> Down -> Self Attention 3x
            ukuran spasial diperkcil
        """
        self.inc = DoubleConv(c_in, 64)
        self.down_1 = Down(64, 128)
        self.sa_1 = SelfAttention(128, 32)
        self.down_2 = Down(128, 256)
        self.sa_2 = SelfAttention(256, 16)
        self.down_3 = Down(256, 256)
        self.sa_3 = SelfAttention(256, 8)

        # PHASE 2
        """
        Bottleneck -- Deepest part
        """
        self.bot_1 = DoubleConv(256, 512)
        self.bot_2 = DoubleConv(512, 512)
        self.bot_3 = DoubleConv(512, 256)

        # PHASE 3 - rekonstruksi
        """
        Flow :
            Input -> Up -> Self Attention
            ukuran spasial diperbesar
        """
        self.up_1 = Up(512, 128)
        self.sa_4 = SelfAttention(128, 16)
        self.up_2 = Up(256, 64)
        self.sa_5 = SelfAttention(64, 32)
        self.up_3 = Up(128, 64)
        self.sa_6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    # Positional Encoding untuk waktu (t)
    """
    Mengubah nilai waktu (t) menjadi vektor embedding berdimensi (channels)
    Agar model dapat membedakan posisi/timestep unik -- memberitahu model dia sedang bekerja pada waktu berapa
    """
    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    
    def forward(self, x, t):
        # tambahkan satu dimensi untuk t ; [batch_size]
        t = t.unsqueeze(-1).type(torch.float32) # output -> [batch_size, 1]
        # jadikan t menjadi vektor posisi
        t = self.pos_encoding(t, self.time_dim) # [batch_size, time_dim]

        # PHASE 1
        x1 = self.inc(x) # [batch_size, 64, height, width]
        x2 = self.down_1(x1, t) # [batch_size, 128, height/2, width/2]
        x2 = self.sa_1(x2)
        x3 = self.down_2(x2, t) # [batch_size, 256, height/4, width/4]
        x3 = self.sa_2(x3) 
        x4 = self.down_3(x3, t) # [batch_size, 256, height/8, width/8]
        x4 = self.sa_3(x4)

        # PHASE 2
        x4 = self.bot_1(x4) # [batch_size, 512, height/8, width/8]
        x4 = self.bot_2(x4) # [batch_size, 512, height/8, width/8]
        x4 = self.bot_3(x4) # [batch_size, 256, height/8, width/8]

        # PHASE 3
        x = self.up_1(x4, x3, t) # [batch_size, 128, height/4, width/4]
        x = self.sa_4(x)
        x = self.up_2(x, x2, t) # [batch_size, 64, height/2, width/2]
        x = self.sa_5(x)
        x = self.up_3(x, x1, t) # [batch_size, 64, height, width]
        x = self.sa_6(x)

        output = self.outc(x) # [batch_size, c_out, height, width]
        return output
