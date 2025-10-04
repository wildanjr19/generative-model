import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter # print tensor

"""GAN For MNIST"""
class Discriminator(nn.Module):
    """
    Bagian diskrimintor dari GAN
    """
    def __init__(self, in_features):
        """
        Args:
            in_features: dimensi input image
            Linear -> LeakyReLU -> Linear -> Sigmoid
        """
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    """
    Bagian generator dari GAN
    """
    def __init__(self, z_dim, img_dim):
        """
        Args:
            z_dim: dimensi input noise
            img_dim: dimensi output image
            Linear -> LeakyReLU -> Linear -> Tanh
        """
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)
    
## -- CONFIGURATION -- ## 
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 0.0004
z_dim = 64
image_dim = 28 * 28 * 28 # 784
batch_size = 32
num_epochs = 50

# inisialisasi dis dan gen
disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)

# inisialisasi noise random awal
fixed_noise = torch.randn((batch_size, z_dim)) .to(device)

# transformasi data
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

## -- DATA -- ##
dataset = datasets.MNIST(root="../GAN/dataset", transform=transforms, download=True)
# data loader
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# optimizers
optimizer_discriminator = optim.Adam(disc.parameters(), lr=lr)
optimizer_generator = optim.Adam(gen.parameters(), lr=lr)

# loss function
"""Loss function yang digunaka untuk GAN adalah BCE, karena BCE umum untuk binary classification."""
criterion = nn.BCELoss()

# logging dengan SummaryWriter
writer_fake = SummaryWriter(f"../GAN/MNIST/fake")
writer_real = SummaryWriter(f"../GAN/MNIST/real")
step = 0

## -- TRAINING -- ##
"""
[lihat bagian 'loss']

Alghoritm
- train discriminator
- train generator

    - 

Kalkulasi Loss
- loss disc real : hitung loss dari hasil disc real dengan tensor 1 (agar hasil mendekati 1 = asli)

"""
for epoch in range(num_epochs):
    for batch_dx, (real, _) in enumerate(loader):
        # flatten data (asli) dengan view
        real = real.view(-1, 784).to(device)
        # ambil ukuran batch
        batch_size = real.shape[0]

        # TRAIN DISCRIMINATOR
        # buat noise dengan z_dim
        noise = torch.randn(batch_size, z_dim).to(device)
        # generate fake dari noise dengan generator
        fake = gen(noise)

        # dapatkan discriminator real
        disc_real = disc(real).view(-1)
        # hitung loss discriminator real
        lossDisc_real = criterion(disc_real, torch.ones_like(disc_real))

        # dapatkan discriminator fake
        disc_fake = disc(fake).view(-1)
        # hitung loss discriminator fake
        lossDisc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        # total loss discriminator (rata-rata)
        lossDisc = (lossDisc_real + lossDisc_fake) / 2

        # backprop dan update
        disc.zero_grad()
        lossDisc.backward(retain_graph=True)
        optimizer_discriminator.step()


        # TRAIN GENERATOR
