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
    
# CONFIGURATION
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 0.0004