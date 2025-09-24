import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter # print tensor

class Discriminator(nn.Module):
    """
    Bagian diskrimintor dari GAN
    """
    def __init__(self, in_features):
        """
        Linear -> LeakyReLU -> Linear -> Sigmoid
        """
        super().__init__()
        self.disc = nn.Sequential(
            nn
        )