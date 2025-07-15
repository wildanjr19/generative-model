import torch
import torch.nn as nn 
import torchvision.datasets as datasets
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from tqdm import tqdm
from  model import VariationalAutoencoder
