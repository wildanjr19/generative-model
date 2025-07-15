import torch
import torch.nn as nn 
import torch.optim as optim
import torchvision.datasets as datasets
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from tqdm import tqdm
from  model import VariationalAutoencoder


# config
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_DIM = 784
H_DIM = 200
Z_DIM = 20
NUM_EPOCHS = 10
BATCH_SIZE = 32
LR_RATE = 3e-4

# Data prep
dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

train_loader = DataLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Model
model = VariationalAutoencoder(INPUT_DIM, H_DIM, Z_DIM).to(DEVICE)

# optimizer
optimizer = optim.Adam(model.parameters(), lr=LR_RATE)

# loss function
loss_fn = nn.BCELoss(reduction='sum')

# training loop