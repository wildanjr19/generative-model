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
H_DIM = 300
Z_DIM = 20
NUM_EPOCHS = 20
BATCH_SIZE = 32
LR_RATE = 0.001

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
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0

    for batch_idx, (data, _) in enumerate(tqdm(train_loader)):
        data = data.to(DEVICE).view(data.size(0), -1)  # ubah menjadi 1D
        
        # Forward
        recon_batch, mu, log_var = model(data)
        
        # Loss calculation
        recon_loss = loss_fn(recon_batch, data)
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recon_loss + kl_divergence
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {train_loss / len(train_loader.dataset):.4f}")

    # simpan hasil rekonstruksi di setiap epoch
    with torch.no_grad():
        sample = data[:16].to(DEVICE)
        recon_sample, _, _ = model(sample)
        recon_sample = recon_sample.view(-1, 1, 28, 28)
        save_image(recon_sample, f'VAE/output/reconstruction_epoch_{epoch+1}.png')