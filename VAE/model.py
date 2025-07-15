import torch
import torch.nn as nn

# Flow Utama
# Input gambar -> Hidden dim -> mean, std -> Reparameterization trick -> Decoder -> Output gambar

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, hid_dim = 200, z_dim = 100):
        super().__init__()

        # encoder
        self.img_to_hid = nn.Linear(input_dim, hid_dim)
        self.hid_to_mu = nn.Linear(hid_dim, z_dim)              # mu
        self.hid_to_sigma = nn.Linear(hid_dim, z_dim)           # sigma

        # decoder
        self.z_to_hid = nn.Linear(z_dim, hid_dim)
        self.hid_to_img = nn.Linear(hid_dim, input_dim)

        # activation function
        self.relu = nn.ReLU()

    def encode(self, x):
        hid = self.relu(self.img_to_hid(x))
        mu, sigma = self.hid_to_mu(hid), self.hid_to_sigma(hid)
        return mu, sigma

    def decode(self, z):
        hid = self.relu(self.z_to_hid(z))
        return torch.sigmoid(self.hid_to_img(hid))              # sigmoid karena nilai tensor antara (0, 1)
    
    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)                       # untuk reparameterization trick
        z_reparametrized = mu + sigma * epsilon
        x_reconstricted = self.decode(z_reparametrized)         # hasil rekontruksi gambar
        return x_reconstricted, mu, sigma
    

if __name__ == "__main__":
    x = torch.randn(4, 28 * 28)

    vae = VariationalAutoencoder(input_dim=784)

    x_recon, mu, sigma = vae(x)

    print(x_recon.shape)
    print(mu.shape)
    print(sigma.shape)