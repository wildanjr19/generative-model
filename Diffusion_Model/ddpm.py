import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet
import logging
from torch.utils.tensorboard import SummaryWriter

# atur logging
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

class DiffusionModel:
    def __init__(self, noise_steps = 1000, beta_start = 1e-4, beta_end = 0.02,img_size = 256, device = 'cuda'):
        """
        Args : 
            noise_steps : jumlah langkah noise -- t -> T
            beta_start : nilai beta awal (pengatur noise)
            beta_end : nilai beta akhir
            img_size = ukuran gambar 
        """
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        """
        Args : 
            beta : berisi fungsi untuk menentukan noise di setiap langkah
            alpha : berisi fungsi untuk menentukan alpha di setiap langkah
        """
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    # noise schedule
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    # menentukan noise
    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)

        return x * sqrt_alpha_hat + sqrt_one_minus_alpha_hat * epsilon, epsilon
    
    # sample timesteps -- membantu model untuk memproses timestep yang acak agar robust
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    
    # Alghoritm 2 Sampling
    def sample(self, model, n):
        logging.info(f"Sampling {n} gambar baru...")
        
        # setting model to the eval mode
        model.eval()

        with torch.no_grad():
            # inisialisasi noise acak
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            # looping mundur -- generasi gambar -- menghilangkan noise
            for i in tqdm(reversed(range(self.noise_steps)), position = 0):
                t = (torch.ones(n) * i).long().to(self.device)
                # prediksi noise pada input x pada waktu t
                predicted_noise = model(x, t)
                # parameter
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                # calculation x_t - 1
                x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise) + torch.sqrt(beta) * noise
        
        model.train()
        # membatasi nilai x agar berada dalam rentang [0, 1]
        # kemudian menambahkan + 1 dan membagi dengan 2
        x = (x.clamp(-1, 1) + 1) / 2 
        # konversi ke pixel
        x = (x * 255).type(torch.uint8)
        
        return x

# training funtion
def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr = args.lr)
    mse = nn.MSELoss()
    diffusion = DiffusionModel(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch} :")
        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.size(0)).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        sampled_images = diffusion.sample(model, n=images.shape[0])
        save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Uncondtional"
    args.epochs = 5
    args.batch_size = 4
    args.image_size = 64
    args.dataset_path = r"C:\Users\LENOVO\Downloads\datasets\landscape_img_folder"
    args.device = "cuda"
    args.lr = 3e-4
    train(args)

if __name__ == "__main__":
    launch()