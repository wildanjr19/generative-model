import torch
import torch.nn as nn
import torch.optim as optim
import config
from datasets import HorseZebraDataset
import sys
# from
from torch.utils.data import DataLoader
from tqdm import tqdm
from discriminator import Discriminator
from generator import Generator
from utils import load_checkpoint

# training function
def train(disc_H, disc_Z, gen_H, gen_Z, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    # set tqdm
    loop = tqdm(loader, leave=True)

    # main loop
    for idx, (zebra, horse) in enumerate(loop):
        # move ke device
        zebra = zebra.to(config.DEVICE)
        horse = horse.to(config.DEVICE)

        # train discriminator H dan Z dengan AMP
        with torch.cuda.amp.autocast():
            # generate fake horse dari zebra
            fake_horse = gen_H(zebra)
            # masuk ke discriminator H
            D_H_real = disc_H(horse) # disc horse asli
            D_H_fake = disc_H(fake_horse.detach())  # disc horse palsu
            # loss
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real)) # 1 untuk asli
            D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake)) # 0 untuk palsu
            D_H_loss = (D_H_real_loss + D_H_fake_loss)

            fake_zebra = gen_Z(horse)
            D_Z_real = disc_Z(zebra) # disc zebra asli
            D_Z_fake = disc_Z(fake_zebra.detach())  # disc zebra palsu
            D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real)) # 1 untuk asli
            D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake)) # 0 untuk palsu
            D_Z_loss = (D_Z_real_loss + D_Z_fake_loss)

            # loss total discriminator
            D_loss = (D_H_loss + D_Z_loss) / 2

        # backward discriminator
        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # train generator H dan Z dengan AMP
        with torch.cuda.amp.autocast():
            # adversarial loss untuk generator
            D_H_fake = disc_H(fake_horse)
            D_Z_fake = disc_Z(fake_zebra)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake)) # agar disc mengira ini asli
            loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake)) # agar disc mengira ini asli

            # cycle loss (L1 loss)
            cycle_zebra = gen_Z(fake_horse) # generate zebra dari fake horse
            cycle_horse = gen_H(fake_zebra) # generate horse dari fake zebra
            cycle_zebra_loss = l1(zebra, cycle_zebra)
            cycle_horse_loss = l1(horse, cycle_horse)

            # identity loss (L1 loss)
            identity_zebra = gen_Z(zebra) # generate zebra dari zebra asli
            identity_horse = gen_H(horse) # generate horse dari horse asli
            identity_zebra_loss = l1(zebra, identity_zebra)
            identity_horse_loss = l1(horse, identity_horse)

            # total loss generator
            G_loss = (
                loss_G_H
                + loss_G_Z
                + cycle_zebra_loss * config.LAMBDA_CYCLE
                + cycle_horse_loss * config.LAMBDA_CYCLE
                + identity_zebra_loss * config.LAMBDA_IDENTITY
                + identity_horse_loss * config.LAMBDA_IDENTITY
            )

        # backward generator
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

# main
def main():
    # inisialisasi
    disc_H = Discriminator(in_channels=3).to(config.DEVICE)
    disc_Z = Discriminator(in_channels=3).to(config.DEVICE)
    gen_H = Generator(img_channels=3, num_resblock=9).to(config.DEVICE)
    gen_Z = Generator(img_channels=3, num_resblock=9).to(config.DEVICE)

    # optimizer disc
    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    # optimizer gen
    opt_gen = optim.Adam(
        list(gen_H.parameters()) + list(gen_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.6, 0.999),
    )

    # define loss funciton
    mse = nn.MSELoss()
    l1 = nn.L1Loss()

    # Load Checkpoint
    # load Gen Z, Gen H, Disc Z, dan Disc H (jika perlu)
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_H,
            gen_H,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_Z,
            gen_Z,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_H,
            disc_H,
            opt_disc,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_Z,
            disc_Z,
            opt_disc,
            config.LEARNING_RATE
        )

    # dataset (train val)
    train_dataset = HorseZebraDataset(
        root_horse=config.TRAIN_DIR + "horse/",
        root_zebra=config.TRAIN_DIR + "zebra/",
        transform=config.transforms,
    )

    val_dataset = HorseZebraDataset(
        root_horse=config.VAL_DIR + "horse/",
        root_zebra=config.VAL_DIR + "zebra/",
        transform=config.transforms,
    ) 

    # dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )
