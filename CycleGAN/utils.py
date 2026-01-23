import random
import torch
import os
import numpy as np
import config

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("Load Checkpoint...")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    # get LR
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(model, optimizer, filename = "my_checkpoint.pth.tar"):
    print("Save Checkpoint...")
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)