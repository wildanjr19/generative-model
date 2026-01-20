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