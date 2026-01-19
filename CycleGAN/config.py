import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/train/"
VAL_DIR = "data/val/"
BATCH_SIZE = 1
LEARNING_RATE = 2e-4
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10.0
NUM_WORKERS = 2
NUM_EPOCHS = 10
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_H = "checkpoint_gen_H.pth"
CHECKPOINT_GEN_Z = "checkpoint_gen_Z.pth"
CHECKPOINT_CRITIC_H = "checkpoint_critic_H.pth"
CHECKPOINT_CRITIC_Z = "checkpoint_critic_Z.pth"

# transform function
transforms = A.Compose(
    [
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ]
)