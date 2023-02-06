import time
from glob import glob
from tqdm import tqdm
from torch.utils.data import DataLoader
from data import train_test_split
from UNet_model import build_unet
from monai.losses import DiceCELoss
from utils import seeding, train_time
import torch
from monai.networks.nets import SwinUNETR
import argparse
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional.classification import multiclass_f1_score


train_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/train/image/*"))
train_y = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/train/mask/*"))

train_dataset = train_test_split(train_x, train_y)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=20,
    shuffle=True,
    num_workers=1
)


for x,y in train_loader:
    print(len(train_loader))
    print(x.size())