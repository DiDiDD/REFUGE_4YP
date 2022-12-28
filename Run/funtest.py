import time
from glob import glob
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from data import DriveDataset
from UNet_model import build_unet
from monai.losses import DiceCELoss
from utils import seeding, create_dir, train_time

test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image/*"))
test_y = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/mask/*"))
test_dataset = DriveDataset(test_x, test_y)
#test_dataset = torch.argmax(test_dataset[0][1], dim=0)
test_dataset = test_dataset[0][1]
print(test_dataset.size())