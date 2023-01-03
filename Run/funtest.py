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
from torchmetrics.functional.classification import multiclass_jaccard_index

f = open('/home/mans4021/Desktop/c1.pth', 'x')
f.close()