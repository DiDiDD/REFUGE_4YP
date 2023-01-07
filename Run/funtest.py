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

x = torch.tensor([1,2,1,2,0])
x1 = torch.tensor([1,1,1,2,0])
score_jaccard_2 = multiclass_jaccard_index(x, x1, num_classes=3, average=None)[2].unsqueeze(0)
print(score_jaccard_2)
