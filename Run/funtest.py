import time
from glob import glob
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from UNet_model import build_unet
from utils import seeding, create_dir, train_time
import random
import os


seed = 40
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
print(os.environ["PYTHONHASHSEED"])
np.random.seed(seed)
print(np.random.seed(seed))
torch.manual_seed(seed)
print(torch.manual_seed(seed))
torch.cuda.manual_seed(seed)
print(torch.cuda.manual_seed(seed))
torch.backends.cudnn.deterministic = True
print(torch.backends.cudnn.deterministic)
