import torch
import numpy as np
from utils import segmentation_score

x = torch.randint(0,3,(1,1,512,512))
y = torch.randint(0,3,(1,1,512,512))
z = segmentation_score(x,y,3)
# print(segmentation_score(x,y,3)+segmentation_score(x,y,3))
