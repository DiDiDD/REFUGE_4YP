import torch
import numpy as np
from utils import f1_valid_score

x = torch.randint(0,3,(1,1,2,2))
y = torch.randint(0,3,(1,1,2,2))
z = f1_valid_score(x,y)
print(x)
print(y)
print(z)
