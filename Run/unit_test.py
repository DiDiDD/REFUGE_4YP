from utils import norm
import torch
from UNet_model import UNet

x = torch.rand(3,2,128,128)

y = UNet(in_c=2, out_c=3, base_c=12, norm_name ='instance')(x)

print(y.shape)