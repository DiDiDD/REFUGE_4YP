from monai.networks.nets import SwinUNETR

import monai

import torch

net = SwinUNETR(img_size=(512,512), in_channels=3, out_channels=2, use_checkpoint=False, spatial_dims=2, norm_name = ("layer", {"normalized_shape":[3,512,512]}))

x = torch.rand((2,3,512,512))

y = net(x)

print(y.shape)

