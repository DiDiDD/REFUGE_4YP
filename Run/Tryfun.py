import torch

x = torch.randint(0,3,(400, 512, 512))

y = x*x

print(y.size())