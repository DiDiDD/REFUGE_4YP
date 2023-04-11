import torch.nn as nn
import torch
import torch.nn.functional as F

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.norm1 = nn.Instance(out_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.norm(x)
        x = self.relu(x)
        return x


# complete
class ResBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, in_c*2, 3, padding=1)
        self.conv2 = nn.Conv2d(in_c*2, out_c, 3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(in_c*2)
        self.batchnorm2 = nn.BatchNorm2d(out_c)

    def convblock(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        return x

    def forward(self, x):
        return x + self.convblock(x)


class encoder_block(nn.Module):
    def __init__(self, in_c, out_c, num_ResBlock, out_v):
        super().__init__()
        self.resblock = ResBlock(in_c, out_c)
        self.num_res = num_ResBlock
        self.pool = nn.MaxPool2d((2, 2))
        self.out_v = out_v

    def forward(self, x):
        for i in range(self.num_res):
            x = self.resblock(x)

        if self.out_v == True:
            x_v = self.pool(x)
            x_h = x
            return x_v, x_h
        else:
            return x


x = torch.rand((1,1,4,4))
model = encoder_block(in_c= 1, out_c = 1, num_ResBlock=3, out_v= True)
y = model(x)
print(y)