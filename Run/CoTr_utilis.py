import torch
import torch.nn as nn


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


class ResBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, in_c*2, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_c, out_c, 3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(hidden_c)
        self.batchnorm2 = nn.BatchNorm2d(out_c)

    def convblock(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        return x

    def forward(self, x):
        return x + self.convblock(x)


class encoder_block(nn.Module):
    def __init__(self, in_c, hidden_c, out_c, num_ResBlock, out_v):
        super().__init__()
        self.resblock = ResBlock(in_c, hidden_c, out_c)
        self.num_res = num_ResBlock
        self.pool = nn.MaxPool2d((2, 2))
        self.out_v = out_v

    def forward(self, inputs):
        for i in range(self.num_ResBlock):
            x = self.resblock(x)

        if out_v==True:
            x_v = self.pool(x)
            x_h = x
            return x_v, x_h
        else:
            return x


def relative_postional_encoding_2d(height, width, d_model):
    h_pos = torch.arange(height).unsqueeze(1).repeat(1, width)
    w_pos = torch.arange(width).unsqueeze(0).repeat(height, 1)
    pos = torch.stack([h_pos, w_pos], dim=-1).float()
    pos = pos.unsqueeze(0)
    return pos


class MS_DMSA(nn.Module):
    def __init__(self):
        super().__init__()


class feed_forward(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        return


class detranslayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.MS_DMSA = MS_DMSA()
        self.feedforward = feed_forward()

    def forward(self, x, num):
        for i in range(num):
            x = MS_DMSA(x)
            x = feed_forward(x)
        return x


class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.resblock = ResBlock(in_c, hidden_size, out_c)
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)

    def forward(self, inputs, skip):
        x = self.resblock(inputs)
        x = self.up(x)
        x = torch.cat((x,skip), dim=1)
        return x


class decoder_block_final(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.resblock = ResBlock()
        self.upsample = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv_1 = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0)

    def forward(self):
        return


