from CoTr_utilis import *
import torch
import torch.nn as nn

class CoTr(nn.Module):
    def __init__(self):
        super().__init__()
        self.e0 = conv_block(3, 32)
        self.e1 = encoder_block(3, 64, 64, 3)
        self.e2 = encoder_block(3, 128, 128, 3)
        self.e3 = encoder_block(3, 128, 128, 2, out_v=False)

        self.TransEncoder = detranslayer(num=3)

        self.d2 = decoder_block(128, 64)
        self.d1 = decoder_block(128, 32)
        self.d0 = decoder_block_final(64, 3)

    def forward(self, input):
        """ CNN-Encoder """
        x0_v = self.e0(input)
        x1_v, x1_h = self.e1(x0_v)
        x2_v, x2_h = self.e2(x1_v)
        x3_h = self.e3(x2_v)

        """ DeTrans-Encoder """
        v1, v2, v3 = self.TransEncoder(x1_h, x2_h, x3_h)

        '''Decoder'''
        de_l2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, padding=0)(v3)
        de_l1 = self.d2(de_l2, v2)
        de_l0 = self.d1(de_l1, v1)
        output = self.d0(de_l0, v0)

        return output









