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

        self.d1 = decoder_block()
        self.d2 = decoder_block()
        self.d3 = decoder_block()

    def forward(self, input):
        """ CNN-Encoder """
        x0_v = self.e0(input)
        x1_v, x1_h = self.e1(x0_v)
        x2_v, x2_h = self.e2(x1_v)
        x3_h = self.e3(x2_v)

        """ DeTrans-Encoder """
        v1, v2, v3 = self.d(x1_h, x2_h, x3_h)


        '''Decoder'''
        v2 = torch.cat((self.d3(v3), h3), dim=1)
        v2
        v1 = torch.cat((self.d2(v2), h2), dim=1)
        v0 = torch.cat((self.d1(v1), h1), dim=1)









