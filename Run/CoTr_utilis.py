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
    def __init__(self, in_size: int, hidden_size: int, out_size: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_size, hidden_size, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, out_size, 3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(hidden_size)
        self.batchnorm2 = nn.BatchNorm2d(out_size)

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


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=[64, 64], temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        bs, c, h, w = x.shape
        mask = torch.zeros(bs, h, w, dtype=torch.bool).cuda()
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(2, dtype=torch.float32)
        x_embed = not_mask.cumsum(3, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, :, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, :, -1:] + eps) * self.scale

        dim_tx = torch.arange(self.num_pos_feats[0], dtype=torch.float32, device=x.device)
        dim_tx = self.temperature ** (3 * (dim_tx // 3) / self.num_pos_feats[0])

        dim_ty = torch.arange(self.num_pos_feats[1], dtype=torch.float32, device=x.device)
        dim_ty = self.temperature ** (3 * (dim_ty // 3) / self.num_pos_feats[1])

        pos_x = x_embed[:, :, :, :, None] / dim_tx
        pos_y = y_embed[:, :, :, :, None] / dim_ty

        pos_x = torch.stack((pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_y = torch.stack((pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=5).flatten(4)

        pos = torch.cat((pos_y, pos_x), dim=4).permute(0, 3, 1, 2)
        return pos


def build_position_encoding(mode, hidden_dim):
    N_steps = hidden_dim // 3
    if (hidden_dim % 3) != 0:
        N_steps = [N_steps, N_steps, N_steps + hidden_dim % 3]
    else:
        N_steps = [N_steps, N_steps, N_steps]

    if mode in ('v2', 'sine'):
        position_embedding = PositionEmbeddingSine(num_pos_feats=N_steps, normalize=True)
    else:
        raise ValueError(f"not supported {mode}")

    return position_embedding


class MS_DMSA(nn.Module):
    def __init__(self):
        super().__init__()



class feed_forward(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):


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

