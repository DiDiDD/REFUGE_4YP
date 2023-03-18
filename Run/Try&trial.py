import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from UTNET_utlis import depthwise_separable_conv, RelativePositionBias
import pdb

in_c=3
heads=4
dim_head=64
attn_drop=0.
proj_drop=0.
reduce_size=16
projection='interp',
rel_pos=True

inner_dim = dim_head * heads
heads = heads
scale = dim_head ** (-0.5)
dim_head = dim_head
reduce_size = reduce_size
projection = projection
rel_pos = rel_pos

        # depthwise conv is slightly better than conv1x1
        # to_qkv = nn.Conv2d(dim, inner_dim*3, kernel_size=1, stride=1, padding=0, bias=True)
        # to_out = nn.Conv2d(inner_dim, dim, kernel_size=1, stride=1, padding=0, bias=True)

to_qkv = depthwise_separable_conv(in_c, inner_dim * 3)
to_out = depthwise_separable_conv(inner_dim, in_c)

attn_drop = nn.Dropout(attn_drop)
proj_drop = nn.Dropout(proj_drop)

if rel_pos:
    # 2D input-independent relative position encoding is a little bit better than
    # 1D input-dependent counterpart
    relative_position_encoding = RelativePositionBias(heads, reduce_size, reduce_size)
    # relative_position_encoding = RelativePositionEmbedding(dim_head, reduce_size)

x = torch.rand((5,3,64,64))
B, C, H, W = x.shape

# B, inner_dim, H, W
qkv = to_qkv(x)
qkv.size()
q, k, v = qkv.chunk(3, dim=1)

if projection == 'interp' and H != reduce_size:
    k, v = map(lambda t: F.interpolate(t, size=reduce_size, mode='bilinear', align_corners=True), (k, v))

elif projection == 'maxpool' and H != reduce_size:
    k, v = map(lambda t: F.adaptive_max_pool2d(t, output_size=reduce_size), (k, v))

q = rearrange(q, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=dim_head, heads=heads,
              h=H, w=W)
print(q.size())
k, v = map(lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=dim_head,
                               heads=heads, h=reduce_size, w=reduce_size), (k, v))

q_k_attn = torch.einsum('bhid,bhjd->bhij', q, k)

if rel_pos:
    relative_position_bias = relative_position_encoding(H, W)
    q_k_attn += relative_position_bias
    # rel_attn_h, rel_attn_w = relative_position_encoding(q, heads, H, W, dim_head)
    # q_k_attn = q_k_attn + rel_attn_h + rel_attn_w

q_k_attn *= scale
q_k_attn = F.softmax(q_k_attn, dim=-1)
q_k_attn = attn_drop(q_k_attn)

out = torch.einsum('bhij,bhjd->bhid', q_k_attn, v)
out = rearrange(out, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=H, w=W, dim_head=dim_head,
                heads=heads)

out = to_out(out)
out = proj_drop(out)

print(out, q_k_attn)

print('')

