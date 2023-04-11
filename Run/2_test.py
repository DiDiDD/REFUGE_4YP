from swin_unetr_model_with_instance import SwinUNETR_instance
import torch

model = SwinUNETR_instance(img_size = (128, 128), in_channels=3, out_channels=3,
                    depths=(1,1,1,1),
                    num_heads=(3, 6, 12, 24),
                    feature_size=12,
                    norm_name= 'instance',
                    drop_rate=0.0,
                    attn_drop_rate=0.0,
                    dropout_path_rate=0.0,
                    normalize=True,
                    use_checkpoint=False,
                    spatial_dims=2,
                    downsample='merging')

device ='mps'
x = torch.rand((1,3,128,128)).to(device)
model.to(device)

y = model(x)