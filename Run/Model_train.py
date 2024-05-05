import time
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from data_aug.data import train_test_split
from UNET.UNet_model import UNet
from monai.losses import DiceCELoss
from utils import *
import torch
from monai.networks.nets import SwinUNETR
from Swin_UNETR.swin_unetr_model_with_batch_in_trans import SwinUNETR_batch
from Swin_UNETR.swin_unetr_model_with_instance import SwinUNETR_instance
import argparse
from torch.utils.tensorboard import SummaryWriter
from UTNET._UTNET_model import UTNet

parser = argparse.ArgumentParser(description='Specify Parameters')

parser.add_argument('lr', metavar='lr', type=float, help='Specify learning rate')
parser.add_argument('b_s', metavar='b_s', type=int, help='Specify bach size')
parser.add_argument('gpu_index', metavar='gpu_index', type=int, help='Specify which gpu to use')
parser.add_argument('model', metavar='model', type=str, choices=['unet', 'swin_unetr', 'utnet'], help='Specify a model')

parser.add_argument('norm_name', metavar='norm_name',  help='Specify a normalisation method')
# parser.add_argument('model_text', metavar='model_text', type=str, help='Describe your mode')
parser.add_argument('--base_c', metavar='--base_c', default = 12,type=int, help='base_channel which is the first output channel from first conv block')
# swin_unetr paras
parser.add_argument('--depth', metavar='--depth', type=str, default = '[2,2,2,2]',  help='num_depths in swin_unetr')
parser.add_argument('--n_h', metavar='--n_h', type=str, default = '[3,6,12,24]',  help='num_heads in swin_unetr')

args = parser.parse_args()
lr, batch_size, gpu_index, model_name, norm_name = args.lr, args.b_s, args.gpu_index, args.model, args.norm_name
base_c = args.base_c
depths = args.depth
depths= json.loads(depths)
depths = tuple(depths)
num_heads = args.n_h
num_heads= json.loads(num_heads)
num_heads = tuple(num_heads)

model_su = SwinUNETR(img_size = (512, 512), in_channels=3, out_channels=3,
                    depths=depths,
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

model_su2 = SwinUNETR_batch(img_size = (512, 512), in_channels=3, out_channels=3,
                    depths=depths,
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

model_su3 = SwinUNETR_instance(img_size = (512, 512), in_channels=3, out_channels=3,
                    depths=depths,
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

utnet = UTNet(in_chan=3, num_classes=3, base_chan=base_c)

unet = UNet(in_c=3, out_c=3, base_c=base_c, norm_name=norm_name)

data_save_path = 'to be specify'
'''select between two model'''
if model_name == 'unet':
    model = unet
    data_save_path = f'/home/mans4021/Desktop/new_data/REFUGE2/test/1600_{model_name}_{norm_name}_lr_{lr}_bs_{batch_size}_fs_{base_c}/'
elif model_name == 'swin_unetr' and norm_name== 'layer':
    model = model_su
    data_save_path = f'/home/mans4021/Desktop/new_data/REFUGE2/test/1600_{model_name}_{norm_name}_lr_{lr}_bs_{batch_size}_fs_{base_c}_nd_{depths}_nh_{num_heads}/'
elif model_name == 'swin_unetr' and norm_name == 'batch':
    model = model_su2
    data_save_path = f'/home/mans4021/Desktop/new_data/REFUGE2/test/1600_{model_name}_{norm_name}_lr_{lr}_bs_{batch_size}_fs_{base_c}_nd_{depths}_nh_{num_heads}/'
elif model_name == 'swin_unetr' and norm_name == 'instance':
    model = model_su3
    data_save_path = f'/home/mans4021/Desktop/new_data/REFUGE2/test/1600_{model_name}_{norm_name}_lr_{lr}_bs_{batch_size}_fs_{base_c}_nd_{depths}_nh_{num_heads}/'
elif model_name == 'utnet':
    model = utnet
    data_save_path = f'/home/mans4021/Desktop/new_data/REFUGE2/test/1600_{model_name}_{norm_name}_lr_{lr}_bs_{batch_size}_fs_{base_c}/'

writer = SummaryWriter(data_save_path, comment = '_training')
device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')
create_dir(data_save_path + 'Checkpoint')


def train(model, data, optimizer, loss_fn, device):
    iteration_loss = 0.0
    model.train()
    for x, y in data:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        iteration_loss += loss.item()
    iteration_loss = iteration_loss/len(data)
    return iteration_loss


def evaluate(model, data, score_fn, device):
    model.eval()
    val_score= 0
    f1_score_record = np.zeros(4)
    with torch.no_grad():
        for x, y in data:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device)
            y_pred = model(x).softmax(dim=1).argmax(dim=1).unsqueeze(dim=1)
            score = score_fn(y, y_pred)
            val_score = val_score + score[1].item()/2 + score[2].item()/2
            f1_score_record += score

    f1_score_record = f1_score_record/len(data)
    val_score = val_score/len(data)
    return f1_score_record[0].item(), f1_score_record[1].item(), f1_score_record[2].item(), f1_score_record[3].item(), val_score


if __name__ == "__main__":
    seeding(42)
    train_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/train/image/*"))
    train_y = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/train/mask/*"))
    valid_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/val/image/*"))
    valid_y = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/val/mask/*"))
    checkpoint_path_lowloss = data_save_path + f'Checkpoint/lr_{lr}_bs_{batch_size}_lowloss.pth'
    checkpoint_path_final = data_save_path + f'Checkpoint/lr_{lr}_bs_{batch_size}_final.pth'
    create_file(checkpoint_path_lowloss)
    create_file(checkpoint_path_final)
    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print(data_str)

    train_dataset, valid_dataset = train_test_split(train_x, train_y),  train_test_split(valid_x, valid_y)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False, num_workers=4)

    model = model.to(device)

    iteration = 2000
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    train_loss_fn = DiceCELoss(include_background=False, softmax=True, to_onehot_y=True, lambda_dice=0.5, lambda_ce=0.5)
    eval_loss_fn = f1_valid_score

    """ Training the model """
    best_valid_score = 0.0

    for iteration_n in tqdm(range(iteration)):
        optimizer = torch.optim.Adam(model.parameters(), lr=get_lr(iteration, lr))
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        start_time = time.time()
        train_loss = train(model, train_loader, optimizer, train_loss_fn, device)
        s_bg, s_outer, s_cup, s_disc, valid_score = evaluate(model, valid_loader, eval_loss_fn, device)

        writer.add_scalar(f'Training Loss', train_loss, iteration_n)
        writer.add_scalar(f'Validation Background F1', s_bg, iteration_n)
        writer.add_scalar(f'Validation Outer Ring F1', s_outer, iteration_n)
        writer.add_scalar(f'Validation Cup F1', s_cup, iteration_n)
        writer.add_scalar(f'Validation Disc F1', s_disc, iteration_n)
        writer.add_scalar(f'Validation Score', valid_score, iteration_n)

        '''updating the learning rate'''
        # if iteration_n+1 == 1000:
        #     lr1 = 1e-4
        # elif iteration_n+1 == 2000:
        #     lr1 = 5e-5

        """ Saving the model """
        if valid_score > best_valid_score:
            data_str = f"Valid score improved from {best_valid_score:2.8f} to {valid_score:2.8f}. Saving checkpoint: {checkpoint_path_lowloss}"
            print(data_str)
            best_valid_score = valid_score
            torch.save(model.state_dict(), checkpoint_path_lowloss)

        if iteration_n+1 == iteration:
            torch.save(model.state_dict(), checkpoint_path_final)

        end_time = time.time()
        iteration_mins, iteration_secs = train_time(start_time, end_time)

        data_str = f'Iteration: {iteration_n+1:02} | Iteration Time: {iteration_mins}min {iteration_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.8f}\n'
        data_str += f'\t Val Score: {valid_score:.8f}\n'
        print(data_str)
writer.flush()
writer.close()