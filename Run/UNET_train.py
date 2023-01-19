import time
from glob import glob
from tqdm import tqdm
from torch.utils.data import DataLoader
from data import DriveDataset
from UNet_model import build_unet
from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss
from utils import seeding, train_time
import torch

import argparse
parser = argparse.ArgumentParser(description='Specify Parameters')
parser.add_argument('lr', metavar='lr', type=float, help='Specify learning rate')
parser.add_argument('b_s', metavar='b_s', type=int, help='Specify bach size')
parser.add_argument('gpu_index', metavar='gpu_index', type=int, help='Specify which gpu to use')
args = parser.parse_args()
lr = args.lr
gpu_index = args.gpu_index
batch_size = args.b_s

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(f'/home/mans4021/Desktop/new_data/REFUGE2/test/UNET_lr_{lr}_bs_{batch_size}/', comment= f'UNET_lr_{lr}_bs_{batch_size}')

'''Initialisation'''
device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')

model_su = SwinUNETR(img_size = [512, 512], in_channels = 3 , out_channels = 1,
                    depths=(2, 2, 2, 2),
                    num_heads=(3, 6, 12, 24),
                    feature_size=24,
                    norm_name='batch',
                    drop_rate=0.0,
                    attn_drop_rate=0.0,
                    dropout_path_rate=0.0,
                    normalize=True,
                    use_checkpoint=False,
                    spatial_dims=2,
                    downsample='merging')

def train(model, loader, optimizer, loss_fn, device):
    iteration_loss = 0.0
    model.train()
    for x, y in loader:
        y = torch.nn.functional.one_hot(y, 3).squeeze()
        y = y.permute(0,3,1,2)
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        iteration_loss += loss.item()

    iteration_loss = iteration_loss/len(loader)
    return iteration_loss

def evaluate(model, loader, loss_fn, device):
    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            y = torch.nn.functional.one_hot(y, 3).squeeze()
            y = y.permute(0, 3, 1, 2)
            x = x.to(device, dtype=torch.float32)
            y = y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            val_loss += loss.item()

        val_loss = val_loss/len(loader)
    return val_loss

if __name__ == "__main__":
    """ Seeding """
    seeding(42)
    """ Load dataset """
    train_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/train/image/*"))
    train_y = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/train/mask/*"))
    valid_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/val/image/*"))
    valid_y = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/val/mask/*"))
    '''check size of datasets'''
    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print(data_str)

    """ Hyperparameters """
    lr1 = lr
    H = 512
    W = 512
    size = (H, W)
    iteration = 5000  #change
    f = open(f'/home/mans4021/Desktop/checkpoint/checkpoint_refuge_unet/lr_{lr}_bs_{batch_size}_lowloss.pth', 'x')
    f.close()
    f = open(f'/home/mans4021/Desktop/checkpoint/checkpoint_refuge_unet/lr_{lr}_bs_{batch_size}_final.pth', 'x')
    f.close()
    checkpoint_path = f'/home/mans4021/Desktop/checkpoint/checkpoint_refuge_unet/lr_{lr}_bs_{batch_size}_lowloss.pth'
    checkpoint_path_final = f'/home/mans4021/Desktop/checkpoint/checkpoint_refuge_unet/lr_{lr}_bs_{batch_size}_final.pth'

    """ Dataset and loader """
    train_dataset = DriveDataset(train_x, train_y)
    valid_dataset = DriveDataset(valid_x, valid_y)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    model = build_unet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr1)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = DiceCELoss(softmax=True, lambda_dice=0.5, lambda_ce=0.5)

    """ Training the model """
    best_valid_loss = float("inf")

    for iteration_n in tqdm(range(iteration)):
        start_time = time.time()
        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss = evaluate(model, valid_loader, loss_fn, device)

        writer.add_scalar(f'Training Loss UNET_lr_{lr}_bs_{batch_size}', train_loss, iteration_n)
        writer.add_scalar(f'Validation Loss UNET_lr_{lr}_bs_{batch_size}', valid_loss, iteration_n)
        writer.add_scalar(f'Validation DICE UNET_lr_{lr}_bs_{batch_size}', 1-valid_loss, iteration_n)

        '''updating the learning rate'''
        # if iteration_n+1 == 1000:
        #     lr1 = 1e-4
        # elif iteration_n+1 == 2000:
        #     lr1 = 5e-5

        """ Saving the model """
        if valid_loss < best_valid_loss:
            data_str = f"Valid loss improved from {best_valid_loss:2.6f} to {valid_loss:2.6f}. Saving checkpoint: {checkpoint_path}"
            print(data_str)
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)

        if iteration_n+1 == iteration:
            torch.save(model.state_dict(), checkpoint_path_final)

        end_time = time.time()
        iteration_mins, iteration_secs = train_time(start_time, end_time)

        data_str = f'Iteration: {iteration_n+1:02} | Iteration Time: {iteration_mins}min {iteration_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.6f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.6f}\n'
        print(data_str)