import time
from glob import glob
from tqdm import tqdm
from torch.utils.data import DataLoader
from data import train_test_split
from UNet_model import build_unet
from monai.losses import DiceCELoss
from utils import seeding, train_time
import torch
from monai.networks.nets import SwinUNETR
import argparse
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional.classification import multiclass_f1_score

model_su = SwinUNETR(img_size = (512, 512), in_channels = 3 , out_channels = 3,
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

parser = argparse.ArgumentParser(description='Specify Parameters')
parser.add_argument('lr', metavar='lr', type=float, help='Specify learning rate')
parser.add_argument('b_s', metavar='b_s', type=int, help='Specify bach size')
parser.add_argument('gpu_index', metavar='gpu_index', type=int, help='Specify which gpu to use')
parser.add_argument('model', metavar='model', type=str, choices = ['unet', 'sur'], help='Specify a model')
args = parser.parse_args()
lr, batch_size, gpu_index, model_name = args.lr, args.b_s, args.gpu_index, args.model
'''select between two model'''
if model_name == 'unet':
    model = build_unet()
    model_text = 'UNET'
elif model_name == 'sur':
    model = model_su
    model_text = 'swin_unetr'

writer = SummaryWriter(f'/home/mans4021/Desktop/new_data/REFUGE2/test/1600_{model_text}_lr_{lr}_bs_{batch_size}/', comment= f'UNET_lr_{lr}_bs_{batch_size}')
device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')

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

def evaluate(model, data, loss_fn, device):
    model.eval()
    val_loss, l0, l1, l2, l12 = 0.0, 0.0, 0.0, 0.0, 0.0
    with torch.no_grad():
        for x, y in data:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device)
            y_12_comb = torch.where(y == 2, 1, y)           # prepare for disc

            y_pred = model(x).softmax(dim=1).argmax(dim=1).unsqueeze(dim=1)
            y_12_comb_pred = torch.where(y_pred == 2, 1, y_pred)

            loss = loss_fn(y_pred, y, num_classes=3, average=None)  # return f1, f1_loss_fn requires both in un_one_hot form
            l_12 = loss_fn(y_12_comb_pred, y_12_comb, num_classes=2, average=None,ignore_index=0)[1].item()

            l_0, l_1, l_2 = loss[0].item(), loss[1].item(), loss[2].item()
            l0 += l_0
            l1 += l_1
            l2 += l_2
            l12 += l_12
        val_loss += l1/2 + l2/2
    return  l_0/len(data), l1/len(data), l_2/len(data), l12/len(data), val_loss/len(data)

if __name__ == "__main__":
    """ Seeding """
    seeding(42)
    """ Load dataset """
    train_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/train/image/*"))
    train_y = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/train/mask/*"))
    valid_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/val/image/*"))
    valid_y = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/val/mask/*"))
    f = open(f'/home/mans4021/Desktop/checkpoint/checkpoint_refuge_{model_text}/lr_{lr}_bs_{batch_size}_lowloss.pth','x')
    f.close()
    f = open(f'/home/mans4021/Desktop/checkpoint/checkpoint_refuge_{model_text}/lr_{lr}_bs_{batch_size}_final.pth', 'x')
    f.close()
    checkpoint_path = f'/home/mans4021/Desktop/checkpoint/checkpoint_refuge_{model_text}/lr_{lr}_bs_{batch_size}_lowloss.pth'
    checkpoint_path_final = f'/home/mans4021/Desktop/checkpoint/checkpoint_refuge_{model_text}/lr_{lr}_bs_{batch_size}_final.pth'
    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print(data_str)

    """ Hyperparameters """
    lr1 = lr
    iteration = 2000

    train_dataset = train_test_split(train_x, train_y)
    valid_dataset = train_test_split(valid_x, valid_y)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=400,
        shuffle=False,
        num_workers=4
    )

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr1)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    train_loss_fn = DiceCELoss(include_background=False, softmax=True, lambda_dice=0.5, lambda_ce=0.5, to_onehot_y=True)
    eval_loss_fn  = multiclass_f1_score

    """ Training the model """
    best_valid_loss = float("inf")

    for iteration_n in tqdm(range(iteration)):
        start_time = time.time()
        train_loss = train(model, train_loader, optimizer, train_loss_fn, device)
        l_0, l_1, l_2, l_12, valid_loss = evaluate(model, valid_loader, eval_loss_fn, device)

        writer.add_scalar(f'Training Loss', train_loss, iteration_n)
        writer.add_scalar(f'Validation Background F1', l_0, iteration_n)
        writer.add_scalar(f'Validation Cup F1', l_1, iteration_n)
        writer.add_scalar(f'Validation outer ring F1', l_2, iteration_n)
        writer.add_scalar(f'Validation Disc F1', l_12 , iteration_n)

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

        writer.add_scalar('Best valid loss', best_valid_loss , iteration_n)

        if iteration_n+1 == iteration:
            torch.save(model.state_dict(), checkpoint_path_final)

        end_time = time.time()
        iteration_mins, iteration_secs = train_time(start_time, end_time)

        data_str = f'Iteration: {iteration_n+1:02} | Iteration Time: {iteration_mins}min {iteration_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.6f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.6f}\n'
        print(data_str)