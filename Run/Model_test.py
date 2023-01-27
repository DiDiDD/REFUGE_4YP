import os, time
from operator import add
from data import DriveDataset
from torch.utils.data import DataLoader
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import torch
from UNet_model import build_unet
from monai.networks.nets import SwinUNETR
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
from utils import create_dir, seeding, calculate_metrics
'''command line initialise hyperparameter'''
import argparse
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
'''Tensorboard'''
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(f'/home/mans4021/Desktop/new_data/REFUGE2/test/{model_text}_lr_{lr}_bs_{batch_size}', comment= f'UNET_lr_{lr}_bs_{batch_size}')

'''Initialisation'''
device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')

def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    return mask

if __name__ == "__main__":
    ''' Seeding, why 42 ???
    Douglas Adams himself revealed the reason why he chose 42 in this message. 
    “It was a joke. It had to be a number, an ordinary, smallish number, and I chose that one. 
    I sat at my desk, stared into the garden and thought ‘42 will do!’ '''
    seeding(42)
    create_dir(f"/home/mans4021/Desktop/new_data/REFUGE2/test/results/lr_{lr}_bs_{batch_size}")

    """ Load dataset """
    test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image/*"))
    test_y = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/mask/*"))
    test_dataset = DriveDataset(test_x, test_y)

    """ Hyperparameters """
    dataset_size = len(test_x)
    H = 512
    W = 512
    size = (W, H)
    checkpoint_path = f'/home/mans4021/Desktop/checkpoint/checkpoint_refuge_{model_text}/lr_{lr}_bs_{batch_size}_lowloss.pth'
    checkpoint_path_final = f'/home/mans4021/Desktop/checkpoint/checkpoint_refuge_{model_text}/lr_{lr}_bs_{batch_size}_final.pth'

    """ Load the checkpoint """
    model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]

    for i in tqdm(range(dataset_size)):
        with torch.no_grad():
            '''Prediction'''
            image = test_dataset[i][0].unsqueeze(0).to(device)            # (1, 3, 512, 512)
            ori_mask = test_dataset[i][1].squeeze(0).to(device)           # (512, 512)
            pred_y = model(image).squeeze(0)                              # (3, 512, 512)
            pred_y = torch.softmax(pred_y, dim=0)                         # (3, 512, 512)
            pred_mask = torch.argmax(pred_y, dim=0).type(torch.int64)     # (512, 512)
            score = calculate_metrics(pred_mask, ori_mask)
            metrics_score = list(map(add, metrics_score, score))
            pred_mask = pred_mask.cpu().numpy()                           # (512, 512)
            ori_mask = ori_mask.cpu().numpy()
            '''Scale value back to image'''
            pred_mask = np.where(pred_mask == 2, 255, pred_mask)
            pred_mask = np.where(pred_mask == 1, 128, pred_mask)
            ori_mask = np.where(ori_mask == 2, 255, ori_mask)
            ori_mask = np.where(ori_mask == 1, 128, ori_mask)
            image= image*127.5 + 127.5
            """ Saving masks """
            ori_mask, pred_mask = mask_parse(ori_mask), mask_parse(pred_mask)
            line = np.ones((512, 20, 3)) * 255     # white line
            '''Create image for us to analyse visually '''
            cat_images = np.concatenate([image.squeeze().permute(1,2,0).cpu().numpy(), line, ori_mask, line, pred_mask], axis=1)
            cv2.imwrite(f"/home/mans4021/Desktop/new_data/REFUGE2/test/results/{model_text}_lr_{lr}_bs_{batch_size}/{i}.png", cat_images)

    jaccard = metrics_score[0]/len(test_x)
    f1 = metrics_score[1]/len(test_x)
    recall = metrics_score[2]/len(test_x)
    precision = metrics_score[3]/len(test_x)
    acc = metrics_score[4]/len(test_x)
    for x in range(len(jaccard)):
        writer.add_scalar(f'Jaccard Score {model_text}_lr_{lr}_bs_{batch_size}', jaccard[x],x)
        writer.add_scalar(f'F1 Score {model_text}_lr_{lr}_bs_{batch_size}', f1[x], x)
        writer.add_scalar(f'Recall Score {model_text}_lr_{lr}_bs_{batch_size}', recall[x], x)
        writer.add_scalar(f'Precision Score {model_text}_lr_{lr}_bs_{batch_size}', precision[x], x)
        writer.add_scalar(f'Accuracy Score {model_text}_lr_{lr}_bs_{batch_size}', acc[x], x)
        print('Jaccard Score', jaccard[x], x)
        print('F1 Score', f1[x], x)
        print('Recall Score', recall[x], x)
        print('Precision Score', precision[x], x)
        print('Accuracy Score', acc[x], x)