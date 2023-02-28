from data import train_test_split
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import torch
from UNet_model import build_unet
from monai.networks.nets import SwinUNETR
from utils import create_dir, segmentation_score, mask_parse
import argparse
from torch.utils.tensorboard import SummaryWriter

'''command line initialisation '''
parser = argparse.ArgumentParser(description='Specify Parameters')
parser.add_argument('lr', metavar='lr', type=float, help='Specify learning rate')
parser.add_argument('b_s', metavar='b_s', type=int, help='Specify bach size')
parser.add_argument('gpu_index', metavar='gpu_index', type=int, help='Specify which gpu to use')
parser.add_argument('model', metavar='model', type=str, choices=['unet', 'swin_unetr', 'utnet', 'cotr'], help='Specify a model')
parser.add_argument('norm_name', metavar='norm_name', type=str, choices=['instance', "batch", "layer"], help='Specify a normalisation method')
parser.add_argument('model_text', metavar='model_text', type=str, help='Describe your mode')
args = parser.parse_args()
lr, batch_size, gpu_index, model_name, norm_name, model_text = args.lr, args.b_s, args.gpu_index, args.model, args.model_text, args.norm_name

'''swin_unetr model initialisation'''
model_su = SwinUNETR(img_size=(512, 512), in_channels=3, out_channels=3,
                     depths=(2, 2, 2, 2),
                     num_heads=(3, 6, 12, 24),
                     feature_size=12,
                     norm_name=norm_name,
                     drop_rate=0.0,
                     attn_drop_rate=0.0,
                     dropout_path_rate=0.0,
                     normalize=True,
                     use_checkpoint=False,
                     spatial_dims=2,
                     downsample='merging')

'''select between two model'''
if model_name == 'unet':
    model = build_unet()
elif model_name == 'swin_unetr':
    model = model_su
elif model_name == 'utnet':
    model = utnet


'''Tensorboard'''
data_save_path = f'/home/mans4021/Desktop/new_data/REFUGE2/test/1600_{model_text}_{norm_name}_lr_{lr}_bs_{batch_size}/'
writer = SummaryWriter(data_save_path)

device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')
create_dir(data_save_path+'results/')

if __name__ == "__main__":
    """ Load dataset """
    test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image/*"))
    test_y = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/mask/*"))
    test_dataset = train_test_split(test_x, test_y)
    dataset_size = len(test_x)
    checkpoint_path_lowloss = data_save_path + f'Checkpoint/lr_{lr}_bs_{batch_size}_lowloss.pth'
    checkpoint_path_final = data_save_path + f'Checkpoint/lr_{lr}_bs_{batch_size}_final.pth'

    """ Load the checkpoint """
    model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    metrics_score = np.zeros((dataset_size,4,5))
    for i in tqdm(range(dataset_size)):
        with torch.no_grad():
            '''Prediction'''
            image = test_dataset[i][0].unsqueeze(0).to(device)         # (1, 3, 512, 512)
            ori_mask = test_dataset[i][1].squeeze(0).to(device)        # (512, 512)
            pred_y = model(image).squeeze(0)                           # (3, 512, 512)
            pred_y = torch.softmax(pred_y, dim=0)                      # (3, 512, 512)
            pred_mask = torch.argmax(pred_y, dim=0).type(torch.int64)  # (512, 512)
            score = segmentation_score(ori_mask, pred_mask, num_classes=3)
            metrics_score[i] = score
            pred_mask = pred_mask.cpu().numpy()  # (512, 512)
            ori_mask = ori_mask.cpu().numpy()

            '''Scale value back to image'''
            pred_mask = np.where(pred_mask == 2, 255, pred_mask)
            pred_mask = np.where(pred_mask == 1, 128, pred_mask)
            ori_mask = np.where(ori_mask == 2, 255, ori_mask)
            ori_mask = np.where(ori_mask == 1, 128, ori_mask)
            image = image * 127.5 + 127.5

            ori_mask, pred_mask = mask_parse(ori_mask), mask_parse(pred_mask)
            line = np.ones((512, 20, 3)) * 255  # white line
            '''Create image for us to analyse visually '''
            cat_images = np.concatenate([image.squeeze().permute(1, 2, 0).cpu().numpy(), line, ori_mask, line, pred_mask], axis=1)
            cv2.imwrite(f"/home/mans4021/Desktop/new_data/REFUGE2/test/1600_{model_text}_{norm_name}_lr_{lr}_bs_{batch_size}/results/{i}.png", cat_images)

    np.save(f"/home/mans4021/Desktop/new_data/REFUGE2/test/1600_{model_text}_{norm_name}_lr_{lr}_bs_{batch_size}/test_score", metrics_score)
    f1_record = metrics_score[:, :, 1]
    f1_mean = metrics_score.mean(axis=0)
    f1_std = np.std(f1_record, axis=0)

    f1_report_str = f'1600_{model_text}_lr_{lr}_bs_{batch_size} test results are:'
    f1_report_str += f'\nBackground F1 score is {f1_mean[0,1]:4f} +- {f1_std[0]:4f}'
    f1_report_str += f'\nOuter Ring F1 score is {f1_mean[1,1]:4f} +- {f1_std[1]:4f}'
    f1_report_str += f'\nCup F1 score is {f1_mean[2,1]:4f} +- {f1_std[2]:4f}'
    f1_report_str += f'\nDisc F1 score is {f1_mean[3,1]:4f} +- {f1_std[3]:4f}'
    writer.add_text('Test f1 score', f1_report_str)
    print(f1_report_str)
writer.flush()
writer.close()