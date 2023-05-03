from data_aug.data import train_test_split
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import torch
from UNET.UNet_model import UNet
from utils import create_dir, segmentation_score, mask_parse, choose_test_set
import argparse
from torch.utils.tensorboard import SummaryWriter
from UTNET._UTNET_model import UTNet
from monai.networks.nets import SwinUNETR
from Swin_UNETR.swin_unetr_model_with_batch_in_trans import SwinUNETR_batch
from Swin_UNETR.swin_unetr_model_with_instance import SwinUNETR_instance
import json

parser = argparse.ArgumentParser(description='Specify Parameters')

parser.add_argument('test_data', metavar='test_data', type=int, help='Specify which test_data')
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
test_data_num = args.test_data
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

utnet = UTNet(in_chan=3, num_classes=3, base_chan=base_c, norm_name=norm_name)

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


'''Tensorboard'''
# data_save_path = f'/home/mans4021/Desktop/new_data/REFUGE2/test/1600_{model_text}_{norm_name}_lr_{lr}_bs_{batch_size}/'

device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')
create_dir(data_save_path+f'results{test_data_num}/')
writer = SummaryWriter(data_save_path, comment = f'_set{test_data_num}')

if __name__ == "__main__":
    """ Load dataset """
    test_y = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/mask/*"))
    test_x = choose_test_set(test_data_num)

    test_dataset = train_test_split(test_x, test_y)
    dataset_size = len(test_x)
    checkpoint_path_lowloss = data_save_path + f'Checkpoint/lr_{lr}_bs_{batch_size}_lowloss.pth'
    checkpoint_path_final = data_save_path + f'Checkpoint/lr_{lr}_bs_{batch_size}_final.pth'
    create_dir(data_save_path + f'results{test_data_num}')

    """ Load the checkpoint """
    model.to(device)
    model.load_state_dict(torch.load(checkpoint_path_lowloss, map_location=device))
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
            if i%10==0:
                cv2.imwrite(data_save_path+f'results{test_data_num}/{i}.png', cat_images)

    np.save(data_save_path+f'results{test_data_num}/'+f'test_score_{test_data_num}', metrics_score)
    
    f1_record = metrics_score[:, :, 1]
    f1_mean = metrics_score.mean(axis=0)
    f1_std = np.std(f1_record, axis=0)

    iou_record = metrics_score[:, :, 0]
    iou_mean = metrics_score.mean(axis=0)
    iou_std = np.std(f1_record, axis=0)

    recall_record = metrics_score[:, :, 2]
    recall_mean = metrics_score.mean(axis=0)
    recall_std = np.std(f1_record, axis=0)

    precison_record = metrics_score[:, :, 3]
    precision_mean = metrics_score.mean(axis=0)
    precision_std = np.std(f1_record, axis=0)

    accuracy_record = metrics_score[:, :, 4]
    accuracy_mean = metrics_score.mean(axis=0)
    accuracy_std = np.std(f1_record, axis=0)

    test_report_str = data_save_path + f' test results {test_data_num} are:'
    # test_report_str += f'\nBackground F1 score is {f1_mean[0,1]:3f} +- {f1_std[0]:3f}'
    test_report_str += f'\nOuter Ring F1 score is {f1_mean[1,1]:3f} +- {f1_std[1]:3f}'
    test_report_str += f'\nCup F1 score is {f1_mean[2,1]:3f} +- {f1_std[2]:3f}'
    test_report_str += f'\nDisc F1 score is {f1_mean[3,1]:3f} +- {f1_std[3]:3f}'
    
    test_report_str += f'\nOuter Ring recall score is {recall_mean[1,2]:3f} +- {recall_std[1]:3f}'
    test_report_str += f'\nCup recall score is {recall_mean[2,2]:3f} +- {recall_std[2]:3f}'
    test_report_str += f'\nDisc recall score is {recall_mean[3,2]:3f} +- {recall_std[3]:3f}'
    
    test_report_str += f'\nOuter Ring precision score is {precision_mean[1,3]:3f} +- {precision_std[1]:3f}'
    test_report_str += f'\nCup precision score is {precision_mean[2,3]:3f} +- {precision_std[2]:3f}'
    test_report_str += f'\nDisc precision score is {precision_mean[3,3]:3f} +- {precision_std[3]:3f}'
    
    test_report_str += f'\nOuter Ring IOU score is {iou_mean[1,0]:3f} +- {iou_std[1]:3f}'
    test_report_str += f'\nCup IOU score is {iou_mean[2,0]:3f} +- {iou_std[2]:3f}'
    test_report_str += f'\nDisc IOU score is {iou_mean[3,0]:3f} +- {iou_std[3]:3f}'
    
    test_report_str += f'\nOuter Ring accuracy score is {accuracy_mean[1,4]:3f} +- {accuracy_std[1]:3f}'
    test_report_str += f'\nCup accuracy score is {accuracy_mean[2,4]:3f} +- {accuracy_std[2]:3f}'
    test_report_str += f'\nDisc accuracy score is {accuracy_mean[3,4]:3f} +- {accuracy_std[3]:3f}'
    
    writer.add_text('Test f1 score', test_report_str)
    for i in range(4):
    #     if test_data_num < 4 or test_data_num >18:
    #         test_data_num_set = test_data_num
    #     else:
    #         test_data_num_set = 4

        writer.add_scalar(f'Test score {test_data_num}', f1_mean[i,1], i)
        # f'Test IOU score {test_data_num}': iou_mean[i, 0],
        # f'Test recall score {test_data_num}': recall_mean[i, 2],
        # f'Test precision score {test_data_num}': precision_mean[i, 3],
        # f'Test Accuracy score {test_data_num}': accuracy_mean[i, 4]},
        # i)
    print(test_report_str)
writer.flush()
writer.close()

