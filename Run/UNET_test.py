import os, time
from operator import add
from data import DriveDataset
from torch.utils.data import DataLoader
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import torch
from torchmetrics.functional.classification import multiclass_jaccard_index, multiclass_f1_score, multiclass_precision, multiclass_recall, multiclass_accuracy
from UNet_model import build_unet
from utils import create_dir, seeding
import argparse
parser = argparse.ArgumentParser(description='Specify Parameters')
parser.add_argument('lr', metavar='lr', type=float, help='Specify learning rate')
parser.add_argument('b_s', metavar='b_s', type=int, help='Specify bach size')
parser.add_argument('gpu_index', metavar='gpu_index', type=int, help='Specify which gpu to use')
args = parser.parse_args()
lr = args.lr
batch_size = args.b_s
gpu_index = args.gpu_index
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('/home/mans4021/Desktop/new_data/REFUGE2/test/test_score/', comment= f'UNET_lr_{lr}_bs_{batch_size}', filename_suffix= f'UNET_lr_{lr}_bs_{batch_size}')

def calculate_metrics(y_pred, y_true):
    score_jaccard_2 = multiclass_jaccard_index(y_pred, y_true, num_classes=3, average=None)[2]
    score_f1_2 = multiclass_f1_score(y_pred, y_true, num_classes=3, average=None)[2]
    score_precision_2 = multiclass_precision(y_pred, y_true, num_classes=3, average=None)[2]
    score_recall_2 = multiclass_recall(y_pred, y_true, num_classes=3, average=None)[2]
    score_acc_2 = multiclass_accuracy(y_pred, y_true, num_classes=3, average=None)[2]

    y_pred_trans = torch.where(y_pred==2, 1, y_pred)
    y_true_trans = torch.where(y_true==2, 1, y_true)

    score_jaccard_0_1 = multiclass_jaccard_index(y_pred_trans, y_true_trans, num_classes=2, average=None)
    score_f1_0_1 = multiclass_f1_score(y_pred_trans, y_true_trans, num_classes=2, average=None)
    score_precision_0_1 = multiclass_precision(y_pred_trans, y_true_trans, num_classes=2, average=None)
    score_recall_0_1 = multiclass_recall(y_pred_trans, y_true_trans, num_classes=2, average=None)
    score_acc_0_1 = multiclass_accuracy(y_pred_trans, y_true_trans, num_classes=2, average=None)

    score_jaccard = torch.cat((score_jaccard_0_1, score_jaccard_2))
    score_f1 = torch.cat((score_f1_0_1, score_f1_2))
    score_recall = torch.cat((score_recall_0_1, score_recall_2))
    score_precision = torch.cat((score_precision_0_1, score_precision_2))
    score_acc = torch.cat((score_acc_0_1, score_acc_2))

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc]

def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    return mask

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Folders """
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
    checkpoint_path = f'/home/mans4021/Desktop/checkpoint/checkpoint_refuge_unet.pth/lr_{lr}_bs_{batch_size}.pth'

    """ Load the checkpoint """
    device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')

    model = build_unet()
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]

    for i in tqdm(range(dataset_size)):
        with torch.no_grad():
            '''Prediction'''
            image = test_dataset[i][0].unsqueeze(0)                       # (1,3,512,512)
            ori_mask = test_dataset[i][1].squeeze(0).cuda()               # (512,512)
            pred_y = model(image.cuda()).squeeze(0)                       # (3,512,512)
            pred_y = torch.softmax(pred_y, dim=0)                         # (3,512,512)
            pred_mask = torch.argmax(pred_y, dim=0).type(torch.int64)     # (512, 512)
            score = calculate_metrics(pred_mask, ori_mask)
            metrics_score = list(map(add, metrics_score, score))
            pred_mask = pred_mask.cpu().numpy()        ## (512, 512)
            ori_mask = ori_mask.cpu().numpy()
            '''Scale value back to image'''
            pred_mask = np.where(pred_mask==2, 255, pred_mask)
            pred_mask = np.where(pred_mask==1, 128, pred_mask)
            ori_mask = np.where(ori_mask==2, 255, ori_mask)
            ori_mask = np.where(ori_mask==1, 128, ori_mask)
            image= image*127.5 + 127.5
            """ Saving masks """
            ori_mask = mask_parse(ori_mask)
            pred_mask = mask_parse(pred_mask)
            line = np.ones((512,20,3)) * 255

            cat_images = np.concatenate(
                [image.squeeze().permute(1,2,0), line, ori_mask, line, pred_mask], axis=1
            )
            cv2.imwrite(f"/home/mans4021/Desktop/new_data/REFUGE2/test/results/lr_{lr}_bs_{batch_size}/{i}.png", cat_images)

    jaccard = metrics_score[0]/len(test_x)
    f1 = metrics_score[1]/len(test_x)
    recall = metrics_score[2]/len(test_x)
    precision = metrics_score[3]/len(test_x)
    acc = metrics_score[4]/len(test_x)
    for x in range(len(jaccard)):
        writer.add_scalar('Jaccard Score', jaccard[x],x)
        writer.add_scalar('F1 Score', f1[x], x)
        writer.add_scalar('Recall Score', recall[x], x)
        writer.add_scalar('Precision Score', precision[x], x)
        writer.add_scalar('Accuracy Score', acc[x], x)
        print('Jaccard Score', jaccard[x], x)
        print('F1 Score', f1[x], x)
        print('Recall Score', recall[x], x)
        print('Precision Score', precision[x], x)
        print('Accuracy Score', acc[x], x)