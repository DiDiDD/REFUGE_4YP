import os, time
from operator import add
from data import DriveDataset
from torch.utils.data import DataLoader
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import torch
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassF1Score, MulticlassPrecision, MulticlassRecall, MulticlassAccuracy
from UNet_model import build_unet
from utils import create_dir, seeding
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('/home/mans4021/Desktop/REFUGE_4YP/Board_val/')

def calculate_metrics(y_true, y_pred):
    score_jaccard = MulticlassJaccardIndex(y_pred, y_true, num_classes=3)
    score_f1 = MulticlassF1Score(y_pred, y_true, num_classes=3)
    score_precision = MulticlassPrecision(y_pred, y_true, num_classes=3)
    score_recall = MulticlassRecall(y_pred, y_true, num_classes=3)
    score_acc = MulticlassAccuracy(y_pred, y_true, nuim_classes=3)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc]

def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    return mask

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Folders """
    create_dir("/home/mans4021/Desktop/new_data/REFUGE2/test/results/")

    """ Load dataset """
    test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image/*"))
    test_y = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/mask/*"))
    test_dataset = DriveDataset(test_x, test_y)

    """ Hyperparameters """
    dataset_size = len(test_x)
    H = 512
    W = 512
    size = (W, H)
    checkpoint_path = "/home/mans4021/Desktop/checkpoint/checkpoint_refuge_unet.pth"

    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_unet()
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]

    for i in tqdm(range(dataset_size)):
        with torch.no_grad():
            '''Prediction'''
            image = test_dataset[i][0].unsqueeze(0)                       # (1,3,512,512)
            ori_mask = test_dataset[i][1].squeeze(0)                        # (512,512)
            pred_y = model(image.cuda()).squeeze(0)                       # (3,512,512)
            pred_y = torch.softmax(pred_y, dim=0)                         # (3,512,512)
            pred_mask = torch.argmax(pred_y, dim=0)                        # (512, 512)
            #score = calculate_metrics(pred_mask, ori_mask)
            #metrics_score = list(map(add, metrics_score, score))
            pred_mask = pred_mask.cpu().numpy()        ## (512, 512)
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
            cv2.imwrite(f"/home/mans4021/Desktop/new_data/REFUGE2/test/results/{i}.png", cat_images)

    '''
    jaccard = metrics_score[0]/len(test_x)
    f1 = metrics_score[1]/len(test_x)
    recall = metrics_score[2]/len(test_x)
    precision = metrics_score[3]/len(test_x)
    acc = metrics_score[4]/len(test_x)
    print(f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f}")
'''
