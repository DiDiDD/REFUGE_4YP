import os
import random
import numpy as np
import torch
from torchmetrics.functional.classification import multiclass_jaccard_index, multiclass_f1_score, multiclass_precision, multiclass_recall, multiclass_accuracy

""" Seeding the randomness. """
def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

""" Create a directory. """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

""" Calculate the time taken """
def train_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def calculate_metrics(y_pred, y_true):
    score_jaccard_2 = multiclass_jaccard_index(y_pred, y_true, num_classes=3, average=None)[2].unsqueeze(0)    # tensor size==0, so we need to unsqueeze
    score_f1_2 = multiclass_f1_score(y_pred, y_true, num_classes=3, average=None)[2].unsqueeze(0)
    score_precision_2 = multiclass_precision(y_pred, y_true, num_classes=3, average=None)[2].unsqueeze(0)
    score_recall_2 = multiclass_recall(y_pred, y_true, num_classes=3, average=None)[2].unsqueeze(0)
    score_acc_2 = multiclass_accuracy(y_pred, y_true, num_classes=3, average=None)[2].unsqueeze(0)

    y_pred_trans = torch.where(y_pred == 2, 1, y_pred)
    y_true_trans = torch.where(y_true == 2, 1, y_true)

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


def data_aug(data_x, data_y, seed_num):
    torch.manual_seed(seed_num)
    angle = torch.rand() * 360

