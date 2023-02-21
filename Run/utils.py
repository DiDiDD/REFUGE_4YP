import os
import random
import numpy as np
import torch

def seeding(seed):  # seeding the randomness
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_file(file):
    if not os.path.exists(file):
        open(file, "w")
    else:
        print(f"{file} Exists")


def train_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def segmentation_score(y_true, y_pred, num_classes):
    # returns confusion matrix (TP, FP, TN, FN) for each class, plus a combined class for class 1+2 (disc)
    if y_true.size() != y_pred.szie():
        raise DimensionError(f'Check dimensions of y_true {y_true.size()} and y_pred {y_pred.szie()}')

    smooth = 0.00001
    y_true = y_true.cpu().numpy().astype(int)
    y_pred = y_pred.cpu().numpy().astype(int)
    score_matrix = np.zeros((num_classes + 1, 5))

    for i in range(num_classes):
        tp = np.sum(np.logical_and(y_true == i, y_pred == i))
        fp = np.sum(np.logical_and(y_true != i, y_pred == i))
        tn = np.sum(np.logical_and(y_true != i, y_pred != i))
        fn = np.sum(np.logical_and(y_true == i, y_pred != i))
        accuracy = (tp + tn)/(tp+fp+tn+fn+smooth)
        precision = tp/(tp+fp+smooth)
        recall = tp/(tp+fn+smooth)
        f1 = 2*tp/(2*tp+fp+fn+smooth)
        IoU = tp/(tp+fp+fn+smooth)
        score_matrix[i] = np.array([IoU, f1, recall, precision, accuracy])
    # DISC
    tp = np.sum(np.logical_and(np.logical_or(y_true == 1, y_true == 2), np.logical_or(y_pred == 1, y_pred == 2)))
    fp = np.sum(np.logical_and(y_true == 0, np.logical_or(y_pred == 1, y_pred == 2)))
    tn = np.sum(np.logical_and(y_true == 0, y_pred == 0))
    fn = np.sum(np.logical_and(np.logical_or(y_true == 1, y_true == 2), y_pred == 0))
    accuracy = (tp + tn) / (tp + fp + tn + fn + smooth)
    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)
    f1 = 2 * tp / (2 * tp + fp + fn + smooth)
    IoU = tp / (tp + fp + fn + smooth)
    score_matrix[3] = np.array([IoU, f1, recall, precision, accuracy])

    return score_matrix


def f1_valid_score(y_true, y_pred):
    if y_true.size() != y_pred.szie():
        raise DimensionError(f'Check dimensions of y_true {y_true.size()} and y_pred {y_pred.szie()}')

    smooth = 0.00001
    y_true = y_true.cpu().numpy().astype(int)
    y_pred = y_pred.cpu().numpy().astype(int)
    score_matrix = np.zeros(4)
    for i in range(3):
        tp = np.sum(np.logical_and(y_true == i, y_pred == i))
        fp = np.sum(np.logical_and(y_true != i, y_pred == i))
        fn = np.sum(np.logical_and(y_true == i, y_pred != i))
        f1 = 2*tp/(2*tp+fp+fn+smooth)
        score_matrix[i] = f1
    tp = np.sum(np.logical_and(np.logical_or(y_true == 1, y_true == 2), np.logical_or(y_pred == 1, y_pred == 2)))
    fp = np.sum(np.logical_and(y_true == 0, np.logical_or(y_pred == 1, y_pred == 2)))
    fn = np.sum(np.logical_and(np.logical_or(y_true == 1, y_true == 2), y_pred == 0))
    f1 = 2 * tp / (2 * tp + fp + fn + smooth)
    score_matrix[3] = f1

    return score_matrix


def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)                # (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  # (512, 512, 3)
    return mask
