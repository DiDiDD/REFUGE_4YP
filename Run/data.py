import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

class DriveDataset(Dataset):
    def __init__(self, images_path, masks_path):

        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        """ Reading image """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        '''Normalise tensity in range [-1,-1]'''
        image = (image-127.5)/127.5
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        """ Reading mask """
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        mask = np.where(mask<128, 2, mask)
        mask = np.where(mask == 128, 1, mask)
        mask = np.where(mask>128, 0, mask)
        mask = mask.astype(np.int64)
        '''convert numpy array into tensor'''
        mask = np.expand_dims(mask, axis=0)
        mask = torch.from_numpy(mask)
        return image, mask

    def __len__(self):
        return self.n_samples
