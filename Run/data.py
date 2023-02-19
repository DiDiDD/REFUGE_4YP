import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class train_test_split(Dataset):
    def __init__(self, images_path, masks_path, get_disc=False):

        self.images_path = images_path
        self.masks_path = masks_path
        self.num_samples = len(images_path)
        self.get_disc = get_disc

    def __getitem__(self, index, get_disc=False):
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        '''Normalise tensity in range [-1,-1]'''
        image = (image-127.5)/127.5
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        image = torch.from_numpy(image)      # (3,512,512)

        """ Reading mask """
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        mask = np.where(mask < 128, 2, mask)     # cup
        mask = np.where(mask == 128, 1, mask)    # disc - cup = outer - ring
        mask = np.where(mask > 128, 0, mask)     # background
        mask = mask.astype(np.int64)
        mask = np.expand_dims(mask, axis=0)
        mask = torch.from_numpy(mask)        # (1,512,512)

        if get_disc == True:
            mask_disc = torch.where(mask == 2, 1, mask)
        return image, mask, mask_disc

    def __len__(self):
        return self.num_samples
