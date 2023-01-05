import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from albumentations import HorizontalFlip, VerticalFlip, Rotate

""" Create a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path_train_x, path_train_y, path_val_x, path_val_y, path_test_x, path_test_y):
    train_x = sorted(glob(os.path.join(path_train_x, '*.jpg')))
    train_y = sorted(glob(os.path.join(path_train_y, '*.bmp')))
    val_x = sorted(glob(os.path.join(path_val_x, '*.jpg')))
    val_y = sorted(glob(os.path.join(path_val_y, '*.bmp')))
    test_x = sorted(glob(os.path.join(path_test_x, '*.jpg')))
    test_y = sorted(glob(os.path.join(path_test_y, '*.bmp')))

    return (train_x, train_y), (val_x, val_y), (test_x, test_y)

def augment_data(images, masks, save_path, augment=True):
    size = (512, 512)

    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        """ Extracting the name """
        name = x.split("/")[-1].split(".")[0]

        """ Reading image and mask """
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y)

        if augment == True:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented["image"]
            y2 = augmented["mask"]

            aug = Rotate(limit=45, p=1.0)
            augmented = aug(image=x, mask=y)
            x3 = augmented["image"]
            y3 = augmented["mask"]

            X = [x, x1, x2, x3]
            Y = [y, y1, y2, y3]

        else:
            X = [x]
            Y = [y]

        index = 0
        for i, m in zip(X, Y):
            i = cv2.resize(i, size)
            m = cv2.resize(m, size, interpolation= cv2.INTER_NEAREST)

            tmp_image_name = f"{name}_{index}.jpg"
            tmp_mask_name = f"{name}_{index}.bmp"

            image_path = os.path.join(save_path, "image", tmp_image_name)
            mask_path = os.path.join(save_path, "mask", tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            index += 1

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)

    """ Load the data """
    data_path_train_x = "/home/mans4021/Desktop/refuge-challenge/REFUGE2-Training/REFUGE2-Training/REFUGE1-Train-400/Images_comb"
    data_path_train_y  = "/home/mans4021/Desktop/refuge-challenge/REFUGE2-Training/REFUGE2-Training/REFUGE1-Train-400/G+N"
    data_path_val_x = "/home/mans4021/Desktop/refuge-challenge/REFUGE2-Training/REFUGE2-Training/REFUGE1-Val-400/REFUGE-Validation400"
    data_path_val_y  = "/home/mans4021/Desktop/refuge-challenge/REFUGE2-Training/REFUGE2-Training/REFUGE1-Val-400/REFUGE-Validation400-GT/Disc_Cup_Masks"
    data_path_test_x = '/home/mans4021/Desktop/refuge-challenge/REFUGE2-Training/REFUGE2-Training/REFUGE1-Test-400/Images'
    data_path_test_y  = '/home/mans4021/Desktop/refuge-challenge/REFUGE2-Training/REFUGE2-Training/REFUGE1-Test-400/G+N'
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_data(data_path_train_x, data_path_train_y,
                                                                      data_path_val_x, data_path_val_y,
                                                                      data_path_test_x, data_path_test_y)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Val: {len(val_x)} - {len(val_y)}")
    print(f"Train: {len(test_x)} - {len(test_y)}")

    """ Create directories to save the augmented data """
    create_dir("/home/mans4021/Desktop/new_data/REFUGE2/train/image/")
    create_dir("/home/mans4021/Desktop/new_data/REFUGE2/train/mask/")
    create_dir("/home/mans4021/Desktop/new_data/REFUGE2/val/image/")
    create_dir("/home/mans4021/Desktop/new_data/REFUGE2/val/mask/")
    create_dir("/home/mans4021/Desktop/new_data/REFUGE2/test/image/")
    create_dir("/home/mans4021/Desktop/new_data/REFUGE2/test/mask/")

    """ Data augmentation """
    augment_data(train_x, train_y, "/home/mans4021/Desktop/new_data/REFUGE2/train/", augment=False)
    augment_data(val_x, val_y, "/home/mans4021/Desktop/new_data/REFUGE2/val/", augment=False)
    augment_data(test_x, test_y, "/home/mans4021/Desktop/new_data/REFUGE2/test/", augment=False)
