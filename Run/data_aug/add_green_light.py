import cv2
import numpy as np
from matplotlib import pyplot as plt
from glob import glob
import os
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

create_dir('/home/mans4021/Desktop/new_data/REFUGE2/test/image/greenlight/')
images = sorted(glob(os.path.join('/home/mans4021/Desktop/new_data/REFUGE2/test/image/', '*.jpg')))
masks = sorted(glob(os.path.join('/home/mans4021/Desktop/new_data/REFUGE2/test/mask/', '*.bmp')))


for index,(x, y) in tqdm(enumerate(zip(images, masks))):
    x = cv2.imread(x)
    y = cv2.imread(y, 0)
    background = np.where(y > 129)
    for i in range(len(background[0])):
        random_number = random.randint(1,10)
        # print(background[0][i], background[1][i])
        if ((background[0][i]-256)**2 + (background[1][i]-240)**2) > 210**2 and ((background[0][i]-256)**2 + (background[1][i]-256)**2 )<=245**2:
            x[background[0][i], background[1][i], 2] /= 1.2
            x[background[0][i], background[1][i], 2].round()
            x[background[0][i], background[1][i], 1] = x[background[0][i], background[1][i], 2]+random_number
            x[background[0][i], background[1][i], 0] = x[background[0][i], background[1][i], 2]//2
    cv2.imwrite(f'/home/mans4021/Desktop/new_data/REFUGE2/test/image/greenlight/{index}.jpg', x)



