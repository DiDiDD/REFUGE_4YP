import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)



def load_data(path_test_x):
    test_x  = sorted(glob(os.path.join(path_test_x, '*.jpg')))
    return test_x


def intensity_change(im, channel: int, intensity_factor):
    channel_slice = im[:,:,channel]
    channel_slice = np.clip(channel_slice * intensity_factor, 0, 255)
    im[:,:,channel] = channel_slice

    return im



if __name__ == "__main__":
    for x in ['b', 'r', 'g']:
        for intensity_factor in [1.1, 1.2, 1.3, 1.4, 1.5]:
            if x == 'r':
                channel_num = 2
            elif x == 'g':
                channel_num = 1
            elif x == 'b':
                channel_num = 0


            processed_test_x_path = "/home/mans4021/Desktop/new_data/REFUGE2/test/image/"
            path_to_save_newdata = f"/home/mans4021/Desktop/new_data/REFUGE2/test/image in change/image_{x}_{intensity_factor}/"
            create_dir(path_to_save_newdata)
            test_x = load_data(processed_test_x_path)

            index = 0
            for i in tqdm(test_x):
                name = i.split("/")[-1].split(".")[0]
                i = cv2.imread(i, cv2.IMREAD_COLOR)
                m = intensity_change(i, channel_num, intensity_factor)
                tmp_image_name = f"{name}_{index}.jpg"

                image_path = os.path.join(path_to_save_newdata, tmp_image_name)
                cv2.imwrite(image_path, m)

                index += 1