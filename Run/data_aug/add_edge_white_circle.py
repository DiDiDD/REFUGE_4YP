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


def draw_white_circle(image):
    center_coordinates = (480, 250)
    axesLength = tuple(np.random.randint(low=20, high=37, size=2, dtype=int))
    angle = np.random.randint(low=0, high=180, size=1, dtype=int).item()
    startAngle = 0
    endAngle = 360
    color = (255, 255, 255)
    thickness = -1
    image = cv2.ellipse(image, center_coordinates, axesLength,
                        angle, startAngle, endAngle, color, thickness)

    return image


if __name__ == "__main__":
    processed_test_x_path = "/home/mans4021/Desktop/new_data/REFUGE2/test/image/"
    path_to_save_newdata = "/home/mans4021/Desktop/new_data/REFUGE2/test/image_with_edge_white_circle/"
    create_dir(path_to_save_newdata)
    test_x = load_data(processed_test_x_path)

    index = 0
    for i in tqdm(test_x):
        name = i.split("/")[-1].split(".")[0]
        i = cv2.imread(i, cv2.IMREAD_COLOR)
        m = draw_white_circle(i)
        tmp_image_name = f"{name}_{index}.jpg"

        image_path = os.path.join(path_to_save_newdata, tmp_image_name)
        cv2.imwrite(image_path, m)

        index += 1