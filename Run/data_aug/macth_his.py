import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import os


images0 = sorted(glob(os.path.join('/home/mans4021/Desktop/new_data/REFUGE2/train/image/', '*.jpg')))
images1 = sorted(glob(os.path.join('/home/mans4021/Desktop/new_data/REFUGE2/test/image', '*.jpg')))

hist_avg_b = np.zeros(256)
hist_avg_g = np.zeros(256)
hist_avg_r = np.zeros(256)
for i in tqdm(images0):
    x = cv2.imread(i)
    hist_b, _ = np.histogram(x[:, :, 0].flatten(), bins=256, range=[0, 256])
    hist_g, _ = np.histogram(x[:, :, 1].flatten(), bins=256, range=[0, 256])
    hist_r, _ = np.histogram(x[:, :, 2].flatten(), bins=256, range=[0, 256])

    hist_avg_b += hist_b
    hist_avg_g += hist_g
    hist_avg_r += hist_r

hist_avg_b /= 1600
hist_avg_g /= 1600
hist_avg_r /= 1600

hist_avg_b = hist_avg_b / np.sum(hist_avg_b)
hist_avg_g = hist_avg_g / np.sum(hist_avg_g)
hist_avg_r = hist_avg_r / np.sum(hist_avg_r)

cdf_avg_b = np.cumsum(hist_avg_b)
cdf_avg_g = np.cumsum(hist_avg_g)
cdf_avg_r = np.cumsum(hist_avg_r)


index = 0
for ii in tqdm(images1):
    name = ii.split("/")[-1].split(".")[0]
    img1 = cv2.imread(ii)
    hist1_b, _ = np.histogram(img1[:, :, 0].flatten(), bins=256, range=[0, 256])
    hist1_g, _ = np.histogram(img1[:, :, 1].flatten(), bins=256, range=[0, 256])
    hist1_r, _ = np.histogram(img1[:, :, 2].flatten(), bins=256, range=[0, 256])

    hist1_b = hist1_b / np.sum(hist1_b)
    hist1_g = hist1_g / np.sum(hist1_g)
    hist1_r = hist1_r / np.sum(hist1_r)


    # Compute CDFs
    cdf1_b = np.cumsum(hist1_b)
    cdf1_g = np.cumsum(hist1_g)
    cdf1_r = np.cumsum(hist1_r)

    # Compute inverse CDF of average histogram
    inv_cdf_avg_b = np.interp(cdf1_b, cdf_avg_b, np.arange(256))
    inv_cdf_avg_g = np.interp(cdf1_g, cdf_avg_g, np.arange(256))
    inv_cdf_avg_r = np.interp(cdf1_r, cdf_avg_r, np.arange(256))

    # Transform Image 1
    img1_matched = np.zeros_like(img1)


    img1_matched[:, :, 0] = np.interp(img1[:, :, 0].flatten(), np.arange(256), inv_cdf_avg_b).reshape(
        img1[:, :, 0].shape).astype(np.uint8).round()
    img1_matched[:, :, 1] = np.interp(img1[:, :, 1].flatten(), np.arange(256), inv_cdf_avg_g).reshape(
        img1[:, :, 1].shape).astype(np.uint8).round()
    img1_matched[:, :, 2] = np.interp(img1[:, :, 2].flatten(), np.arange(256), inv_cdf_avg_r).reshape(
        img1[:, :, 2].shape).astype(np.uint8).round()

    for i in range(512):
        for j in range(512):
            if img1[i,j,0] ==0 and img1[i,j,1] ==0 and img1[i,j,2] ==0:
                img1_matched[i,j,0] =0
                img1_matched[i,j,1] =0
                img1_matched[i, j, 2] = 0
    # Display the results

    tmp_image_name = f"{name}_{index}.jpg"
    image_path = os.path.join('/home/mans4021/Desktop/new_data/REFUGE2/test/image_match/', tmp_image_name)
    cv2.imwrite(image_path, img1_matched)
    index+=1

