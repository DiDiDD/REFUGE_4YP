import cv2
import matplotlib.pyplot as plt
import numpy as np

mask = cv2.imread('g0001.bmp',cv2.IMREAD_GRAYSCALE)
mask = np.where(mask == 0, 2, mask)
mask = np.where(mask == 128, 1, mask)
mask = np.where(mask == 255, 0, mask)
mask = np.expand_dims(mask, axis=0)
print(mask.shape)
image1 = mask
# print((image[:,:,0] == image[:,:,1]).all())
print(image1.shape)
# im = plt.imshow(image1)
# plt.show()
