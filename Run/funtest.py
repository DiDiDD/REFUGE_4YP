import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('g0001.bmp',cv2.IMREAD_GRAYSCALE)
image = (image-127.5)/127.5
image =np.where((image != 1) & (image!=-1), 0, image)
print(image.shape)
image1 = image
# print((image[:,:,0] == image[:,:,1]).all())
print(image1.shape)
im = plt.imshow(image1)
plt.show()
