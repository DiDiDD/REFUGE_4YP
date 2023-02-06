# Import MNIST
from torchvision.datasets import MNIST

# Download and Save MNIST
data_train = MNIST('~/mnist_data', train=True, download=True)


import matplotlib.pyplot as plt

random_image = data_train[0][0]
random_image_label = data_train[0][1]

# Print the Image using Matplotlib
plt.imshow(random_image)
print("The label of the image is:", random_image_label)

import torch
from torchvision import transforms

data_train = torch.utils.data.DataLoader(
    MNIST(
          '~/mnist_data', train=True, download=True,
          transform = transforms.Compose([
              transforms.ToTensor()
          ])),
          batch_size=1,
          shuffle=False
          )

for batch_idx, samples in enumerate(data_train):
      print(batch_idx, samples[0].size())