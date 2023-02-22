
# import packages
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split

import matplotlib.pyplot as plt
import numpy as np


# specify data directory
data_dir = '/Users/bean/Documents/plant_disease_classification/data/raw/plantifydr_dataset'
batch_size = 32

# define how to transform data and convert to PyTorch tensors
# training data can be resized, flipped, etc.
train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# validation data should not be changed other than Normalized
validation_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# load image data with datasets.ImageFolder
dataset = datasets.ImageFolder(data_dir, transform = transform)

# split data into train and validation data
train_data, validation_data = random_split(dataset, [0.8, 0.2])



# pass data to DataLoader
# generates batches of images and corresponding labels
dataloader = torch.utils.data.DataLoader(dataset, 
                                         batch_size = batch_size, 
                                         shuffle = True)


# # loop through dataloader to get single batches
# for images, labels in dataloader:
#     pass


#  helper function prints images to screen
def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax

images, labels = next(iter(dataloader))
print(imshow(images[0], normalize=False))