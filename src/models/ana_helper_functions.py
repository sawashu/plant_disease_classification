import numpy as np
import os

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import random_split


from ana_copy_basicCNN import  Net

''' 
############################################################################################
Load Dataset and split into train and test sets:
############################################################################################

code copied and augmented from:
https://ryanwingate.com/intro-to-machine-learning/deep-learning-with-pytorch/loading-image-data-into-pytorch/

'''
# specify data directory
# data_dir = '/Users/bean/Documents/plant_disease_classification/data/raw/plantifydr_dataset'
# data_dir = os.getcwd() + os.sep + os.pardir + os.sep + os.pardir + '/data/raw/plantifydr_dataset/color'
data_dir = os.getcwd()  + '/data/raw/plantifydr_dataset/color'


# define how to transform data and convert to PyTorch tensors
# training data can be resized, flipped, etc.
train_transform = transforms.Compose([transforms.Resize((256, 256)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# validation data should not be changed other than Normalized
validation_transform = transforms.Compose([transforms.ToTensor(), 
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# load image data with datasets.ImageFolder
dataset = datasets.ImageFolder(data_dir, transform = train_transform)

# split data into train and validation data
train_data, validation_data = random_split(dataset, [0.8, 0.2])



'''
############################################################################################
Other Helper functions :
############################################################################################
'''

'''
FUNCTION : get_t_v_loader
INPUT : (int) batch_size - desired batch_size
OUTPUT : train_loader - batches of [images, labels]
       : valid_loader - batches of [images, labels]

This function passes data to DataLoader
and generates batches of images and corresponding labels
'''
def get_t_v_loader(batch_size):
        train_loader = torch.utils.data.DataLoader(train_data, 
                                                batch_size = batch_size, 
                                                shuffle = True)

        valid_loader = torch.utils.data.DataLoader(validation_data,
                                                        batch_size = batch_size,
                                                        shuffle = True)

        return(train_loader, valid_loader)

'''
FUNCTION : imshow

helper function prints images to screen

How to call:
    images, labels = next(iter(train_loader))
    imshow(images[0], normalize=False);
'''
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

'''
FUNCTION : get_mean_std
INPUT : (array) vals - values to get the mean and standard deviation of
OUTPUT : (float) mean - mean of all values in array
       : (float) std - standard deviation of all values in array
'''
def get_mean_std(vals):
    mean = np.mean(vals)
    std = np.std(vals)
    return mean, std

'''
############################################################################################

CNN Training :

We test different parameters for CNN, namely :

        batch size :
        optimizer functions used :
        learning rate :
        epoch values :

############################################################################################
'''

# batch_vals = [16]
# loss_vals = [nn.CrossEntropyLoss()] # CHECK IF WORKS LIKE THIS
# optimizer_vals = []
# learning_rate_vals = []
# epoch_vals = []

# # change number of batches
# for batch_val in batch_vals:

#     train_l, valid_l = get_t_v_loader(batch_val)

#     for loss_val in loss_vals:
         
#          for optimizer_val in optimizer_vals:
              
#               for learning_rate_val in learning_rate_vals:
                   
#                    net = Net()

         
