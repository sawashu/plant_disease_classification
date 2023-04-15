'''

This code:

* uploads the already split dataset into the corresponding training, testing, and validation sets

* augments training data with AutoAugment

* transforms all datasets as needed (resizing, normalization, ToTensor)

* generates the train and validation loader

'''

# import packages
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import os


train_dir = os.getcwd()  + '/train_val_test_split/train'
val_dir = os.getcwd() + '/train_val_test_split/val'
test_dir = os.getcwd() + '/train_val_test_split/test'

# transform training data with AutoAugment and convert to PyTorch tensors
train_transform = transforms.Compose([transforms.Resize((256, 256)),
                                            transforms.AutoAugment(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# validation and test data should not be changed other than Normalized
val_test_transform = transforms.Compose([transforms.Resize((256, 256)),
                                           transforms.ToTensor(), 
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# load image data with datasets.ImageFolder
train_dataset = datasets.ImageFolder(train_dir, transform = train_transform)
val_dataset = datasets.ImageFolder(val_dir, transform = val_test_transform)
test_dataset = datasets.ImageFolder(test_dir, transform = val_test_transform)

batch_size = 32

# generates batches of images and corresponding labels
train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                batch_size = batch_size, 
                                                shuffle = True)

valid_loader = torch.utils.data.DataLoader(val_dataset,
                                                        batch_size = batch_size,
                                                        shuffle = True)