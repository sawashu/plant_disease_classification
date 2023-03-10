
# import packages
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os

import matplotlib.pyplot as plt
import numpy as np

    
# def __init__(self, batch_size):
#     self.batch_size = batch_size

    
# https://ryanwingate.com/intro-to-machine-learning/deep-learning-with-pytorch/loading-image-data-into-pytorch/
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

batch_size = 4
# pass data to DataLoader
# generates batches of images and corresponding labels
train_loader = torch.utils.data.DataLoader(train_data, 
                                                batch_size = batch_size, 
                                                shuffle = True)

valid_loader = torch.utils.data.DataLoader(validation_data,
                                                        batch_size = batch_size,
                                                        shuffle = True)

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
        # calling imshow
        # images, labels = next(iter(train_loader))
        # imshow(images[0], normalize=False);


# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# define CNN
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(59536, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 38)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':

    net = Net()

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    # optimizer = optim.Adam(net.parameters(), lr=0.001)
    

    # train the network
    for epoch in range(10):

        running_loss = 0.0
        for i,data in enumerate(train_loader,0):
            # get the inputs, data is a list of [inputs, labels]
            inputs,labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print(f"Finished Training")

    PATH = './ana_copy_basicCNN_pth'
    torch.save(net.state_dict(), PATH)