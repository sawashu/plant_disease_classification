import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.optimize import linear_sum_assignment
# from lapsolver import solve_dense
import time

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class SimpleCNNContainerConvBlocks(nn.Module):
    def __init__(self, input_channel, num_filters, kernel_size, output_dim=10):
        super(SimpleCNNContainerConvBlocks, self).__init__()
        '''
        A testing cnn container, which allows initializing a CNN with given dims
        We use this one to estimate matched output of conv blocks

        num_filters (list) :: number of convolution filters
        hidden_dims (list) :: number of neurons in hidden layers 

        Assumptions:
        i) we use only two conv layers and three hidden layers (including the output layer)
        ii) kernel size in the two conv layers are identical
        '''
        self.conv1 = nn.Conv2d(input_channel, num_filters[0], kernel_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(num_filters[0], num_filters[1], kernel_size)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x


class ModerateCNNContainerConvBlocks(nn.Module):
    def __init__(self, num_filters, output_dim=10):
        super(ModerateCNNContainerConvBlocks, self).__init__()

        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=num_filters[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[0], out_channels=num_filters[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=num_filters[1], out_channels=num_filters[2], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[2], out_channels=num_filters[3], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=num_filters[3], out_channels=num_filters[4], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[4], out_channels=num_filters[5], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.conv_layer(x)
        return x


class ModerateCNNContainerConvBlocksMNIST(nn.Module):
    def __init__(self, num_filters, output_dim=10):
        super(ModerateCNNContainerConvBlocksMNIST, self).__init__()

        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=1, out_channels=num_filters[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[0], out_channels=num_filters[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=num_filters[1], out_channels=num_filters[2], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[2], out_channels=num_filters[3], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=num_filters[3], out_channels=num_filters[4], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[4], out_channels=num_filters[5], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.conv_layer(x)
        return x


class LeNetContainer(nn.Module):
    def __init__(self, num_filters, kernel_size=5):
        super(LeNetContainer, self).__init__()
        self.conv1 = nn.Conv2d(1, num_filters[0], kernel_size, 1)
        self.conv2 = nn.Conv2d(num_filters[0], num_filters[1], kernel_size, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        #x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        #x = F.relu(x)
        return x

# CODE ADDED FOR OUR USE ----------------------------------------------------------------

class InceptionCNNContainerPlant(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.init_conv = nn.Conv2d(3, 192, 1)
        self.layer1_conv1 = nn.Conv2d(192, 128, 1)
        self.layer1_conv2 = nn.Conv2d(192, 32, 1)
        self.layer1_pool1 = nn.MaxPool2d(3,stride=1,padding=1)
        self.layer2_conv1 = nn.Conv2d(192, 64, 1)
        self.layer2_conv2 = nn.Conv2d(128,128,3,padding=1)
        self.layer2_conv3 = nn.Conv2d(32,32,5,padding=2)
        self.layer2_conv4 = nn.Conv2d(192,32,1)

        self.fc1 = nn.Linear(256*256*256, 38)
        # self.fc2 = nn.Linear(64*64*64, 16*16*16)
        # self.fc3 = nn.Linear(16*16*16, 38)

    def forward(self, x):
        # print(x.shape)
        x = self.init_conv(x)
        # print(x.shape)
        l1_o1 = F.relu(self.layer1_conv1(x))
        l1_o2 = F.relu(self.layer1_conv2(x))
        # print(self.layer1_pool1(x).shape)
        l1_o3 = F.relu(self.layer1_pool1(x))

        # print(l1_o3.shape)

        l2_o1 = F.relu(self.layer2_conv1(x))
        l2_o2 = F.relu(self.layer2_conv2(l1_o1))
        l2_o3 = F.relu(self.layer2_conv3(l1_o2))
        l2_o4 = F.relu(self.layer2_conv4(l1_o3))

        o = torch.cat((l2_o1,l2_o2,l2_o3,l2_o4),1)

        # print(o.shape)
        o = torch.flatten(o,1)
        o = self.fc1(o)
        # o = F.relu(self.fc2(o))
        # o = self.fc3(o)

        return o