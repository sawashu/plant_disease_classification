import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


class MyCNN(nn.Module):

    def __init__(self):

        super(MyCNN, self).__init__()

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
    
    def fit(self, images, labels, criterion, optimizer):

        optimizer.zero_grad()

        out = self(images)
        loss = criterion(out, labels)

        loss.backward()
        optimizer.step()

        return(loss.item(), (out.argmax(-1) == labels).sum())

    def val_predict(self, images, labels, criterion):

        out = self(images)
        loss = criterion(out, labels)

        return (loss.item(), (out.argmax(-1) == labels).sum())

    def predict(self, test_loader, criterion):

        five = True

        total_loss, num_correct, num_samples = 0.0, 0, 0

        with torch.no_grad(): # no backprop step so turn off gradients
            for j,(images,labels) in enumerate(test_loader):

                if j > 2 :
                    break
                else:

                    # Compute prediction output and loss
                    out = self(images)
                    loss = criterion(out, labels)

                    if five:
                        five_images = images[:5]
                        five_out = out[:5]
                        five_labels = labels[:5]
                        five = False

                        # plot five images from generator
                        num_row = 1 
                        num_col = 5 
                        fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
                        for im in range(len(five_images)):
                            ax = axes[im%num_col]
                            ax.imshow(five_images[im].detach().numpy().reshape(28,28), cmap='gray')

                            plt.savefig(f'five_images.png', bbox_inches='tight')
                            plt.close()

                    # Measure loss and error rate and record
                    total_loss += loss.item()

                    num_correct += (out.argmax(-1) == labels).sum()
                    num_samples += len(labels)
            
            err_rate = 1 - (num_correct / num_samples)

        # Print/return test loss and error rate
        return(total_loss, err_rate)