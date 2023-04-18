import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


class MyCNN(nn.Module):

    def __init__(self, max_epochs):

        super(MyCNN, self).__init__()

        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3, stride=1, padding=0, dilation=1)
        # self.activate = nn.ReLU()
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # self.drop = nn.Dropout(p = 0.5)
        # self.flat = nn.Flatten()
        # self.linear1 = nn.Linear(3380, 128)
        # self.linear2 = nn.Linear(128, 10)

        self.max_epochs = max_epochs

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

        # x = self.conv1(x)
        # x = self.activate(x)
        # x = self.pool1(x)
        # x = self.drop(x)
        # x = self.flat(x)
        # x = self.linear1(x) 
        # x = self.activate(x)
        # x = self.drop(x)
        # x = self.linear2(x)

        # return x

    # def forward(self, x):
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
    
    def fit(self, train_loader, criterion, optimizer):

        prev_loss = 0
        # Epoch loop
        for i in range(self.max_epochs):

            total_loss, num_correct, num_samples = 0.0, 0, 0
            # Mini batch loop
            for j,(images,labels) in enumerate(train_loader):

                if j < 10:
                    # Forward pass
                    out = self(images)
                    loss = criterion(out, labels)

                    # Backward pass and optimize 
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Track the loss and error rate
                    total_loss += loss.item()

                    num_correct += (out.argmax(-1) == labels).sum()
                    num_samples += len(labels)

            error_rate = 1 - (num_correct / num_samples)

            # Print/return training loss and error rate in each epoch
            print(f'Epoch {i}: loss = {total_loss}     error rate = {error_rate}')

            # check if training has converged
            if prev_loss == 0:
                prev_loss = total_loss
            elif np.abs(prev_loss - total_loss)/num_samples < 0.001:
                break

    def predict(self, test_loader, criterion):

        five = 0
        wrong_images_len = 0
        wrong_images = []
        wrong_labels = []
        correct_labels = []

        total_loss, num_correct, num_samples = 0.0, 0, 0

        with torch.no_grad(): # no backprop step so turn off gradients
            for j,(images,labels) in enumerate(test_loader):

                if j < 10:

                    # Compute prediction output and loss
                    out = self(images)
                    loss = criterion(out, labels)

                    # Measure loss and error rate and record
                    total_loss += loss.item()

                    num_correct += (out.argmax(-1) == labels).sum()
                    num_samples += len(labels)

                    # save index of incorrectly labeled images
                    if five <= 5:
                        for i, out_label, true_label in zip(range(len(labels)), out.argmax(-1), labels):
                            if out_label != true_label:
                                wrong_images.append(images[i].numpy())
                                wrong_labels.append(out_label)
                                correct_labels.append(true_label)
                                wrong_images_len += 1
                                five += 1

            # plot 5 random images incorrectly predicted
            # give correct label and label MyCNN predicted

            fig, ax = plt.subplots(1, 5, figsize = (8, 12))

            for i in range(5):
                img  = wrong_images[i]
                w_lab = wrong_labels[i]
                t_lab = correct_labels[i]
                ax[i].set_title(f'True Label: {t_lab}')
                ax[i].set_xlabel(f'MyCNN Label: {w_lab}')
                ax[i].imshow(img.squeeze())

            fig.savefig('Five_Incorrect_Images.png')   # save the figure to file
            plt.close(fig)    # close the figure window
            fig.show
            
            err_rate = 1 - (num_correct / num_samples)

        # Print/return test loss and error rate
        return(total_loss, err_rate)
    

# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#         self.init_conv = nn.Conv2d(3, 192, 1)
#         self.layer1_conv1 = nn.Conv2d(192, 128, 1)
#         self.layer1_conv2 = nn.Conv2d(192, 32, 1)
#         self.layer1_pool1 = nn.MaxPool2d(3,stride=1,padding=1)
#         self.layer2_conv1 = nn.Conv2d(192, 64, 1)
#         self.layer2_conv2 = nn.Conv2d(128,128,3,padding=1)
#         self.layer2_conv3 = nn.Conv2d(32,32,5,padding=2)
#         self.layer2_conv4 = nn.Conv2d(192,32,1)

#         self.fc1 = nn.Linear(256*256*256, 38)
#         # self.fc2 = nn.Linear(64*64*64, 16*16*16)
#         # self.fc3 = nn.Linear(16*16*16, 38)

#     def forward(self, x):
#         # print(x.shape)
#         x = self.init_conv(x)
#         # print(x.shape)
#         l1_o1 = F.relu(self.layer1_conv1(x))
#         l1_o2 = F.relu(self.layer1_conv2(x))
#         # print(self.layer1_pool1(x).shape)
#         l1_o3 = F.relu(self.layer1_pool1(x))

#         # print(l1_o3.shape)

#         l2_o1 = F.relu(self.layer2_conv1(x))
#         l2_o2 = F.relu(self.layer2_conv2(l1_o1))
#         l2_o3 = F.relu(self.layer2_conv3(l1_o2))
#         l2_o4 = F.relu(self.layer2_conv4(l1_o3))

#         o = torch.cat((l2_o1,l2_o2,l2_o3,l2_o4),1)

#         # print(o.shape)
#         o = torch.flatten(o,1)
#         o = self.fc1(o)
#         # o = F.relu(self.fc2(o))
#         # o = self.fc3(o)

#         return o

# if __name__ == '__main__':
    
#     device = "mps" if torch.backends.mps.is_available() else "cpu"
#     device = torch.device(device)
#     print(f"Using device: {device}")

#     net = Net().to(device)

#     criterion = nn.CrossEntropyLoss()
#     # optimizer = optim.Adam(net.parameters(), lr=0.01)
#     optimizer = optim.SGD(net.parameters(), lr=1e-5, momentum=0.9)

#     for epoch in range(1):

#         running_loss = 0.0
#         for i,data in enumerate(train_loader,0):

#             inputs,labels = data
#             inputs,labels = inputs.to(device),labels.to(device)

#             optimizer.zero_grad()

#             outputs = net(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()

#             if i % 100 == 99:
#                 print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
#                 running_loss = 0.0

#     print(f"Finished Training the Inception")

#     PATH = './inception_net.pth'
#     torch.save(net.state_dict(), PATH)