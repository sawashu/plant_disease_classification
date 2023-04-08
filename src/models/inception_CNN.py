import os
import torch
import torchvision
import torchvision.transforms as transforms
from pandas.core.common import flatten
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import glob
import random
import cv2

import copy
    
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 2

# trainset = torchvision.datasets.CIFAR10(root='./raw', train=True,
#                                         download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#                                           shuffle=True, num_workers=2)

# testset = torchvision.datasets.CIFAR10(root='./raw', train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                          shuffle=False, num_workers=2)

data_path = os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + os.pardir) + '/data/raw/plantifydr_dataset/color'

image_paths = []
classes = []

print(data_path)


for path in glob.glob(data_path + '/*'):
    classes.append(path.split('/')[-1])
    image_paths.append(glob.glob(path + '/*'))

image_paths = list(flatten(image_paths))
random.shuffle(image_paths)

# print(image_paths)

train_image_paths,valid_image_paths = image_paths[:int(len(image_paths) * 0.8)],image_paths[int(len(image_paths) * 0.8):]

idx_to_class = {i:j for i,j in enumerate(classes)}
class_to_idx = {j:i for i,j in idx_to_class.items()}

class PlantDataset(Dataset):
    def __init__(self, image_paths, transform=False):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        
        image_filepath = self.image_paths[index]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = image_filepath.split('/')[-2]
        label = class_to_idx[label]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

train_dataset = PlantDataset(train_image_paths, transform=transform)
valid_dataset = PlantDataset(valid_image_paths, transform=transform)

# skipped test images for now 



# def visualize_image(dataset, idx = 0, samples = 10, cols = 5, random_img = False):

#     dataset = copy.deepcopy(dataset)
#     dataset.transform = transform
#     rows = samples // cols

#     figure, ax = plt.subplots(rows, cols, figsize=(15, 8))
#     for i in range(samples):
#         if random_img:
#             idx = np.random.randint(1, len(train_image_paths))
#         images, lab = dataset[idx]
#         # print(lab)
#         ax.ravel()[i].imshow(images[2])
#         ax.ravel()[i].set_axis_off()
#         ax.ravel()[i].set_title(idx_to_class[lab])

#     plt.tight_layout(pad=1)
#     plt.show()

# visualize_image(train_dataset,random_img=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size,shuffle=True)

# print(next(iter(train_loader))[0].shape)

class Net(nn.Module):
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

if __name__ == '__main__':
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")

    net = Net().to(device)

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(net.parameters(), lr=0.01)
    optimizer = optim.SGD(net.parameters(), lr=1e-5, momentum=0.9)

    for epoch in range(1):

        running_loss = 0.0
        for i,data in enumerate(train_loader,0):

            inputs,labels = data
            inputs,labels = inputs.to(device),labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print(f"Finished Training the Inception")

    PATH = './inception_net.pth'
    torch.save(net.state_dict(), PATH)