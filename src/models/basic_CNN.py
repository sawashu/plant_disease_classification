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

batch_size = 4


# print(os.path.dirname(os.path.abspath(__file__))) 
data_path = os.getcwd() + os.sep + os.pardir + os.sep + os.pardir + '/data/raw/plantifydr_dataset/color'
print(data_path)
image_paths = []
classes = []


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
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(59536, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 38)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':

    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(10):

        running_loss = 0.0
        for i,data in enumerate(train_loader,0):

            inputs,labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print(f"Finished Training")

    PATH = './basic_cnn_pth'
    torch.save(net.state_dict(), PATH)


