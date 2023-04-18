import os
import random
import glob
from pandas.core.common import flatten

import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from basic_CNN import Net, PlantDataset, transform, batch_size, classes, image_paths

model = Net()
model.load_state_dict(torch.load(os.getcwd() + os.sep + os.pardir + os.sep + os.pardir + '/models/basic_cnn.pth'))
# print(model.eval())
data_path = os.getcwd() + os.sep + os.pardir + os.sep + os.pardir + '/data/raw/plantifydr_dataset/color'
# image_paths = []
# classes = []

# for path in glob.glob(data_path + '/*'):
    # classes.append(path.split('/')[-1])
    # image_paths.append(glob.glob(path + '/*'))

image_paths = list(flatten(image_paths))
random.shuffle(image_paths)

idx_to_class = {i:j for i,j in enumerate(classes)}
class_to_idx = {j:i for i,j in idx_to_class.items()}

test_dataset = PlantDataset(image_paths, transform=transform)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# print(images.shape)

# print(f'Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct // total} %')
