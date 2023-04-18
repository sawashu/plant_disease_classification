
import numpy as np

import torch
import torch.nn as nn

import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize

from Ana_Basic_CNN import MyCNN

from utils import load_data

np.random.seed(2023)

batch_size = 32

# load MNIST dataset
train_dataset, val_dataset, test_dataset, train_loader, valid_loader, test_loader = load_data(batch_size)

#########################################################################################################
# Run Basic CNN
#########################################################################################################
# Max Epochs
N = 1

# initialize MLP class
simple_cnn_model = MyCNN(max_epochs = N)

# initialize optimizer
optimizer = torch.optim.SGD(simple_cnn_model.parameters(), lr = 0.01)

# print training loss, error rate for each epoch
print(f'CNN model with SGD, learning rate 0.1:')
print('* * * * * TRAINING * * * * *')

# fit model with training data
cnn_t_loss, cnn_t_error_rate = simple_cnn_model.fit(train_loader = train_loader, criterion = nn.CrossEntropyLoss(), optimizer = optimizer)

# test model with test data
test_loss, test_error = simple_cnn_model.predict(test_loader = test_loader, criterion = nn.CrossEntropyLoss())

print('* * * * * TESTING * * * * *')
print(f'Loss = {test_loss}  error rate = {test_error}')
print(' ')