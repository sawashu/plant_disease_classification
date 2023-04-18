################################
# DO NOT EDIT THE FOLLOWING CODE
################################
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

#####################
# ADD YOUR CODE BELOW
#####################

## Testing: small batch:
# print(f'train d = {train_dataset.__sizeof__}, train l = {len(train_loader)}')
# train_dataset = train_dataset[:50, :]
# val_dataset = val_dataset[:50, :]
# test_dataset = test_dataset[:50, :]

# train_loader = train_loader[:10]
# valid_loader = valid_loader[:10]
# test_loader = test_loader[:10]

# Max Epochs
N = 3

# initialize MLP class
model = MyCNN(max_epochs = N)

# initialize optimizer
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

# print training loss, error rate for each epoch
print(f'CNN model with SGD, learning rate 0.1:')
print('* * * * * TRAINING * * * * *')

# fit model with training data
model.fit(train_loader = train_loader, criterion = nn.CrossEntropyLoss(), optimizer = optimizer)

# test model with test data
test_loss, test_error = model.predict(test_loader = test_loader, criterion = nn.CrossEntropyLoss())

print('* * * * * TESTING * * * * *')
print(f'Loss = {test_loss}  error rate = {test_error}')
print(' ')