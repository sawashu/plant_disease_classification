#########################################################################################################
# IMPORTS
#########################################################################################################

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize

from Ana_Basic_CNN import MyCNN
from utils import load_data

#########################################################################################################
# Initialization
#########################################################################################################

np.random.seed(2023)
batch_size = 32
Max_epochs = 1

# load dataset
train_dataset, val_dataset, test_dataset, train_loader, valid_loader, test_loader = load_data(batch_size)

# models
cnn = MyCNN()
# fedMA_cnn = 

# optimizers
cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr = 0.01)
# fedMA_cnn_optimizer = torch.optim.SGD(fedMA_cnn.parameters(), lr = 0.01)

# training loss, error rate
cnn_loss_err = np.ndarray(())
#########################################################################################################

#########################################################################################################
for i in range(Max_epochs):



# print training loss, error rate for each epoch
print(f'CNN model with SGD, learning rate 0.1:')
print('* * * * * TRAINING * * * * *')



# fit model with training data
cnn_t_loss, cnn_t_error_rate, cnn_v_loss, cnn_v_error_rate = simple_cnn_model.fit(train_loader = train_loader, criterion = nn.CrossEntropyLoss(), optimizer = optimizer)

# test model with test data
test_loss, test_error = simple_cnn_model.predict(test_loader = test_loader, criterion = nn.CrossEntropyLoss())

print('* * * * * TESTING * * * * *')
print(f'Loss = {test_loss}  error rate = {test_error}')
print(' ')

#########################################################################################################
#Run FedMA with CNN
#########################################################################################################