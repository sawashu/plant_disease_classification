#########################################################################################################
# IMPORTS
#########################################################################################################

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize

from CNN_inception import MyCNN
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

# criterion
cnn_criterion = nn.CrossEntropyLoss()
fedMA_cnn_criterion = nn.CrossEntropyLoss()

# training loss, error rate
cnn_loss_err = np.ndarray((2, Max_epochs))  # matrix to save loss (col 1), error rate (col 2)
# fedMA_cnn_loss_err = np.ndarray((2, Max_epochs))  # matrix to save loss (col 1), error rate (col 2)

# validation loss, error rate
cnn_val_loss_err = np.ndarray((2, Max_epochs))  # matrix to save loss (col 1), error rate (col 2)
# fedMA_cnn_val_loss_err = np.ndarray((2, Max_epochs))  # matrix to save loss (col 1), error rate (col 2)

# testing error:
cnn_test_loss_err = np.array((1, 2))
#########################################################################################################

#########################################################################################################
for i in range(Max_epochs):

    print(f'Epoch {i}:')

    cnn_running_loss = 0.0
    cnn_num_correct, cnn_num_samples = 0, 0

    # fedMA_cnn_running_loss = 0.0
    # fed_MA_cnn_num_correct, num_samples = 0, 0

    print('Starting training...')
    for j,(images,labels) in enumerate(train_loader):
        
        if j > 2:
            break
        else:
            # run cnn training part
            cnn_cur_loss, cnn_cur_correct = cnn.fit(images, labels, cnn_criterion, cnn_optimizer)
            cnn_running_loss += cnn_cur_loss
            cnn_num_correct += cnn_cur_correct
            cnn_num_samples += len(labels)

            # if j % 2 == 0 and j < 20:
            #     print('num images j: ', j, 'current loss = ', cnn_cur_loss)

            # run fedMA part

    # cnn save training data
    cnn_error_rate = 1 - (np.divide(cnn_num_correct, cnn_num_samples))   
    
    cnn_loss_err[0,i] = cnn_running_loss
    cnn_loss_err[1,i] = cnn_error_rate
    print(f'loss = {cnn_running_loss}   error rate = {cnn_error_rate}')
    # fedMA

    print('Starting validation...')
    # cnn validation
    cnn_val_running_loss = 0.0
    cnn_val_num_correct, cnn_val_num_samples = 0, 0
    with torch.no_grad():
        for j,(images, labels) in enumerate(valid_loader):

            if j>2:
                break
            else:

                cnn_val_loss, cnn_val_correct = cnn.val_predict(images, labels, criterion = nn.CrossEntropyLoss())
                cnn_val_running_loss += cnn_val_loss
                cnn_val_num_correct += cnn_val_correct
                cnn_val_num_samples += len(labels)

                # if j % 2 == 0 and j < 20:
                #     print('num images j: ', j, 'current loss = ', cnn_cur_loss)

    cnn_val_error_rate = 1 - (np.divide(cnn_val_num_correct, cnn_val_num_samples)) 
    cnn_val_loss_err[0,i] = cnn_val_running_loss
    cnn_val_loss_err[1,i] = cnn_val_error_rate
    print(f'loss = {cnn_val_running_loss}   error rate = {cnn_val_error_rate}')
    
print('Starting prediction....')
# end Max Epochs, now use trained model to predict
cnn_test_loss, cnn_test_error_rate = cnn.predict(test_loader, criterion = nn.CrossEntropyLoss())
cnn_test_loss_err[0, 0] = cnn_test_loss
cnn_test_loss_err[0, 1] = cnn_test_error_rate


# save all info in csv file
np.savetxt("cnn_training.csv", cnn_loss_err,
              delimiter = ",")

np.savetxt("cnn_validation.csv", cnn_val_loss_err,
              delimiter = ",")

np.savetxt("cnn_test.csv", cnn_test_loss_err,
              delimiter = ",")
