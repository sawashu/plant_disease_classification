'''
This code is the code that genearted the train, validation, and testing data folders
from the dataset 'plant_doc_and_plant_village_images'

It DOES NOT need to be run again, you can just use the different datasets stored in 
the train_val_test_split. 

Note there is a set seed, so you can run it again and get the 
same test, train, val folders.
'''

import splitfolders

data_dir = 'plant_doc_and_plant_village_images'

splitfolders.ratio(data_dir, output="train_val_test_split", seed=1337, ratio=(.8, 0.1, 0.1)) 