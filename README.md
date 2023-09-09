plant_disease_classification
==============================

We aim to create an algorithm to classify plants and plant diseases. This is useful at the individual scale for people new to plant care and for farmers to use for crop monitoring at a larger scale. A simple and effective form of plant disease classification is by visual inspection of the leaves, thus, we proposed using a Convolutional Neural Network (CNN) to classify plant images from the various datasets. Since these neural networks and trained models require images that people might not want to share, we decided to consider and compare using Federated Learning to train the CNN.

This project combines two plant disease datasets Plant Village https://github.com/spMohanty/PlantVillage-Dataset and Plant Doc https://doi.org/10.1145/3371158.3371196 to train several ML models including FedMA https://github.com/IBM/FedMA, pFedHN https://github.com/AvivSham/pFedHN, and inception CNN https://www.mdpi.com/2079-9292/10/12/1388 to determine the best model for predicting plant diseases.

The trained model accomplishes over 96% accuracy in classifying test images.

Project Organization
------------
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    ├── our_pFedHN
    │   ├── __init__.py 
    │   │
    │   ├── Generate_Dataset         <- Create dataset of plant disease
    │   │   |
    |   |   ├── split_dataset_train_test.py   
    |   |   ├── train_val_test_split   <- actual dataset for machine learning
    |   |   |
    |   |   └── plant_doc_and_plant_village_images   <- actual dataset
    │   |
    │   ├── pfedhn_hetro_res       <- directory to save result
    │   │   └── results_50_inner_steps_seed_42.json     <- result of pFedHN with json file
    |   |
    |   ├── pred_res      <- examples of input images and prediction result
    │   │
    │   ├── pred_res1        <- examples of input images and prediction result part 2
    |   |
    |   │
    |   ├── dataset.py        <- dataset preprocessing for pFedHN
    │   │ 
    |   ├── models.py       <- model CNN and hyper networks 
    |   |
    |   ├── node.py        <- client class for federated learning
    |   |
    |   ├── requirements.txt        <- requirements packages
    |   |
    |   ├── t2.py       <- edit version of pFedHN main file to apply into our plant disease classifcation task
    │   │ 
    |   ├── test_ave_acc.png        <- plot of test average accuracy
    |   |
    |   ├── test_ave_loss.png        <- plot of test average loss
    |   |
    |   ├── trainer.py        <- original pFedHN main file
    |   |
    |   ├── utils.py        <- utils for pFedHN
    |   |
    |   ├── val_ave_acc.png       <- plot of validation average accuracy
    |   | 
    │   └── val_ave_loss.png  <- plot of validation average loss
    

--------


