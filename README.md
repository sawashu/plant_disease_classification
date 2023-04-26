plant_disease_classification
==============================

A short description of the project.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
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
    │
    |
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
    

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
