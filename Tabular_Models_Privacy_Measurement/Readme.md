# README

This document provides instructions for running the training scripts for tabular data.
Scipts should be run in this order:
1- train_generative_tab.py
2- train_target_tab.py
3- train_baseline_mia_tab.py

## Scripts

### 1. `train_generative_tab.py`

This script is used to train the generative model. 

**Command:**
python train_generative_tab.py

**Arguments:**
- '--continuous_col': 'List of continuous columns'
- '--target':'Target variable'
- '--data_pth':'Path to the  dataset'
- '--domain_pth':'Path to the  json file describing the dataset's format following this format https://tapas-privacy.readthedocs.io/en/latest/dataset-schema.html'
- '--save_data_folder': 'Path to save datasets'
- '--random_seed': 'random seed for consistancy'
- '--mia_train_size': 'Size of MIA training set'
- '--mia_test_size': 'Size of MIA test set'
- '--gen_train_size': 'Size of generative model training set'

### 2. `train_target_tab.py`

This script is used to train the target and helper models. 

**Command:**
python train_target_tab.py

**Arguments:**
- '--continuous_col': 'List of continuous columns'
- '--target':'Target variable'
- '--lr': 'Learning rate default = 0.001'
- '--data_pth':'Path to the  dataset'
- '--domain_pth':'Path to the  json file describing the dataset's format following this format https://tapas-privacy.readthedocs.io/en/latest/dataset-schema.html'
- '--save_data_folder': 'Path to save datasets'
- '--random_seed': 'random seed for consistancy'
- '--mia_train_size': 'Size of MIA training set'
- '--mia_test_size': 'Size of MIA test set'
- '--gen_train_size': 'Size of generative model training set'
- '--name': 'Experiment's name'


### 3. `train_baseline_mia_tab.py`

This script is used to train a Bseline model as well as Membership Inference Attacks (MIA) on tabular data.

**Command:**
python train_baseline_mia_tab.py

**Arguments:**
- '--target':'Target variable'
- '--save_data_folder': 'Path to save datasets'
- '--random_seed': 'random seed for consistancy'
- '--nb_run': 'Number of time to run the auditing experiment (used to add CI)'
- '--name': 'Experiment's name'
- '--audit_test_size': 'Size of audit test set before Bernouilli sampling'

For more details, refer to the script documentation or use the `--help` flag with each command.
