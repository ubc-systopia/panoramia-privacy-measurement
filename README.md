# PANORAMIA: Privacy Auditing of Machine Learning Models Without Retraining (NeurIPS 2024)
[Mishaal Kazmi*](https://www.linkedin.com/in/mishaalkazmi555/), [Hadrien Lautraite*](https://www.linkedin.com/in/hadrienlautraite/), [Alireza Akbari*](https://www.linkedin.com/in/alireza-akbari-9373aa175/), [Qiaoyue Tang*](https://scholar.google.ca/citations?user=qr_qHm4AAAAJ&hl=en), [Mauricio Sororco](https://www.linkedin.com/in/mauricio-soroco-9b6472326/?originalSubdomain=ca), [Tao Wang](https://scholar.google.ca/citations?user=6r3AEG8AAAAJ&hl=en), [Sébastien Gambs](https://scholar.google.ca/citations?hl=en&user=2q1NjMgAAAAJ), [Mathias Lécuyer](https://mathias.lecuyer.me/)

 
[![arXiv](https://img.shields.io/badge/arXiv-PANORAMIA-b31b1b.svg)](https://arxiv.org/abs/2402.09477)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/ubc-systopia/panoramia-privacy-measurement/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/NeurIPS-2024-brightgreen.svg)](https://neurips.cc/virtual/2024/poster/96581)

<p align="center">
<img src="https://github.com/Mishaa1/priv-audit/blob/neurips_code/images_models_priv_audit/system_design.png" alt="PANORAMIA System Design" width="600" height="300">
 
 *PANORAMIA’s two-phase privacy audit. Phase 1: training of generative model G using member data. Phase 2: training a MIA
on a subset of member data and generated non-member data, along with the loss values of target model f on these data points. Then, the comparison of
the performance of the MIA to a baseline classifier that does not have access to f is performed.*
</p>

<hr style="border:0.5px solid gray">

## ◆ Overview: 
We present PANORAMIA, a privacy leakage measurement framework for machine learning models that relies on membership inference attacks using generated data as non-members.
By relying on generated non-member data, PANORAMIA eliminates the common dependency of privacy measurement tools on in-distribution non-member data.
As a result, PANORAMIA does not modify the model, training data, or training process, and only requires access to a subset of the training data.
We evaluate PANORAMIA on ML models for image and tabular data classification and large-scale language models.

## ◆ Usecases
This work is instrumental in allowing model owners and privacy auditors to measure their ML model's privacy leakage without the need to retrain the ML pipeline, and allows privacy measurement under distribution shifts. Some other use cases include measuring ML model privacy leakage in 
- a post-hoc setting,
- Federated learning scenarios where the client has access to the ML model API and their own training data and wants to get an assessment on how safe the ML model is towards their own data,
- In-distribution non-member data availability is limited or not available at all


## ◆ Repository Structure:
A brief overview of the structure of the repository:
```
./
├── Image_Models_Privacy_Measurement/   # PANORAMIA for privacy measurement of image classification models
│
│   └──baseline_and_MIA/             # Baseline Attack and MIA for PANORAMIA Privacy Measurement/Audit Phase
│       ├── image_module.py          # Module to distinguish member and non-member images 
│       ├── losses_module.py         # Module to distinguish member and non-members based on losses from target or helper model
|       ├── panoramia_audit.py       # Main file to run the baseline and MIA pipeline
|       ├── config.yaml              # Experimental details to run the main panoramia_audit file
|       └── Readme.md                # Readme file with the relevant run instructions
|
|   └──generative_model/             # Generative model for cifar10 dataset
│       ├── dataset_tool.py          # Linear and convolution layers with equalized learning rate
│       ├── generate.py              # Function that generates the noise input for each generator block
│       ├── generate.sh              # Script to run the trained generator model to generate samples
│       ├── style_gan_train.sh       # Script to train the generator on the GAN train data subset
│       └── Readme.md                # Readme file with the relevant run instructions
│
|
|   └──image_target_models/          # Code to train image target models WRN-28-10 and ViT-pre-trained-small
|       ├── DP_models.py             # Code for all the image DP target models in the paper
│       ├── wideresnet_28_10.py      # Wide ResNet image classification model for CIFAR10
|       ├── ResNet.py                # ResNet image classification model for CIFAR10
│       └── Readme.md                # Readme file with the relevant run instructions and link to ViT
│
│
├── O1_Steinke_Code/   # Code Engineered for the paper: Privacy Auditing for O(1) Training by Steinke et al.
│       ├── o1_audit.py
|       ├── audit_with_relaxation.py    # code for section E in Appendix of paper
│       └── Readme.md      # Readme file with the relevant run instructions 
│
|
├── LLM_NLP_Privacy_Measurement/   # PANORAMIA for privacy measurement of  text data modality
│   ├── src                                 # Contains the core code for the project
│   │   ├── datasets                        # Data handling and preparation for all stages of PANORAMIA privacy auditing pipeline
│   │   │   └── datamodule.py               # Initializes datasets and dataloaders, ensures data consistency across all experiments in PANORAMIA pipeline
│   │   ├── generator                       # Synthetic Text Data generation for PANORAMIA                      
│   │   │   ├── generate.py                 # Generates synthetic data samples
│   │   │   ├── train.py                    # Fine-tunes a text generator model (Language Model) on real data, supports differentially private training
│   │   │   └── utils.py                    # Utility functions for model training, length-checking of synthetic samples, and privacy configurations
│   │   ├── audit_model                     # Contains modules for preparing PANORAMIA audit models, including model wrappers, training scripts, and utilities supporting differentially private (DP) training
│   │   │   ├── audit.py                    # Defines core audit model classes for privacy auditing, with embedding extraction and model-freezing functions
│   │   │   ├── dp_trainer.py               # Implements differentially private (DP) training routines and privacy configurations for audit models
│   │   │   ├── train.py                    # Trains audit models (either target or helper) with support for differentially private (DP) and regular training modes
│   │   │   └── utils.py                    # Utility functions for model setup, initialization, and reproducibility in audit model training
│   │   ├── attacks                         # Contains modules for executing privacy attacks, including model definitions, custom training routines, and utilities for deterministic setup and configuration management for membership inference and baseline attacks
│   │   │   ├── custom_trainer.py           # Defines a two-phase trainer class to manage model training and evaluation, supporting metrics logging, deterministic training, and adaptable configurations for privacy attacks
│   │   │   ├── model.py                    # Implements a GPT-2 based distinguisher network for privacy vulnerability detection (or real-fake detection in case of our baseline), combining text embeddings with featurized logits (either from the target model or the helper model). It includes configurable layers for embedding transformations and supports two-phase optimization to enhance classification effectiveness in privacy attacks.
│   │   │   ├── train.py                    # Manages the training process for privacy attack models, setting up configurations, data modules, and logging. Supports model training for both baseline and membership inference attacks.
│   │   │   └── utils.py                    # Provides utility functions for setting random seeds, ensuring deterministic training, managing logging groups for baseline and membership inference attacks in wandb, and computing the privacy measurement on the validation set
│   │   ├── main.py                         # Orchestrates the PANORAMIA pipeline, managing data preparation, generative model training, synthetic sample generation, audit model training, and privacy attack execution
│   │   ├── arguments.py                    # Defines command-line arguments for PANORAMIA pipeline configuration
│   │   └── utils.py                        # Contains utility functions for managing output directories and configuring paths based on experiment parameters for reproducibility and organized output storage
│   ├── requirements.txt
│   └── README.md
│
├── Tabular_Models_Privacy_Measurement/   # PANORAMIA for privacy measurement of tabular data models
│   └── data/results/
|          ├── mia_baseline_default_0.csv
|      └── models
|          ├── MLP_classifier.py
|          ├── f_default.pth
|      └── train_baseline_mia_tab.py
|      └── train_generative_tab.py
│      └── train_target_tab.py
|      └── Readme.md      # Readme file with the relevant run instructions 
│
├── Plotting/
│   ├── compute_plot_audit_values.py   # run the plotting code to generate plots
│   └── Readme.md      # Readme file with the relevant run instructions
└── README.md
```
## ◆ Usage:
This auditing framework works in two parts: i) The Data Generation Phase ii) The Audit Evaluation Phase. Below, we provide a semantic overview of how to run the framework using the image data modality as an example. For detailed instructions/commands specific to each data modality, please refer to the respective directory.

### ◇ Data Generation Phase:

### ◇ ◇ Generative Model for Non-Member Data
Step 1: Train a generative model on a subset of data used to train your target ML model. This generative model will then generate samples for the non-member dataset. This repository provides a generative model for CIFAR10 32x32 dataset <!-- as well as WikiText 2 and the Adult dataset-->. Link to pre-trained model for CIFAR10 is shared under the `generative_model` folder under the image modality.


### ◇ Audit Evaluation Phase:

#### ◇ ◇ Target and Helper Models:
Step 2: Train the target model (if needed otherwise use a pre-trained version available) for loss values for <!--black-box--> the Membership Inference Attack (MIA). Train the helper model on synthetic data for the baseline attack.
The target model is the ML model whose training data privacy leakage we are concerned about. The helper model helps the baseline model by providing additional features (i.e., embeddings) that serve as side information about the synthetic data distribution. This baseline classifier distinguishes between member and non-member data without access to the target model, as opposed to the MIA. We also analyze image models trained with DP-SGD under the file `DP_models.py` along with the instructions to run the code.


#### ◇ ◇ The MIA and Baseline Attacks
Step 3: Use code and running guidelines under the baseline_and_MIA folders to train the baseline and MIA classifier distinguishing between real member and synthetic non-member using helper and target models as side information respectively. Here we provide code for training the baseline and MIA attacks using the image and losses modules. The main pipeline that trains both is under `panoramia_audit.py` file. The folder contains relevant instructions and a Yaml config file highlighting experiment guidelines. Eventually, we evaluate the baseline and MIA classifiers on a test set consisting of real member and synthetic non-members.


### ◇ Plotting
Step 4: Finally we can generate pretty pictures, and run the plotting code under plotting_scripts folder using

`python3 compute_plot_audit_values.py --results_data_folder`
This takes as input path to results folder containing csv file(s) with the format `member, baselinepred, mia<model_name1>pred, ...` 

### ◇ Comparison with SOTA

This repository also includes our implementation of the threshold-based attack for O(1) code by Steinke et al. under the O1_Steinke_Code folder along with the instructions to run the code.


## ◆ Requirements:
You will need to install fairly standard dependencies, further, each data modality folder contains its own relevent dependencies under the respective `requirements.txt` file.

`pip install scipy, sklearn, numpy, matplotlib`

and also some machine learning frameworks to train models. We train our models with Pytorch. This code has been specifically tested using Python=3.7 but should be adaptable to newer versions of Python.


## ◆ Citation:

You can cite this paper with:

 ```
@article{kazmi2024panoramia,
  title={PANORAMIA: Privacy Auditing of Machine Learning Models without Retraining},
  author={Mishaal Kazmi* and Hadrien Lautraite* and Alireza Akbari* and Qiaoyue Tang* and  Mauricio Soroco and Tao Wang and Sébastien Gambs and Mathias Lécuyer},
  journal={Advances in Neural Information Processing Systems},
  pdf={https://arxiv.org/abs/2402.09477}
  year={2024},
}
```
---
---


