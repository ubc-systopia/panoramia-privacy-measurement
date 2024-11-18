# PANORAMIA: Text Data Modality Privacy Measurement
This folder contains for conducting the PANORAMIA Privacy Measurement for the Text Data Modality, on Large Language Models.

## Installation
* Create a virtual environment using python 3.

`virtualenv -p python3 panoramia_venv`

* Activate the virtual environment.

`source panoramia_venv/bin/activate`

* Clone the repo, and install the necessary python packages with `requirements.txt` file.

`pip install -r requirements.txt`

## Running the Code
Executing the complete PANORAMIA pipeline, from training the target model through to obtaining the privacy measurements, can be time-intensive. In this section, we provide guidance on running each module independently for flexibility. 

<!--
For users interested in executing the entire pipeline in one go, instructions for that option are also provided below.
-->

### 1. Training the Target Model

The initial step in the pipeline involves training the target model that will be audited.

```python
python -m src.main --base_train_load_target \
                   --base_log_dir "logs/target/" \
                   --audit_target_saving_dir "outputs/target/" \
                   --audit_target_pretrained_model_name_or_path "gpt2" 
```

This process will result in multiple checkpoints throughout training. Select a checkpoint that will serve as the model for privacy auditing. Ensure sufficient storage space is available, as the training process produces numerous checkpoints.


```bash
TARGET_CHECKPOINT_DIR="outputs/target/epoch_200/checkpoint-25000/" # Note: The save path is influenced by the epoch number. Adapt in case of changing the epoch number.
```


### 2. Training the Generative Model

The next step in our pipeline is to train a generator model. You can train the generator with the command

```python
python -m src.main --base_train_load_generator \
                   --base_log_dir "logs/generator/train/" \
                   --generator_train_pretrained_model_name_or_path "gpt2" \
                   --generator_train_saving_dir "outputs/generator/saved_model/"                 
```

The generator model with the lowest validation loss will be saved in `outputs/generator/saved_model/checkpoint-XXXX/`, where `XXXX` represents the checkpoint number.


### 3. Generating Synthetic Samples

To run the next step using the saved checkpoint of the generator model, first, retrieve the checkpoint directory of the generator model automatically by running:

```bash
GEN_CHECKPOINT_DIR=$(ls -td outputs/generator/saved_model/checkpoint-* | head -1)
```

Then, you can generate synthetic samples with the command

```python
python -m src.main --base_train_load_generator \
                   --base_generate_samples \
                   --base_log_dir "logs/generator/generation/" \ 
                   --generator_train_pretrained_model_name_or_path "gpt2" \
                   --generator_train_saving_dir $GEN_CHECKPOINT_DIR \
                   --generator_generation_saving_dir "outputs/generator/generation/"              
```

The synthetic data will be saved as `outputs/generator/generation/syn_data.csv`. 


### 4. Training the Helper Model

After generating synthetic data, the helper model can be trained. This model assists the baseline classifier in distinguishing real data from synthetic data.

```python
python -m src.main --base_train_load_helper \
                   --base_log_dir "logs/helper/" \
                   --dataset_path_to_synthetic_data "outputs/generator/generation/syn_data.csv" \
                   --audit_helper_saving_dir "outputs/helper/" \
                   --audit_helper_pretrained_model_name_or_path "gpt2"
```  

The helper model with the lowest validation loss will be saved in `outputs/helper/epoch_60/checkpoint-XXXX/`, where `XXXX` represents the checkpoint number. To specify it:

```bash
HELPER_CHECKPOINT_DIR=$(ls -td outputs/helper/epoch_60/checkpoint-* | head -1) # Note: The save path is influenced by the epoch number. Adapt in case of changing the epoch number.
```


### 5. Training the Baseline Classifier and Saving its Predicitions on the Evaluation Set

To evaluate the quality of the synthetic data, we train a baseline classifier to distinguish real data from synthetic data.

```python
python -m src.main  --base_log_dir "logs/baseline/" \
                    --base_train_load_helper \
                    --base_train_baseline \    
                    --dataset_path_to_synthetic_data "outputs/generator/generation/syn_data.csv" \
                    --dataset_mia_num_train 10000 \
                    --dataset_mia_num_val 1000 \
                    --dataset_mia_num_test 10000 \
                    --audit_helper_saving_dir $HELPER_CHECKPOINT_DIR \
                    --attack_baseline_training_args_output_dir "outputs/baseline/"                  
```
<!-- base_attack_main argument has been deleted. Take care of it -->

This command will output several files in `outputs/baseline/`, organized as follows:

```
outputs/baseline/
- model.pth # Model with best validation performance
- test_preds.npy # Predictions (real or synthetic) on the test set
- test_true_labels.npy # Ground truth labels (real or synthetic) for the test
- test_result.txt # Baseline model performance on the test set
- result_best_val.txt # Baseline model performance on the validation set
```

### 6. Training the MIA Classifier and Saving its Predicitions on the Evaluation Set

Finally, we train the Membership Inference Attack (MIA) classifier to distinguish members from non-members of the target model.

```python
python -m src.main  --base_log_dir "logs/MIA/" \
                    --base_train_load_target \
                    --base_train_mia \ 
                    --dataset_path_to_synthetic_data "outputs/generator/generation/syn_data.csv" \
                    --dataset_mia_num_train 10000 \
                    --dataset_mia_num_val 1000 \
                    --dataset_mia_num_test 10000 \
                    --audit_target_saving_dir $TARGET_CHECKPOINT_DIR \
                    --attack_mia_training_args_output_dir "outputs/MIA/"
```

The output files for the MIA classifier are structured similarly to those for the baseline classifier.


## O(1) scores
In order to get the loss values of the auditing set in O1 (the loss threshold attack), run:

```python
python -m src.main  --base_log_dir "logs/o1/" \
                    --base_train_load_target \
                    --base_evaluate_o1 \
                    --dataset_path_to_synthetic_data "outputs/generator/generation/syn_data.csv" \
                    --dataset_mia_num_train 0 \
                    --dataset_mia_num_val 0 \
                    --dataset_mia_num_test 10000 \
                    --dataset_audit_mode "RMRN" \
                    --audit_target_saving_dir $TARGET_CHECKPOINT_DIR \
                    --audit_target_embedding_type "loss" \
                    --attack_o1_output_dir "outputs/o1/"
```

This command will output two files in `outputs/o1/`, organized as follows:
```
outputs/o1/
- O(1)_members_loss.npy # The scores (loss values) of the members from the test set (auditing examples in O(1) terminology) under the target model.
- O(1)_nonmembers_loss.npy # The scores (loss values) of the non-members from the test set (auditing examples in O(1) terminology) under the target model.
```

## Plots and Audit measurements

Refer to the main repository page in the parent directory.

<!--
## Running the Full Pipeline At Once

## Reproducing All the Results Included in the Paper

-->

## Project Structure
<!--
```
.
├── src                                 # Contains the core code for the project
│   ├── datasets                        # Data handling and preparation for all stages of PANORAMIA privacy auditing pipeline
│   │   └── datamodule.py               # Initializes datasets and dataloaders, ensures data consistency across all experiments in PANORAMIA pipeline
│   ├── generator                       # Synthetic Text Data generation for PANORAMIA                      
│   │   ├── generate.py                 # Generates synthetic data samples
│   │   ├── train.py                    # Fine-tunes a text generator model (Language Model) on real data, supports differentially private training
│   │   └── utils.py                    # Utility functions for model training, length-checking of synthetic samples, and privacy configurations
│   ├── audit_model                     # Contains modules for preparing PANORAMIA audit models, including model wrappers, training scripts, and utilities supporting differentially private (DP) training
│   │   ├── audit.py                    # Defines core audit model classes for privacy auditing, with embedding extraction and model-freezing functions
│   │   ├── dp_trainer.py               # Implements differentially private (DP) training routines and privacy configurations for audit models
│   │   ├── train.py                    # Trains audit models (either target or helper) with support for differentially private (DP) and regular training modes
│   │   └── utils.py                    # Utility functions for model setup, initialization, and reproducibility in audit model training
│   ├── attacks                         # Contains modules for executing privacy attacks, including model definitions, custom training routines, and utilities for deterministic setup and configuration management for membership inference and baseline attacks
│   │   ├── custom_trainer.py           # Defines a two-phase trainer class to manage model training and evaluation, supporting metrics logging, deterministic training, and adaptable configurations for privacy attacks
│   │   ├── model.py                    # Implements a GPT-2 based distinguisher network for privacy vulnerability detection (or real-fake detection in case of our baseline), combining text embeddings with featurized logits (either from the target model or the helper model). It includes configurable layers for embedding transformations and supports two-phase optimization to enhance classification effectiveness in privacy attacks.
│   │   ├── train.py                    # Manages the training process for privacy attack models, setting up configurations, data modules, and logging. Supports model training for both baseline and membership inference attacks.
│   │   └── utils.py                    # Provides utility functions for setting random seeds, ensuring deterministic training, managing logging groups for baseline and membership inference attacks in wandb, and computing the privacy measurement on the validation set
│   ├── main.py                         # Orchestrates the PANORAMIA pipeline, managing data preparation, generative model training, synthetic sample generation, audit model training, and privacy attack execution
│   ├── arguments.py                    # Defines command-line arguments for PANORAMIA pipeline configuration
│   └── utils.py                        # Contains utility functions for managing output directories and configuring paths based on experiment parameters for reproducibility and organized output storage
├── experiments
├── requirements.txt
└── README.md

```
-->
<!--
- `src/`: Contains the core code for the project
    - `datasets/`: Data handling and preparation for all stages of the PANORAMIA privacy auditing pipeline
        - `datamodule.py`: Initializes datasets and dataloaders, ensuring data consistency across all experiments in the PANORAMIA pipeline
    - `generator/`: Synthetic text data generation for PANORAMIA
        - `generate.py`: Generates synthetic data samples
        - `train.py`: Fine-tunes a text generator model (Language Model) on real data, with support for differentially private training
        - `utils.py`: Utility functions for model training, length-checking of synthetic samples, and privacy configurations
    - `audit_model/`: Contains modules for preparing PANORAMIA audit models, including model wrappers, training scripts, and utilities supporting differentially private (DP) training
        - `audit.py`: Defines core audit model classes for privacy auditing, with embedding extraction and model-freezing functions
        - `dp_trainer.py`: Implements differentially private (DP) training routines and privacy configurations for audit models
        - `train.py`: Trains audit models (either target or helper) with support for differentially private (DP) and regular training modes
        - `utils.py`: Utility functions for model setup, initialization, and reproducibility in audit model training
    - `attacks/`: Contains modules for executing privacy attacks, including model definitions, custom training routines, and utilities for deterministic setup and configuration management for membership inference and baseline attacks
        - `custom_trainer.py`: Defines a two-phase trainer class to manage model training and evaluation, supporting metrics logging, deterministic training, and adaptable configurations for privacy attacks
        - `model.py`: Implements a GPT-2 based distinguisher network for privacy vulnerability detection (or real-fake detection in case of our baseline), combining text embeddings with featurized logits (either from the target model or the helper model). Includes configurable layers for embedding transformations and supports two-phase optimization to enhance classification effectiveness in privacy attacks
        - `train.py`: Manages the training process for privacy attack models, setting up configurations, data modules, and logging. Supports model training for both baseline and membership inference attacks
        - `utils.py`: Provides utility functions for setting random seeds, ensuring deterministic training, managing logging groups for baseline and membership inference attacks in `wandb`, and computing the privacy measurement on the validation set
    - `main.py`: Orchestrates the PANORAMIA pipeline, managing data preparation, generative model training, synthetic sample generation, audit model training, and privacy attack execution
    - `arguments.py`: Defines command-line arguments for PANORAMIA pipeline configuration
    - `utils.py`: Contains utility functions for managing output directories and configuring paths based on experiment parameters for reproducibility and organized output storage
- `experiments/`: Directory for managing experimental configurations and results
- `requirements.txt`: Lists the dependencies required for the project
- `README.md`: Documentation for understanding and running the project
-->
- `src/`: Contains the core code for the project
    - `datasets/`: Data handling and preparation for PANORAMIA's privacy auditing pipeline
        - `datamodule.py`: Initializes datasets and dataloaders for consistency across experiments
    - `generator/`: Synthetic text data generation for PANORAMIA
        - `generate.py`: Generates synthetic data samples
        - `train.py`: Fine-tunes a text generator model with optional DP training
        - `utils.py`: Utility functions for model training, sample length-checking, and privacy configurations
    - `audit_model/`: Modules for preparing PANORAMIA audit models, supporting DP training
        - `audit.py`: Core audit model classes with embedding extraction and model-freezing
        - `dp_trainer.py`: DP training routines and privacy configurations for audit models
        - `train.py`: Trains audit models with support for DP and regular training
        - `utils.py`: Utility functions for model setup and reproducibility in audit training
    - `attacks/`: Modules for executing baseline/MIA attacks, including custom training and utilities
        - `custom_trainer.py`: Two-phase trainer for model training and evaluation with logging
        - `model.py`: GPT-2 based distinguisher for baseline/MIA classifiers
        - `train.py`: Manages training for baseline and MIA privacy attack models
        - `utils.py`: Utilities for seed setting, deterministic training, and privacy metric computation
    - `main.py`: Orchestrates PANORAMIA pipeline from data prep to attack execution
    - `arguments.py`: Defines command-line arguments for PANORAMIA configuration
    - `utils.py`: Utilities for output directory management and path configuration
<!-- - `experiments/`: Directory for experimental configurations and results-->
- `requirements.txt`: Lists dependencies required for the project
- `README.md`: Documentation for project setup and usage


## Configuration Parameters
This section details the configurable arguments in PANORAMIA, organized by functionality. Each group of arguments lets you control a specific part of the pipeline, like handling data over the whole pipeline, training target models, generating synthetic data, running audits, and etc. Each parameter includes a description, and a default setting.

**Note:** This implementation was developed with the `EleutherAI/wikitext_document_level` dataset and the `gpt2` (as a target model) in mind. The behavior with other datasets or models may be undefined, and adjustments to the code may be necessary for compatibility.

### Base Arguments
These arguments control general settings for the PANORAMIA pipeline, such as specifying paths for logs, managing logging with `wandb`, and selecting which parts of the pipeline to run (e.g., training target models or generating synthetic data). 

- `--base_log_dir`: Path to where the log file would be saved. (Default: `"logs/"`)
- `--base_project_name`: Project name for `wandb` logging. (Default: `"panoramia"`)
- `--base_train_load_target`: Train the target model or load an existing one if available at `--audit_target_saving_dir`. (Default: `False`)
- `--base_train_load_generator`: Train the generative model or load an existing one if available at `--generator_train_saving_dir`. (Default: `False`)
- `--base_generate_samples`: Generate synthetic samples if none are available in `--generator_generation_saving_dir`. (Default: `False`)
- `--base_train_load_helper`: Train the helper model or load it if available from `--audit_helper_saving_dir`. (Default: `False`)
- `--base_train_baseline`: Train and evaluate the baseline classifier. (Default: `False`)
- `--base_train_mia`: Train and evaluate the Membership Inference Attack (MIA) classifier. (Default: `False`)
- `--base_evaluate_o1`: Compute the scores (loss values) for the O(1) auditing. (Default: `False`)
- `--base_full_pipeline`: Execute the entire PANORAMIA pipeline from start to finish. (Default: `False`)

### Data Handler Module Arguments
These arguments configure dataset management, including specifying dataset paths, controlling data splits for training, validation, and testing of the target model, controlling data splits for training, validation, and testing of the privacy attacks (baseline and MIA) and setting options for synthetic data handling. They help ensure that data is prepared and consistent among all stages of the PANORAMIA pipeline.

- `--dataset_path`: Path to the dataset, equivalent to the `path` argument in `datasets.load_dataset`. (Default: `"EleutherAI/wikitext_document_level"`)
- `--dataset_name`: Name of the dataset to use, equivalent to the `name` argument in `datasets.load_dataset`. (Default: `"wikitext-103-raw-v1"`)

**Note:** The code is primarily written with these two default options in mind, and certain parts may rely on assumptions specific to this dataset. Simply changing this argument to point to a different dataset may not work without further code adjustments.

- `--dataset_data_split_percentage`: Percentage of the train split of `wikitext-103-raw-v1 (document level)` taken as the underlying dataset for the `target` model. (Default: `16`)
- `--dataset_validation_size`: Fraction of the data reserved for validation, for the `target` model. (Default: `0.1`)
- `--dataset_test_size`: Fraction of data reserved for testing of the `target` model. (Default: `0.1`)
- `--dataset_num_chunks_keep`: `Stratified Sampling`: Number of data chunks to retain from each taken document from the `wikitext-103-raw-v1`. (Default: `50`)
- `--dataset_path_to_synthetic_data`: Path to synthetic data needed for different stages of the PANORAMIA pipeline. (Default: `None`)
- `--dataset_synthetic_text_column_name`: Column name in the synthetic dataset containing text. (Default: `"text"`)
- `--dataset_seed`: Seed for dataset shuffling and splitting the underlying dataset, for the target model task. (Default: `8`)
- `--dataset_do_shuffle`: Shuffle the dataset if set to `True`. (Default: `True`)
- `--dataset_pretrained_model_name_or_path`: Pre-trained model for specifying the tokenizer. (Default: `"gpt2"`)
- `--dataset_block_size`: Specifies the chunk size for breaking the dataset into equal-sized blocks during preprocessing. (Default: `64`)
- `--dataset_generator_train_percent`: Percentage of the target model training dataset used for generator training. (Default: `35`)
- `--dataset_prompt_sampling_percent`: Percentage of the target model training dataset used for prompt sampling in synthetic data generation. (Default: `15`)
- `--dataset_target_model_percent`:  **[Deprecated]** This argument is no longer used in the current implementation. Although it remains in the codebase for compatibility, it should not be used as it does not affect any processing or training steps. (Default: `45`)
- `--dataset_helper_model_percent`: Only useful in the experiment where we ablate the helper model, when it's trained on synthetic data versus real data. (Default: `100`)
- `--dataset_helper_model_train_data_mode`: Data mode for helper model training (choose `"syn"` for synthetic data and `"real"` for real data). (Default: `"syn"`)
- `--dataset_syn_audit_percent`: Percentage of synthetic data reserved for auditing purposes. The rest would be used for the training of the helper model (Default: `45`)
- `--dataset_mia_num_train`: Number of training examples (per class) for the baseline or MIA classifier. (Default: `6000`)
- `--dataset_mia_num_val`: Number of validation examples (per class) for the baseline or MIA classifier. (Default: `1000`)
- `--dataset_mia_num_test`: Number of test examples for the baseline or MIA (`m` in the PANORAMIA game). (Default: `10000`)
- `--dataset_mia_seed`: Seed used for MIA or baseline dataset splitting into train/validation/test. (Default: `10`)
- `--dataset_include_synthetic`: Include synthetic data in the target model training dataset if `True`. (Default: `False`)
- `--dataset_audit_mode`: Audit mode selection, choose between `"RMFN_fixed_test"`, `"RMRN"`, `"RMFN_train_test_complement"`. (Default: `"RMFN_fixed_test"`)
- `--dataset_num_syn_canary`: Number of synthetic canaries to put in the training dataset, if `--dataset_include_synthetic` is set. (Default: `2000`)
- `--dataset_game_seed`: Seed for the random bits in the PANORAMIA game. (Default: `10`)
- `--dataset_extra_synthetic`: Supply additional synthetic data for the audit purposes if needed if `True`. (Default: `False`)
- `--dataset_path_to_extra_synthetic_data`: Path to the additional synthetic data, if included. (Default: `None`)
- `--dataset_extra_m`: Number of extra synthetic examples to include. (Default: `10000`)

### Generator Module Arguments
These arguments configure the training and generation processes for the synthetic data generator model. They include options for model paths, training parameters (like batch size, learning rate, and differentially private training settings), and generation parameters (like sampling strategy). These settings control how synthetic data is generated for use in the PANORAMIA pipeline.

- `--generator_train_pretrained_model_name_or_path`: Path or name of the pre-trained model to fine-tune for synthetic data generation. (Default: `"gpt2"`)
- `--generator_train_saving_dir`: Directory to save the trained generator model. (Default: `"outputs/generator/saved_model/"`)
- `--generator_train_run_name`: Name for the generator training run, useful for logging or tracking experiments. (Default: `"generator-fine-tune"`)
- `--generator_train_seed`: Random seed for generator model training. (Default: `42`)
- `--generator_train_train_with_dp`: Enables differentially private (DP) training for the generator model if set to `True`. (Default: `False`)
- `--generator_train_optimization_max_steps`: Maximum number of training steps for optimization. (Default: `-1`, which indicates it’s unspecified)
- `--generator_train_optimization_per_device_batch_size`: Batch size per device during generator training. (Default: `64`)
- `--generator_train_optimization_epoch`: Number of training epochs for generator optimization. (Default: `60`)
- `--generator_train_optimization_learning_rate`: Learning rate for generator model optimization. (Default: `2e-05`)
- `--generator_train_optimization_weight_decay`: Weight decay for regularization during optimization. (Default: `0.01`)
- `--generator_train_optimization_warmup_steps`: Number of warmup steps for learning rate scheduling. (Default: `100`)
- `--generator_train_optimization_gradient_accumulation_steps`: Number of steps to accumulate gradients for before updating. (Default: `1`)
- `--generator_train_dp_per_example_max_grad_norm`: Maximum gradient norm per example in DP training to control privacy loss, useful when DP training is enabled. (Default: `0.1`)
- `--generator_train_dp_target_epsilon`: Target epsilon value for differential privacy, useful when DP training is enabled. (Default: `3`)

#### Generator Synthetic Text Generation Arguments

- `--generator_generation_saving_dir`: Directory to save generated synthetic data. (Default: `"outputs/generator/saved_synthetic_data/"`)
- `--generator_generation_syn_file_name`: Filename for saved synthetic data. (Default: `"syn_data.csv"`)
- `--generator_generation_save_loss_on_target`: If `True`, saves the loss values of the generated data, under the target models. (Default: `False`)
- `--generator_generation_seed`: Random seed for generating synthetic data. (Default: `42`)
- `--generator_generation_parameters_batch_size`: Batch size for generating synthetic samples. (Default: `128`)
- `--generator_generation_parameters_prompt_sequence_length`: Length of prompt sequences used in generation, as the input. (Default: `64`)
- `--generator_generation_parameters_max_length`: Maximum length of generated sequences. (Default: `128`)
- `--generator_generation_parameters_top_k`: Top-K sampling value for controlling the diversity of generated text. (Default: `200`)
- `--generator_generation_parameters_top_p`: Top-p (nucleus) sampling value for probability mass selection in generated text. (Default: `1`)
- `--generator_generation_parameters_temperature`: Sampling temperature for controlling randomness in generation. (Default: `1`)
- `--generator_generation_parameters_num_return_sequences`: Number of generated sequences to return per prompt. (Default: `8`)

### Audit Model Module Arguments
These arguments control the setup and training of audit models, including target and helper models. They specify options for loading pre-trained models, configuring training parameters (such as batch size, learning rate, and DP settings), and defining embedding types.

#### Target Model Training Arguments

- `--audit_target_pretrained_model_name_or_path`: Path or identifier for the pre-trained model to use as the target model in privacy auditing, equivalent to `pretrained_model_name_or_path` in `from_pretrained` in HuggingFace. (Default: `"gpt2"`)

**Note:** The code is primarily written with the gpt2 in mind, and certain parts may rely on assumptions specific to this model. Simply changing this argument to point to a different dataset may not work without further code adjustments.

- `--audit_target_saving_dir`: Directory to save the trained target model. (Default: `"outputs/audit_model/target/"`)
- `--audit_target_seed`: Random seed for training the target model. (Default: `42`)
- `--audit_target_run_name`: Name for the target model training run, used for logging and tracking. (Default: `"target_train"`)
- `--audit_target_train_with_DP`: Enables differentially private (DP) training for the target model if set to `True`. (Default: `False`)
- `--audit_target_embedding_type`: Type of embedding to extract from the target model, choose between `"loss_seq"` and `"loss"`. (Default: `"loss_seq"`)
- `--audit_target_do_save_weight_initialization`: Saves the initial weights of the model if set to `True`. Useful for calibration of the membership scores. (Default: `False`)
- `--audit_target_optimization_learning_rate`: Learning rate for optimizing the target model. (Default: `2e-05`)
- `--audit_target_optimization_weight_decay`: Weight decay for regularization in the target model's optimization. (Default: `0.01`)
- `--audit_target_optimization_warmup_steps`: Number of warmup steps for the learning rate scheduler. (Default: `100`)
- `--audit_target_optimization_batch_size`: Batch size for target model training. (Default: `64`)
- `--audit_target_optimization_epoch`: Number of training epochs for the target model. (Default: `200`)
- `--audit_target_optimization_save_strategy`: Strategy for saving model checkpoints (e.g., `"steps"` or `"epoch"`). (Default: `"steps"`)
- `--audit_target_optimization_load_best_model_at_end`: Loads the best-performing model at the end of training if `True`. (Default: `False`)
- `--audit_target_optimization_save_total_limit`: Limits the total number of saved checkpoints. (Default: `None`)
- `--audit_target_optimization_gradient_accumulation_steps`: Number of steps to accumulate gradients before updating the model, only applicable when DP training is enabled. (Default: `64`)
- `--audit_target_dp_per_example_max_grad_norm`: Maximum gradient norm per example for DP training, to control privacy loss. (Default: `0.1`)
- `--audit_target_dp_target_epsilon`: Target epsilon value for differential privacy, applicable when DP training is enabled. (Default: `3`)

#### Helper Model Training Arguments

- `--audit_helper_pretrained_model_name_or_path`: Path or identifier for the pre-trained model to use as the helper model in privacy auditing, equivalent to `pretrained_model_name_or_path` in `from_pretrained` in HuggingFace. (Default: `"gpt2"`)

**Note:** The code is primarily written with the gpt2 in mind, and certain parts may rely on assumptions specific to this model. Simply changing this argument to point to a different dataset may not work without further code adjustments.

- `--audit_helper_saving_dir`: Directory to save the trained helper model. (Default: `"outputs/audit_model/helper/"`)
- `--audit_helper_seed`: Random seed for helper model training. (Default: `42`)
- `--audit_helper_run_name`: Name for the helper model training run, for logging and tracking purposes. (Default: `"helper_train"`)
- `--audit_helper_embedding_type`: Type of embedding used in the helper model, similar to target model embeddings, choose between `"loss_seq"` and `"loss"`. (Default: `"loss_seq"`)
- `--audit_helper_do_save_weight_initialization`: Saves the initial weights of the helper model if set to `True`, supporting calibration. (Default: `False`)
- `--audit_helper_optimization_learning_rate`: Learning rate for optimizing the helper model. (Default: `2e-05`)
- `--audit_helper_optimization_weight_decay`: Weight decay for regularization in the helper model's optimization. (Default: `0.01`)
- `--audit_helper_optimization_warmup_steps`: Number of warmup steps for the learning rate scheduler. (Default: `100`)
- `--audit_helper_optimization_batch_size`: Batch size for helper model training. (Default: `64`)
- `--audit_helper_optimization_epoch`: Number of training epochs for the helper model. (Default: `60`)
- `--audit_helper_optimization_save_strategy`: Strategy for saving checkpoints (e.g., `"epoch"`). (Default: `"epoch"`)
- `--audit_helper_optimization_load_best_model_at_end`: Loads the best-performing helper model at the end of training if `True`. (Default: `True`)
- `--audit_helper_optimization_save_total_limit`: Limits the number of saved checkpoints for the helper model. (Default: `1`)


### Attack (Baseline or MIA) Module Arguments

These arguments configure settings for running privacy attacks, including Membership Inference Attacks (MIA) and baseline attacks. They provide options for attack model configurations, training parameters, evaluation strategies, and logging settings.

#### MIA (Membership Inference Attack) Arguments

- `--attack_mia_net_type`: Type of network architecture to use for the MIA model. Choose between `"mix"` which means looking at the combination of samples and the target model embeddings, `"raw"` which means looking only at the samples, and `"all"` which means looking at the combination of samples, target model embeddings and the helper model embeddings. (Default: `"mix"`)
- `--attack_mia_distinguisher_type`: Type of distinguisher model to use. Currently, the only option is `"GPT2Distinguisher"`. (Default: `"GPT2Distinguisher"`)
- `--attack_mia_run_name`: Name for the MIA run, useful for logging and experiment tracking. (Default: `"RMFN_main_table"`)
- `--attack_mia_training_args_seed`: Random seed for MIA training. (Default: `0`)
- `--attack_mia_training_args_output_dir`: Directory to save the MIA outputs, such as the predictions on the test set. (Default: `"outputs/attacks/mia/"`)
- `--attack_mia_training_args_which_test`: Specifies the file name of the test results. (Default: `"test"`)
- `--attack_mia_training_args_max_steps`: Maximum training steps for MIA in the phase 2 of optimization. (Default: `6000`)
- `--attack_mia_training_args_batch_size`: Batch size for MIA training in the phase 2 of optimization. (Default: `64`)
- `--attack_mia_training_args_warmup_steps`: Warmup steps for learning rate scheduler in MIA training in the phase 2 of optimization. (Default: `500`)
- `--attack_mia_training_args_weight_decay`: Weight decay factor for regularization in MIA training in the phase 2 of optimization. (Default: `0.01`)
- `--attack_mia_training_args_learning_rate`: Learning rate for MIA model optimization in the phase 2 of optimization. (Default: `3e-05`)
- `--attack_mia_training_args_reg_coef`: Regularization coefficient in MIA training in the phase 2 of optimization. (Default: `0`)
- `--attack_mia_training_args_phase1_max_steps`: Maximum steps for the first phase of training. (Default: `1500`)
- `--attack_mia_training_args_phase1_batch_size`: Batch size for the first phase of training. (Default: `64`)
- `--attack_mia_training_args_phase1_learning_rate`: Learning rate for the first training phase. (Default: `0.003`)
- `--attack_mia_training_args_phase1_reg_coef`: Regularization coefficient for phase 1. (Default: `1`)
- `--attack_mia_training_args_logging_steps`: Number of steps between logging outputs. (Default: `10`)
- `--attack_mia_training_args_save_strategy`: Strategy for saving checkpoints (e.g., `"no"` or `"epoch"`). (Default: `"no"`)
- `--attack_mia_training_args_evaluation_strategy`: Strategy for evaluating model during training, such as `"epoch"`. (Default: `"epoch"`)
- `--attack_mia_training_args_overwrite_output_dir`: If `True`, overwrites output directory if it exists. (Default: `True`)
- `--attack_mia_training_args_max_fpr`: Maximum false positive rate for evaluation. (Default: `0.1`)
- `--attack_mia_training_args_evaluate_every_n_steps`: Frequency of evaluations during training, specified in steps. (Default: `100`)
- `--attack_mia_training_args_metric_for_best_model`: Metric used to select the model with the best performance on the validation set, choose between `"acc"` for accuracy, `"auc"` for area under curve, and `"eps"` for the privacy measurement, {c + epsilon }_{lb}. (Default: `"eps"`)


#### Baseline Attack Arguments

- `--attack_baseline_net_type`: Type of network architecture to use for the baseline model. Choose between `"mix"` which means looking at the combination of samples and the helper model embeddings, and `"raw"` which means looking only at the samples. (Default: `"mix"`)
- `--attack_baseline_distinguisher_type`: Type of distinguisher model to use. Currently, the only option is `"GPT2Distinguisher"`.  (Default: `"GPT2Distinguisher"`)
- `--attack_baseline_run_name`: Name for the baseline attack run, useful for logging and tracking. (Default: `"RMFN_main_table"`)
- `--attack_baseline_training_args_seed`: Random seed for baseline attack training. (Default: `0`)
- `--attack_baseline_training_args_output_dir`: Directory for saving baseline attack outputs, such as predictions on the test set. (Default: `"outputs/attacks/baseline/"`)
- `--attack_baseline_training_args_which_test`: Specifies the file name of the test results. (Default: `"test"`)
- `--attack_baseline_training_args_max_steps`: Maximum training steps for baseline attack in the phase 2 of optimization.. (Default: `6000`)
- `--attack_baseline_training_args_batch_size`: Batch size for baseline attack training in the phase 2 of optimization.. (Default: `64`)
- `--attack_baseline_training_args_warmup_steps`: Number of warmup steps for the learning rate schedule in baseline training in the phase 2 of optimization.. (Default: `500`)
- `--attack_baseline_training_args_weight_decay`: Weight decay for regularization in baseline attack training in the phase 2 of optimization.. (Default: `0.01`)
- `--attack_baseline_training_args_learning_rate`: Learning rate for baseline attack model in the phase 2 of optimization.. (Default: `3e-05`)
- `--attack_baseline_training_args_reg_coef`: Regularization coefficient for baseline attack training in the phase 2 of optimization.. (Default: `0`)
- `--attack_baseline_training_args_phase1_max_steps`: Maximum steps in the first phase of baseline training. (Default: `1500`)
- `--attack_baseline_training_args_phase1_batch_size`: Batch size for the first phase of baseline training. (Default: `64`)
- `--attack_baseline_training_args_phase1_learning_rate`: Learning rate for the first phase of baseline training. (Default: `0.003`)
- `--attack_baseline_training_args_phase1_reg_coef`: Regularization coefficient for the first phase. (Default: `1`)
- `--attack_baseline_training_args_logging_steps`: Number of steps between logging outputs. (Default: `10`)
- `--attack_baseline_training_args_save_strategy`: Strategy for saving checkpoints (e.g., `"no"`). (Default: `"no"`)
- `--attack_baseline_training_args_evaluation_strategy`: Evaluation strategy during training, such as `"epoch"`. (Default: `"epoch"`)
- `--attack_baseline_training_args_overwrite_output_dir`: Overwrite existing output directory if `True`. (Default: `True`)
- `--attack_baseline_training_args_max_fpr`: Maximum false positive rate for evaluation. (Default: `0.1`)
- `--attack_baseline_training_args_evaluate_every_n_steps`: Step frequency for evaluations. (Default: `100`)
- `--attack_baseline_training_args_metric_for_best_model`: Metric used to select the model with the best performance on the validation set, choose between `"acc"` for accuracy, `"auc"` for area under curve, and `"eps"` for the c closeness measurement, {c}_{lb}. (Default: `"eps"`)

#### O(1) Attack Arguments

- `--attack_o1_output_dir`: Directory for saving the outputs of the O(1) auditing method. (Default: `"outputs/o1/"`)