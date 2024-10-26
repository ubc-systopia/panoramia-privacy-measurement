## PANORAMIA: Image Modality Privacy Measurement

This folder contains the code for conducting the PANORAMIA Privacy Measurement for the Image Modality.

### Generative Model:
In order to generate synthetic data for non-members, please refer to the code under `generative_model` folder. We provide code and references to StyleGAN2 to generate data for CIFAR 32X32 images
as well as a pre-trained generative model for the CIFAR10 dataset along with relevant data splits used for training the GAN in our experiments in the `dataset_tool.py` file. Instructions are present
in the `Readme.md` file in the `generative_model` folder. For training from scratch using your own configurations please refer to the styleGAN NVIDIA link provided in the folder for reference as well.


### Target and Helper Models:
Example target models are provided in the `image_target_models` folder. The same can be used to train the helper model trained on synthetic data. This synthetic data should be 
separate from the MIA and baseline training data to ensure no overlap.

### Baseline and MIA 
Code and experimental details for the baseline attack model as well as MIA are given under `baseline_and_MIA` folder. Please run this from the root of the project due to the package structure based that way.
The config file provides experimental setup for each module used in the pipeline along with instructions for running the experiments.
