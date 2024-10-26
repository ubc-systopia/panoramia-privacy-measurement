## Image Classification Model Examples
This folder contains details for the image classification models audited for privacy leakage using Panoramia framework.


### WRN-28-10
The code for Wide ResNet-28-10 is given in the file wideresnet_28_10.py. Instructions to run are:
`python wideresnet_28_10.py`
In the paper we report results for models trained up till 150 epochs and 300 epochs.


### ResNet
The code for different variants for ResNet is given in the file ResNet.py. Instructions to run are:
`python ResNet.py`
In the paper, we report results for the ResNet101 model trained up to 20, 50, and 100 epochs.

### ViT-small 
We used a ViT-small pre-trained on imagenet, and a one-layer linear classifier head. The Pre-Trained Vit is from Google/vit-base-patch16-384 (https://huggingface.co/google/vit-base-patch16-384). It was trained up to 35 epochs. 

### DP models
`DP_models.py` contain the code for training differentially private (DP) target models. DP is implemented with `Opacus`. In the paper we train DP Resnet18 and DP WRN-16-4 models with various target privacy budget as target models.