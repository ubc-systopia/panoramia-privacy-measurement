## Generative Model for Non-Member Data Creation (CIFAR10)
This folder contains the necessary files for cifar10 styleGAN2 pytorch model to generate the data for non-member set.

The trained generative model can be found at https://drive.google.com/file/d/1owinmAGS9XvLJl9UIDAS7121yerR1BY3/view?usp=sharing

### Generate Synthetic Samples
Once the Generator is trained, generate synthetic samples using the following shell file.

`./generate.sh`

The model has been adapted from this repository here: https://github.com/NVlabs/stylegan2-ada-pytorch 


You can either use the above to train from scratch or use our updated dataset_tool.py file that contains the dataset split for training the GAN as well as the `style_gan_train.sh` script.
Citation for the generative model:
```
@inproceedings{Karras2020ada,
  title     = {Training Generative Adversarial Networks with Limited Data},
  author    = {Tero Karras and Miika Aittala and Janne Hellsten and Samuli Laine and Jaakko Lehtinen and Timo Aila},
  booktitle = {Proc. NeurIPS},
  year      = {2020}
}
```
