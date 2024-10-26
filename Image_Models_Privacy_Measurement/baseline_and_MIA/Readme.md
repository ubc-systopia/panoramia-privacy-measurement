## PANORAMIA Pipeline

This folder contains the necessary componenets for the PANORAMIA image based pipeline. After the non-member synthetic data generation is complete, and you have your target and helper models at hand we can use this pipeline to run the privacy measurement via PANORAMIA. The necessary configurations for the image_module, loss_module, panoramia audit training as well as datasets creation is contained in the `config.yaml` file. Feel free to change the file according to your specific requirements. This one is specifically for CIFAR10 dataset used to train various target models.


Please run the code using the below command:

`python3 panoramia_audit.py`


