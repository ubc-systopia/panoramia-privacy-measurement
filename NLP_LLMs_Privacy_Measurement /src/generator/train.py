import math
import logging


from transformers import AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
import wandb
from easydict import EasyDict

from src.datasets.datamodule import PANORAMIADataModule
from src.generator.utils import regular_training, dp_training

def fine_tune_generator(
    config: EasyDict, 
    dm: PANORAMIADataModule,
    train_with_dp: bool = False
):
    # load training and validation datasets from data module
    train_dataset, validation_dataset, _ = dm.get_generator_training_datasets()

    # load pre-trained model
    # Commented this later. Now, we leave the instantiation of the model to trainer, to ensure reproducibility (Since the LM Head would be initialized from scratch)
    # model = AutoModelForCausalLM.from_pretrained(config.generator.train.pretrained_model_name_or_path)

    # loading optimization hyperparameters from the config file
    opt_hyp_paramrs = config.generator.train.optimization

    # setting the training arguments
    training_args = TrainingArguments(
        # report_to = 'wandb',
        # run_name = config['train']['exp_name'],
        output_dir=config.generator.train.saving_dir,
        seed=config.generator.train.seed, # Ensuring Reproducibility
        num_train_epochs=opt_hyp_paramrs.epoch,
        max_steps=opt_hyp_paramrs.max_steps,
        gradient_accumulation_steps=opt_hyp_paramrs.gradient_accumulation_steps,
        learning_rate=opt_hyp_paramrs.learning_rate,
        weight_decay=opt_hyp_paramrs.weight_decay,
        warmup_steps=opt_hyp_paramrs.warmup_steps,
        per_device_train_batch_size=opt_hyp_paramrs.per_device_batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        do_train=True,
        do_eval=True
    )

    if train_with_dp:
        dp_training(config, training_args, train_dataset, validation_dataset)
    else:
        regular_training(config, training_args, train_dataset, validation_dataset)
    
    