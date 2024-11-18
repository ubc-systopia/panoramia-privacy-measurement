import math
import os
import logging

from transformers import AutoModelForCausalLM
from transformers import TrainingArguments
from easydict import EasyDict

from src.datasets.datamodule import PANORAMIADataModule
from src.audit_model.utils import dp_training, regular_training

    
    
def train_audit_model(
    config: EasyDict, 
    dm: PANORAMIADataModule,
    train_helper=False,
    train_with_DP=False
):
    # load training and validation datasets from data module
    train_dataset, validation_dataset, _ = dm.get_target_model_datasets()
    if train_helper:
        train_dataset, validation_dataset, _ = dm.get_helper_model_dataset()

    # loading the config, either from target or helper
    audit_config = config.audit.target
    if train_helper:
        audit_config = config.audit.helper
    
    # # asserting the provided epoch argument is list, for studying the role of overtraining
    # assert isinstance(audit_config.optimization.epochs, list), "Epochs argument is not a list"

    epoch = audit_config.optimization.epoch
    
    # setting the training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(
            audit_config.saving_dir,
            f"epoch_{epoch}"
        ),

        seed=audit_config.seed,

        num_train_epochs=epoch,
        learning_rate=audit_config.optimization.learning_rate,
        weight_decay=audit_config.optimization.weight_decay,
        warmup_steps=audit_config.optimization.warmup_steps,
        per_device_train_batch_size=audit_config.optimization.batch_size,
        gradient_accumulation_steps=audit_config.optimization.gradient_accumulation_steps if train_with_DP else 1,

        save_strategy=audit_config.optimization.save_strategy,
        load_best_model_at_end=audit_config.optimization.load_best_model_at_end,
        save_total_limit=audit_config.optimization.save_total_limit,
        # save_steps=audit_config.optimization.save_steps,
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
    )


    if train_with_DP:
        return dp_training(config, training_args, train_dataset, validation_dataset)
    else:
        return regular_training(config, training_args, train_dataset, validation_dataset, train_helper)

    
    

    
