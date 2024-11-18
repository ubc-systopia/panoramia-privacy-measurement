import os
import logging
from copy import deepcopy

import torch
from datasets import load_dataset, Dataset
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer
import numpy as np
from easydict import EasyDict
import wandb

from src.datasets.datamodule import PANORAMIADataModule
from src.audit_model.audit import AuditModel
from src.attacks.model import TextAttackModel
from src.attacks.custom_trainer import TwoPhaseTrainer
from src.attacks.utils import set_wand_group

os.environ["WANDB__SERVICE_WAIT"]="3000"

def train_attack(
    config: EasyDict, 
    dm: PANORAMIADataModule,
    audit_model: AuditModel,
    train_baseline=False
):
    # specifying the attack config between mia/baseline
    if train_baseline:
        attack_config = config.attack.baseline
    else: 
        attack_config = config.attack.mia

    # loading the seed number
    seed = attack_config.training_args.seed

    # updating parameters based on the input seed
    attack_config.run_name = f"train_num_{config.dataset.mia_num_train}" + f"_mia_seed_{config.dataset.mia_seed}" + f"_training_seed_{seed}"
    
    logging.info(f"Training with seed: {seed}." + 'baseline mode.' if train_baseline else 'mia mode.' )

    # loading train/val/test datasets
    train_dataset, val_dataset, test_dataset = dm.get_mia_datasets()

    # instantiating the attack model_init. It's a callable for controlling reproducibility 
    model_init = lambda: TextAttackModel(
        side_net=audit_model,
        net_type=attack_config.net_type,
        distinguisher_type=attack_config.distinguisher_type
    )

    wandb_config = deepcopy(attack_config.training_args)
    wandb_config.mia_seed = config.dataset.mia_seed
    wandb_config.training_num = config.dataset.mia_num_train
    wandb_config.test_num = config.dataset.mia_num_test
    # initializing wandb for visualization 
    wandb_logger = wandb.init(
            project=config.base.project_name,
            group=set_wand_group(config, train_baseline),
            name=attack_config.run_name,
            config=wandb_config,
            reinit=True
        )

    # setting the training arguments
    training_args = attack_config.training_args

    # instantiate the trainer for the attack model
    trainer = TwoPhaseTrainer(
        model_init=model_init,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        seed=seed,
        training_args=training_args,
        wandb_logger=wandb_logger
    )

    if not os.path.exists(os.path.join(training_args.output_dir, 'model.pth')):
        # training is dependent on net_type
        if attack_config.net_type == 'mix' or attack_config.net_type == 'all':
            trainer.train()
        elif attack_config.net_type == 'raw':
            trainer.one_phase_train()
        else:
            raise NotImplementedError
    
    
    # test evaluation
    test_model = TextAttackModel(
        side_net=audit_model,
        net_type=attack_config.net_type,
        distinguisher_type=attack_config.distinguisher_type
    )

    test_model.load_state_dict(torch.load(attack_config.training_args.output_dir+'/model.pth'))

    trainer.test(
        model = test_model, 
        test_dataset= test_dataset, 
        output_dir=attack_config.training_args.output_dir,
        which_test=training_args.which_test
    )
    
    
    





















    

