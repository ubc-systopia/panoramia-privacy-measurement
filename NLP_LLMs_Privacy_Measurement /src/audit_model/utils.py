from typing import Callable
import logging
import math
import os
from copy import deepcopy

import torch
from easydict import EasyDict
from transformers import Trainer, AutoModelForCausalLM, AutoConfig
from transformers.trainer_utils import enable_full_determinism
import wandb

from src.audit_model.dp_trainer import DPCustomTrainer, PrivacyArguments

os.environ["WANDB__SERVICE_WAIT"]="3000"

def setup_model(
    config: EasyDict, 
) -> Callable:
    """
    Return a callable for model_init parameter
    This function needs to be extended for supporting different audit models.
    """
    return lambda: AutoModelForCausalLM.from_pretrained(config.pretrained_model_name_or_path)

def save_init(
    model_init: Callable,
    seed: int,
    saving_dir
):
    enable_full_determinism(seed)
    initialized_model = model_init()
    path = os.path.join(saving_dir, "init_model/")
    os.makedirs(path, exist_ok=True)
    torch.save(
        initialized_model.state_dict(),  
        path + "model.pth"
    )


def init_dp_model(config):
    model_checkpoint = config.audit.target.pretrained_model_name_or_path

    # load config. note that ghost clipping doesn't support parameter sharing -> tie_word_embeddings = False
    lm_config = AutoConfig.from_pretrained(model_checkpoint)
    lm_config.return_dict = True
    lm_config.tie_word_embeddings = False

    # loading the model
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint, config=lm_config)

    # Clone the embedding into the lm_head for better initialization.
    lm_head = model.get_output_embeddings()
    embedding = model.get_input_embeddings()
    lm_head.weight.data.copy_(embedding.weight.data)
    logging.info(f'Cloning initial embedding into lm_head, '
          f'checking norms... \n'
          f'\tlm_head: {lm_head.weight.norm()}, embedding: {embedding.weight.norm()}')
    torch.testing.assert_allclose(lm_head.weight, embedding.weight)
    del lm_head, embedding

    return lambda: model

def dp_training(config, training_args, train_dataset, validation_dataset):
    logging.info(f"Fine-tuning the target with DP with hyperparameters:\n{training_args}")
    
    wandb_config = deepcopy(training_args)
    wandb_config.epsilon = config.audit.target.dp.target_epsilon
    wandb_config.per_example_max_grad_norm = config.audit.target.dp.per_example_max_grad_norm

    # initializing wandb for visualization 
    wandb_logger = wandb.init(
            project=config.base.project_name,
            group="target-dp-fine-tune",
            name=config.audit.target.run_name,
            config=wandb_config,
            reinit=True
    )
    
    privacy_args = PrivacyArguments(
        per_example_max_grad_norm=config.audit.target.dp.per_example_max_grad_norm, 
        target_epsilon=config.audit.target.dp.target_epsilon,
        target_delta=(1/len(train_dataset)),
        clipping_mode= 'ghost'
    )
    
    trainer = DPCustomTrainer(
        model_init=init_dp_model(config),
        train_dataset=train_dataset.with_format("torch"),
        val_dataset=validation_dataset.with_format("torch"),
        seed=config.audit.target.seed,
        training_args=training_args,
        privacy_args=privacy_args,
        wandb_logger=wandb_logger
    )

    # fine-tune the generator
    trainer.train()
    


def regular_training(config, training_args, train_dataset, validation_dataset, train_helper):

    # loading the config, either from target or helper
    audit_config = config.audit.target
    if train_helper:
        audit_config = config.audit.helper

    audit_type = 'helper' if train_helper else f'target_epoch_{audit_config.optimization.epoch}'
    
    # initializing wandb for visualization
    wandb.init(
            project=config.base.project_name,
            group=audit_type,
            name=audit_config.run_name,
            config=training_args,
            reinit=True
        )

    logging.info(f"Fine-tuning the {audit_type} without DP with hyperparameters:\n{training_args}")

    model_init = setup_model(audit_config)

    if audit_config.do_save_weight_initialization:
        save_init(model_init, audit_config.seed, audit_config.saving_dir)

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
    )

    # fine-tune the audit model
    trainer.train()

    # in target model mode, we only care about the model at the end of training. Training args should be set to not save during training.
    # in helper model mode, we  care about the best model on the validation set. Training args should be set to save the best one durinng training.
    if not train_helper:
        trainer.save_model()
    
    # Evaluating the audit model
    eval_results = trainer.evaluate()
    logging.info(f"Evaluation results:\n{eval_results}")
    logging.info(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    return trainer.model