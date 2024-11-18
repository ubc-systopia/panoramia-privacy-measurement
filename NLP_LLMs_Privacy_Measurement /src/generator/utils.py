import logging
import math
from copy import deepcopy

import torch
import pandas as pd
import numpy as np
from transformers import Trainer, AutoModelForCausalLM, AutoConfig
import wandb

from src.audit_model.dp_trainer import DPCustomTrainer, PrivacyArguments

def check_length_to_block_size(synthetic_data: pd.DataFrame, tokenizer, block_size):
    encodings = tokenizer(synthetic_data['text'].values.tolist())

    lengths = np.array([len(encodings["input_ids"][i]) for i in range(len(synthetic_data))])

    logging.info(f"Number of synthetic samples with length=block_size={block_size}")
    logging.info(np.sum(lengths == np.array([block_size for i in range(len(synthetic_data))])))
    
    neq_block_size_indices = np.where(lengths != np.array([block_size for i in range(len(synthetic_data))]))[0]

    for idx in neq_block_size_indices[::-1]:
        synthetic_data.drop(idx, inplace=True)
    
    return synthetic_data


def regular_training(config, training_args, train_dataset, validation_dataset):
    logging.info(f"Fine-tuning the generator with hyperparameters:\n{training_args}")

    # initializing wandb for visualization
    wandb.init(
            project=config.base.project_name,
            group="generator-fine-tune",
            name=config.generator.train.run_name,
            config=training_args,
            reinit=True
        )
    
    trainer = Trainer(
        model_init=lambda: AutoModelForCausalLM.from_pretrained(config.generator.train.pretrained_model_name_or_path),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
    )

    # fine-tune the generator
    trainer.train()

    # saving the final model
    # Commented this later. Saving model is now assigned to trainer.
    # trainer.save_model()
    # model.save_pretrained(config.generator.train.saving_dir)

    
    eval_results = trainer.evaluate()
    logging.info(f"Evaluation results:\n{eval_results}")
    logging.info(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    return trainer.model

def init_dp_model(config):
    model_checkpoint = config.generator.train.pretrained_model_name_or_path

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
    logging.info(f"Fine-tuning the generator with DP with hyperparameters:\n{training_args}")
    
    wandb_config = deepcopy(training_args)
    wandb_config.epsilon = config.generator.train.dp.target_epsilon
    wandb_config.per_example_max_grad_norm = config.generator.train.dp.per_example_max_grad_norm

    # initializing wandb for visualization 
    wandb_logger = wandb.init(
            project=config.base.project_name,
            group="generator-fine-tune",
            name=config.generator.train.run_name,
            config=wandb_config,
            reinit=True
    )
    
    privacy_args = PrivacyArguments(
        per_example_max_grad_norm=config.generator.train.dp.per_example_max_grad_norm, 
        target_epsilon=config.generator.train.dp.target_epsilon,
        target_delta=(1/len(train_dataset)),
        clipping_mode= 'ghost'
    )
    
    trainer = DPCustomTrainer(
        model_init=init_dp_model(config),
        train_dataset=train_dataset.with_format("torch"),
        val_dataset=validation_dataset.with_format("torch"),
        seed=config.generator.train.seed,
        training_args=training_args,
        privacy_args=privacy_args,
        wandb_logger=wandb_logger
    )

    # fine-tune the generator
    trainer.train()
    