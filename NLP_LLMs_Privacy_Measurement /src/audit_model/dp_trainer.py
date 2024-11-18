import logging
from dataclasses import dataclass, field
from typing import Callable
import os

import numpy as np
import torch
from easydict import EasyDict
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam, AdamW
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F
from private_transformers import PrivacyEngine

from src.attacks.utils import enable_full_determinism

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'


@dataclass
class PrivacyArguments:
    """Arguments for differentially private training."""
    per_example_max_grad_norm: float = field(
        default=.1, metadata={
            "help": "Clipping 2-norm of per-sample gradients."
        }
    )
    noise_multiplier: float = field(
        default=None, metadata={
            "help": "Standard deviation of noise added for privacy; if `target_epsilon` is specified, "
                    "use the one searched based budget"
        }
    )
    target_epsilon: float = field(
        default=None, metadata={
            "help": "Privacy budget; if `None` use the noise multiplier specified."
        }
    )
    target_delta: float = field(
        default=None, metadata={
            "help": "Lax probability in approximate differential privacy; if `None` use 1 / len(train_data)."
        }
    )
    accounting_mode: str = field(
        default="rdp", metadata={"help": "One of `rdp`, `glw`, `all`."}
    )
    non_private: str = field(default="no")
    clipping_mode: str = field(default="default")


class DPCustomTrainer:
    def __init__(
            self, 
            model_init: Callable,
            train_dataset: torch.utils.data.Dataset,
            val_dataset: torch.utils.data.Dataset,
            seed: int,
            training_args,
            privacy_args: PrivacyArguments,
            wandb_logger
            ) -> None:
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.training_args = training_args
        self.privacy_args = privacy_args
        self.wandb_logger = wandb_logger
        
        self.best_eval_perplexity = np.inf

        # handling the reproduciblity manually in this case, since we use a custom trainer
        enable_full_determinism(seed)

        # Initialize the model. Seed's already set for reproduciblity
        self.model = model_init()

        # Initialize the dataloaders
        self._setup_dataloaders()

        os.makedirs(os.path.dirname(self.training_args.output_dir), exist_ok=True)
        
    def _setup_dataloaders(self):
        self.train_dataloader = DataLoader(
            self.train_dataset,
            sampler = RandomSampler(self.train_dataset), # Select batches randomly. Seed's already set for reproduciblity
            batch_size = self.training_args.per_device_train_batch_size
        )

        # For validation the order doesn't matter, so we'll just read them sequentially.
        self.validation_dataloader = DataLoader(
            self.val_dataset, 
            sampler = SequentialSampler(self.val_dataset), # Pull out batches sequentially.
            batch_size = self.training_args.per_device_train_batch_size
        )
    
    def create_optimizer_and_scheduler(self, num_training_steps: int, use_scheduler=False):
        """
        Setup the optimizer and the learning rate scheduler.
        """
        
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if
                            (not any(nd in n for nd in no_decay)) and p.requires_grad],
                "weight_decay": self.training_args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if
                            any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.training_args.learning_rate,
        )
        if use_scheduler:
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.training_args.warmup_steps, num_training_steps=num_training_steps
            )

    def train(self):

        # number of training steps
        num_update_steps_per_epoch = len(self.train_dataloader) // self.training_args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        if self.training_args.max_steps > 0:
            self.t_total = self.self.training_args.max_steps
            self.num_train_epochs = self.self.training_args.max_steps // num_update_steps_per_epoch + int(
                self.self.training_args.max_steps % num_update_steps_per_epoch > 0
            )
        else:
            self.t_total = int(num_update_steps_per_epoch * self.training_args.num_train_epochs)
            self.num_train_epochs = self.training_args.num_train_epochs
            self.training_args.max_steps = self.t_total

        # creating the optimizers and scheduler 
        self.create_optimizer_and_scheduler(num_training_steps=self.t_total, use_scheduler=True)

        # correcting the actual batch size based on gradient_accumulation_steps
        self.actual_batch_size = self.training_args.per_device_train_batch_size * self.training_args.gradient_accumulation_steps
        
        # creating the privacy accountant
        privacy_engine = PrivacyEngine(
            module=self.model,
            batch_size=self.actual_batch_size,
            sample_size=len(self.train_dataset),
            epochs=self.num_train_epochs,
            max_grad_norm=self.privacy_args.per_example_max_grad_norm,
            target_epsilon=self.privacy_args.target_epsilon,
            clipping_mode=self.privacy_args.clipping_mode,
        )
        
        # Originally, these could have been null.
        self.privacy_args.noise_multiplier = privacy_engine.noise_multiplier
        self.privacy_args.target_delta = privacy_engine.target_delta

        
        logging.info(f'Privacy args: \n {self.privacy_args}')

        privacy_engine.attach(self.optimizer)
        
        self.training_loop()
        

    def compute_loss(self, outputs, labels):
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        seq_lens = (shift_labels != -100).sum(dim=1) # there shoudn't be any padding in my use case tho
        loss = F.cross_entropy(shift_logits.permute(0, 2, 1), shift_labels, reduction="none")
        loss = loss.sum(dim=1) / seq_lens  # Per token loss.
        return loss  # (batch_size,)
    
    def training_loop(self):
        logging.info("***** Running training *****")
        logging.info("  Num examples = %d", len(self.train_dataloader.dataset))
        logging.info("  Num Epochs = %d", self.num_train_epochs)
        logging.info("  Instantaneous batch size per device = %d", self.training_args.per_device_train_batch_size)
        logging.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                       self.actual_batch_size)
        logging.info("  Gradient Accumulation steps = %d", self.training_args.gradient_accumulation_steps)
        logging.info("  Total optimization steps = %d", self.t_total)        

        training_stats = []

        self.model.to(device)

        tr_loss = torch.tensor(0.0).to(device)
        
        logging_loss_scalar = 0.0

        
        for epoch_i in range(0, int(np.ceil(self.num_train_epochs))):

            # ========================================
            #               Training
            # ========================================
            
            logging.info("")
            logging.info('======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.num_train_epochs))
            logging.info('Training...')

            # should be cautious when we do zero_grad, since we do accumulation of gradients
            self.model.zero_grad(set_to_none=True)
        
            for step, batch in enumerate(self.train_dataloader):
                self.model.train()
                b_input_ids = batch["input_ids"].to(device)
                b_masks = batch["attention_mask"].to(device)
                b_labels = batch["labels"].to(device)

                
                outputs = self.model(input_ids=b_input_ids,
                                labels=b_labels, 
                                attention_mask = b_masks
                                )

                loss = self.compute_loss(outputs, b_labels)
                vector_loss = loss
                scalar_loss = loss.mean(dim=0) / self.training_args.gradient_accumulation_steps
                scalar_loss = scalar_loss.detach()
                losses = dict(vector_loss=vector_loss, scalar_loss=scalar_loss)
                tr_loss += losses["scalar_loss"]
                

                if (step + 1) % self.training_args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    self.training_args.gradient_accumulation_steps >= len(self.train_dataloader) == (step + 1)
                ):
                    vector_loss = losses.get("vector_loss")
                    self.optimizer.step(loss=vector_loss)
                    self.lr_scheduler.step()
                    self.model.zero_grad(set_to_none=True)

                    tr_loss_scalar = tr_loss.item()
                    
                    logging.info("")
                    logging.info("  Step training loss: {0:.2f}".format((tr_loss_scalar - logging_loss_scalar)))
                    self.wandb_logger.log({"train/train_loss": tr_loss_scalar - logging_loss_scalar})

                    logging_loss_scalar = tr_loss_scalar

                    
                else:
                    self.optimizer.virtual_step(loss=losses.get("vector_loss"))
                

            # ========================================
            #               Validation
            # ========================================

            logging.info("")
            logging.info("Running Validation...")
            
            self.model.eval()

            total_eval_loss = 0

            # Evaluate data for one epoch
            for batch in self.validation_dataloader:
                
                b_input_ids = torch.tensor(batch["input_ids"]).to(device)
                b_masks = torch.tensor(batch["attention_mask"]).to(device)
                b_labels = torch.tensor(batch["labels"]).to(device)
                
                with torch.no_grad():        

                    outputs  = self.model(b_input_ids,
                                    attention_mask = b_masks,
                                    labels=b_labels)

                    loss = self.compute_loss(outputs, b_labels)
                    scalar_ev_loss = loss.sum(dim=0)
                    scalar_ev_loss = scalar_ev_loss.item()
                
                
                
                total_eval_loss += scalar_ev_loss

                
            avg_val_loss = total_eval_loss / len(self.validation_dataloader.dataset)
            
            current_perplexity = np.exp(avg_val_loss)

            logging.info("  Validation Loss: {0:.2f}".format(avg_val_loss))
            logging.info(f" Validation Perplexity: {current_perplexity}")
            self.wandb_logger.log({'val/val_loss': avg_val_loss})

            if current_perplexity < self.best_eval_perplexity:
                self.best_eval_perplexity = current_perplexity
                self.model.save_pretrained(self.training_args.output_dir)
                logging.info(f"best perplexity decreased to: {current_perplexity}. Saving model...")

            pe = self.optimizer.privacy_engine
            privacy_metrics = pe.get_privacy_spent(accounting_mode="all", lenient=True)
            privacy_stats = pe.get_training_stats()

            logging.info(privacy_metrics)
            

            # self.run.log({'val/val_loss': avg_val_loss, 'val/val_accuracy': val_acc})

            # Record all statistics from this epoch.
            # training_stats.append(
            #     {
            #         'epoch': epoch_i + 1,
            #         'Training Loss': avg_train_loss,
            #         'Valid. Loss': avg_val_loss,
            #         'Training Time': training_time,
            #         'Validation Time': validation_time
            #     }
            # )

        logging.info("")
        logging.info("Training complete!")
        
            
