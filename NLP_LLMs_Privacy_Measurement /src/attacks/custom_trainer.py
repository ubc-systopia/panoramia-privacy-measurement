import os
import logging
import gc
from typing import Callable
import datetime
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam, AdamW
from transformers import get_linear_schedule_with_warmup
from easydict import EasyDict
import sklearn.metrics as metrics

from src.attacks.utils import enable_full_determinism, get_max_eps_validation

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

class TwoPhaseTrainer:
    def __init__(
        self,
        model_init: Callable,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset,
        seed: int,
        training_args: EasyDict,
        wandb_logger
        ) -> None:
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.training_args = training_args
        self.wandb_logger = wandb_logger

        self.best_eval_accuracy = 0.
        self.best_eval_auc = 0.
        self.best_eval_score = None
        self.best_eval_true = None
        self.best_eval_eps = 0.

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
            batch_size = self.training_args.batch_size
        )

        # For validation the order doesn't matter, so we'll just read them sequentially.
        self.validation_dataloader = DataLoader(
            self.val_dataset, 
            sampler = SequentialSampler(self.val_dataset), # Pull out batches sequentially.
            batch_size = self.training_args.batch_size
        )
    
    def create_optimizer_and_scheduler(self):

        # TODO: assert model has logits linear head. Currently, it's hardcoded in the code
        phase_1_optimizer = Adam(
            [self.model.model.logits_linear_head] + [self.model.model.output_neuron_bias] + [self.model.model.logits_hidden_layer.weight, self.model.model.logits_hidden_layer.bias],  # TODO: this is hardcoded in the code, and it's also dependent on the use_hidden_logits of the model. Update it.
            lr= self.training_args.phase1_learning_rate
        )

        # no decay for bias and normalization layers
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
        
        phase_2_optimizer = AdamW(optimizer_grouped_parameters, 
            lr = self.training_args.learning_rate
        )
        
        phase_2_scheduler = get_linear_schedule_with_warmup(phase_2_optimizer, 
                                                        num_warmup_steps = self.training_args.warmup_steps, 
                                                        num_training_steps = self.training_args.max_steps)

        return phase_1_optimizer, phase_2_optimizer, phase_2_scheduler
        
    def _re_init_gpt_head(self):
        with torch.no_grad():
            self.model.model.gpt_linear_head.data = torch.randn_like(self.model.model.gpt_linear_head)

    
    def one_phase_train(self):
        """
        This is temporary function to support training baseline with no helper model, whose training doesn't need first phase
        """
        # no decay for bias and normalization layers
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
        
        phase_2_optimizer = AdamW(optimizer_grouped_parameters, 
            lr = self.training_args.learning_rate
        )
        
        phase_2_scheduler = get_linear_schedule_with_warmup(phase_2_optimizer, 
                                                        num_warmup_steps = self.training_args.warmup_steps, 
                                                        num_training_steps = self.training_args.max_steps)

        # re-initialize the gpt head. It had initlized with zero. Now, with standard gaussian
        self._re_init_gpt_head()

        # second loop training loop
        self.training_loop(
            optimizer=phase_2_optimizer,
            optim_steps=self.training_args.max_steps,
            use_scheduler=True,
            scheduler=phase_2_scheduler
        )


    def train(self):
        # creating the optimizers and scheduler for the two phase optimizations
        phase_1_optimizer, phase_2_optimizer, phase_2_scheduler = self.create_optimizer_and_scheduler()

        # first phase training loop
        self.training_loop(
            optimizer=phase_1_optimizer,
            optim_steps=self.training_args.phase1_max_steps,
            use_scheduler=False,
            scheduler=None
        )

        # re-initialize the gpt head. It had initlized with zero. Now, with standard gaussian
        self._re_init_gpt_head()

        # second loop training loop
        self.training_loop(
            optimizer=phase_2_optimizer,
            optim_steps=self.training_args.max_steps,
            use_scheduler=True,
            scheduler=phase_2_scheduler
        )
    
    def _regularize_head(self):
        raise NotImplementedError

    @staticmethod
    def _format_time(elapsed):
            return str(datetime.timedelta(seconds=int(round((elapsed)))))

    def _training_epoch_end(self, total_train_loss, all_train_scores, all_train_labels, t0):
        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(self.train_dataloader)  

        train_auc = metrics.roc_auc_score(np.concatenate(all_train_labels), np.concatenate(all_train_scores), max_fpr=self.training_args.max_fpr)     
        
        
        # Measure how long this epoch took.
        training_time = self._format_time(time.time() - t0)

        
        logging.info("Average training loss: {0:.2f}".format(avg_train_loss))
        logging.info("Training auc: {0:.3f}".format(train_auc))
        logging.info("Training epoch took: {:}".format(training_time))

        self.wandb_logger.log({"train/train_loss": avg_train_loss})
    
    def _validation_epoch_end(self, total_eval_loss, num_correct, all_scores, all_labels, t0):
        logging.info(f"Number of corrects: {num_correct} out of {len(self.validation_dataloader.sampler)}")

        val_acc = num_correct / len(self.validation_dataloader.sampler)

        auc = metrics.roc_auc_score(np.concatenate(all_labels), np.concatenate(all_scores), max_fpr=self.training_args.max_fpr)
        
        if val_acc > self.best_eval_accuracy:
            self.best_eval_accuracy = val_acc
            if self.training_args.metric_for_best_model == 'acc':
                self.best_eval_score = all_scores
                self.best_eval_true = all_labels
                torch.save(self.model.state_dict(), self.training_args.output_dir+"/model.pth")

                logging.info(f"acc increased to: {val_acc}. Saving model...")
        
        if auc > self.best_eval_auc: 
            self.best_eval_auc = auc
            if self.training_args.metric_for_best_model == 'auc':
                self.best_eval_score = all_scores
                self.best_eval_true = all_labels
                torch.save(self.model.state_dict(), self.training_args.output_dir+"/model.pth")

                logging.info(f"auc increased to: {auc}. Saving model...")
            
        if self.training_args.metric_for_best_model == 'eps':
            eps = get_max_eps_validation(
                preds=np.concatenate(all_scores), 
                labels=np.concatenate(all_labels), 
                dataset_size=np.concatenate(all_labels).shape[0],
                delta=0.,
                audit_CI=0.05
            )
            self.wandb_logger.log({'val/val_measurement': eps})
            if eps > self.best_eval_eps:
                self.best_eval_eps = eps
                self.best_eval_score = all_scores
                self.best_eval_true = all_labels
                torch.save(self.model.state_dict(), self.training_args.output_dir+"/model.pth")
                
                logging.info(f"eps increased to: {eps} on validation. Saving model...")
        
        avg_val_loss = total_eval_loss / len(self.validation_dataloader)
        validation_time = self._format_time(time.time() - t0)    

        logging.info("  Validation Loss: {0:.2f}".format(avg_val_loss))
        logging.info("  Validation acc: {0:.3f}".format(val_acc))
        logging.info("  Validation auc: {0:.3f}".format(auc))
        logging.info("  Validation took: {:}".format(validation_time))

        self.wandb_logger.log({'val/val_loss': avg_val_loss, 'val/val_accuracy': val_acc})
    

    def training_loop(
        self, 
        optimizer: torch.optim.Optimizer, 
        optim_steps: int,
        use_scheduler: bool = False,
        scheduler = None
    ):
        # moving model to the available device. torch.nn.module.to happens in-place
        self.model.to(device)
        # self.model.side_net.model.to(device)

        num_step = 0
        start_t0 = time.time()

        while num_step < optim_steps:

            total_train_loss = 0
            all_train_scores = []
            all_train_labels = []
            train_t0 = time.time()
            
            for batch in self.train_dataloader:
                self.model.train()

                b_input_ids = batch["input_ids"].to(device)
                b_masks = batch["attention_mask"].to(device)
                b_labels = batch["labels"].to(device)

                self.model.zero_grad()        

                model_outputs = self.model(
                    input_ids=b_input_ids,
                    attention_mask=b_masks,
                    labels=b_labels
                )

                train_logits = model_outputs.logits

                probs = torch.sigmoid(train_logits).view(-1)

                all_train_scores.append(probs.detach().cpu().numpy())
                all_train_labels.append(b_labels.detach().cpu().numpy())

                loss = model_outputs.loss

                batch_loss = loss.item() 
                total_train_loss += batch_loss

                loss.backward()

                optimizer.step()
                if use_scheduler:
                    scheduler.step()

                num_step += 1

                if num_step % self.training_args.evaluate_every_n_steps == 0: 
                    
                    self.model.eval()

                    all_val_scores = []
                    all_val_labels = []
                    total_eval_loss = 0
                    num_correct = 0
                    t0 = time.time()

                    for batch in self.validation_dataloader:
                        
                        b_input_ids = batch["input_ids"].to(device)
                        b_masks = batch["attention_mask"].to(device)
                        b_labels = batch["labels"].to(device)
                        
                        with torch.no_grad():        

                            model_outputs = self.model(
                                input_ids=b_input_ids,
                                attention_mask=b_masks,
                                labels=b_labels
                            )
                        
                            loss = model_outputs.loss 
                            val_logits = model_outputs.logits
                        
                        
                        val_pred_prob = torch.sigmoid(val_logits).view(-1)
                        all_val_scores.append(val_pred_prob.detach().cpu().numpy())
                        all_val_labels.append(b_labels.detach().cpu().numpy())
                        
                
                        preds = torch.where(val_pred_prob >= 0.5, 1., 0.)
                        
                        num_correct += torch.sum(preds.view(-1) == b_labels.view(-1)).item()
                        
                        batch_loss = loss.item()
                        total_eval_loss += batch_loss 
                    
                    self._validation_epoch_end(
                        total_eval_loss,
                        num_correct,
                        all_val_scores,
                        all_val_labels,
                        t0
                    )
            
            self._training_epoch_end(
                        total_train_loss,
                        all_train_scores,
                        all_train_labels,
                        train_t0
                    )
        
        with open(self.training_args.output_dir + 'result_best_val.txt', 'w+') as f:
            f.write("Best validation accuracy: " + str(self.best_eval_accuracy) + '\n' + "Best validation AUC:" + str(self.best_eval_auc) + '\n' + "Best validation eps:" + str(self.best_eval_eps))

        logging.info("Training complete!")
        logging.info("Total training took {:} (h:mm:ss)".format(self._format_time(time.time()-start_t0)))

            
    def test(self, model, test_dataset, output_dir, which_test='test'):
        model.eval()
        model.to(device)
        # model.side_net.model.to(device)

        test_dataloader = DataLoader(
            test_dataset, 
            sampler = SequentialSampler(test_dataset), # Pull out batches sequentially.
            batch_size = 32
        )
        
        all_scores = []
        all_labels = []
        num_correct = 0
        t0 = time.time()

        for batch in test_dataloader:
            
            b_input_ids = batch["input_ids"].to(device)
            b_masks = batch["attention_mask"].to(device)
            b_labels = batch["labels"].to(device)
            
            with torch.no_grad():        

                model_outputs = model(
                    input_ids=b_input_ids,
                    attention_mask=b_masks,
                    labels=b_labels
                )
            
                logits = model_outputs.logits
            
            
            pred_prob = torch.sigmoid(logits).view(-1)
            all_scores.append(pred_prob.detach().cpu().numpy())
            all_labels.append(b_labels.detach().cpu().numpy())
            
    
            preds = torch.where(pred_prob >= 0.5, 1., 0.)
            
            num_correct += torch.sum(preds.view(-1) == b_labels.view(-1)).item()
        
        test_acc = num_correct / len(test_dataloader.sampler)
        test_auc = metrics.roc_auc_score(np.concatenate(all_labels), np.concatenate(all_scores), max_fpr=0.01)

        
        logging.info("  Test acc: {0:.3f}".format(test_acc))
        logging.info("  Test auc: {0:.3f}".format(test_auc))

        best_test_preds_dir = output_dir + which_test + '_preds.npy'
        test_true_labels_dir = output_dir + which_test + '_true_labels.npy'

        with open(output_dir + which_test + '_result.txt', 'w+') as f:
            f.write("Test accuracy: " + str(test_acc) + '\n' + "Test AUC:" + str(test_auc))


        np.save(test_true_labels_dir, np.concatenate(all_labels))
        np.save(best_test_preds_dir, np.concatenate(all_scores))
        

        
        
        


        
