import logging
import os
from datetime import datetime

import yaml
from easydict import EasyDict
from transformers import AutoModelForCausalLM

from src.arguments import init_args, args_to_nested_dict
from src.utils import setup_attack_output_dir
from src.datasets.datamodule import PANORAMIADataModule
from src.generator.train import fine_tune_generator
from src.generator.generate import generate_synthetic_samples
from src.audit_model.train import train_audit_model
from src.audit_model.audit import AuditModelGPT2CLM
from src.attacks.train import train_attack
from src.o1_loss_th_attack.evaluate_membership_loss import compute_and_save_membership_loss


def main(config: EasyDict):
    """
    runs the whole pipeline of PANORAMIA
    """
    # instantiating a data module of PANORAMIA
    dm = PANORAMIADataModule(
        **config.dataset
    )
    
    # --------------------
    # Part 1. Generative Model Training/Loading
    # --------------------

    # Train if the generative model is not provided
    if config.base.train_load_generator or config.base.full_pipeline:
        if os.path.exists(config.generator.train.saving_dir):
            logging.info(f"Loading the generator model from {config.generator.train.saving_dir} ...")
            generator_model = AutoModelForCausalLM.from_pretrained(config.generator.train.saving_dir)
        else:
            generator_model = fine_tune_generator(config, dm, train_with_dp=config.generator.train.train_with_dp)

    
    # --------------------
    # Part 2. Generate/Load Synthetic Samples
    # --------------------
    if config.base.generate_samples or config.base.full_pipeline:
        if not os.path.exists(config.generator.generation.saving_dir):
            generate_synthetic_samples(config, dm, generator_model)

    
    # Handling the synthetic dataset in data module
    dm.setup_synthetic_dataset()


    # del the generator model from memory
    if config.base.train_load_generator or config.base.full_pipeline:
        del generator_model

    # --------------------
    # Part 3. Train/Load Audit Model
    # --------------------

    # train the target model
    if config.base.train_load_target or config.base.full_pipeline:
        if os.path.exists(config.audit.target.saving_dir):
            target_model = AutoModelForCausalLM.from_pretrained(config.audit.target.saving_dir)
        else:
            target_model = train_audit_model(
                config=config,
                dm=dm,
                train_helper=False,
                train_with_DP=config.audit.target.train_with_DP
            )

    # train the helper model
    if config.base.train_load_helper or config.base.full_pipeline:
        if os.path.exists(config.audit.helper.saving_dir):
            helper_model = AutoModelForCausalLM.from_pretrained(config.audit.helper.saving_dir)
        else:
            helper_model = train_audit_model(
                config=config,
                dm=dm,
                train_helper=True,
                train_with_DP=False
            )

    # --------------------
    # Part 4. MIA/Baseline Attack
    # --------------------

    # instantiate audit model objects
    if config.base.train_load_target or config.base.full_pipeline:
        target_audit_model = AuditModelGPT2CLM(
            model=target_model,
            embedding_type=config.audit.target.embedding_type,
            block_size=dm.block_size
        )

    if config.base.train_load_helper or config.base.full_pipeline:
        helper_audit_model = AuditModelGPT2CLM(
            model=helper_model,
            embedding_type=config.audit.helper.embedding_type,
            block_size=dm.block_size
        )
    
    # train the baseline and mia

    # updating output dir based on input parameters
    # config = setup_attack_output_dir(config)
    

    if config.base.train_baseline or config.base.full_pipeline:
        if os.path.exists(os.path.join(config.attack.baseline.training_args.output_dir, config.attack.baseline.training_args.which_test +'_preds.npy')):
            ...
        else:
            baseline_trainer = train_attack(
                config=config,
                dm=dm,
                audit_model=helper_audit_model,
                train_baseline=True
            )

    if config.base.train_mia or config.base.full_pipeline:
        if os.path.exists(os.path.join(config.attack.mia.training_args.output_dir, config.attack.mia.training_args.which_test + '_preds.npy')):
            ...
        else:
            if config.attack.mia.net_type == 'all':
                mia_trainer = train_attack(
                    config=config,
                    dm=dm,
                    audit_model=[target_audit_model, helper_audit_model],
                    train_baseline=False
                )
            else:
                mia_trainer = train_attack(
                    config=config,
                    dm=dm,
                    audit_model=target_audit_model,
                    train_baseline=False
                )
    
    if config.base.evaluate_o1 or config.base.full_pipeline:
        compute_and_save_membership_loss(
            dm=dm,
            audit_model=target_audit_model,
            output_dir=config.attack.o1.output_dir
        )
                

    
    

    


if __name__ == "__main__":
    # argument settings
    args = init_args()

    # determine to load from a yaml config or the arguments
    if args.use_yml_config:
        # read from yaml
        with open(args.path_yml_config, 'r') as stream:
            config = yaml.safe_load(stream)
    else:
        config = args_to_nested_dict(args)
        
    # convert normal dictionary of config to attribute dictionary
    config = EasyDict(config)
    

    # Setup logging directory
    os.makedirs(config.base.log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(
            config.base.log_dir,
            f'output_{datetime.now().strftime("%d_%m_%Y-%H_%M_%S_%f")}.log'
        ),
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main(config)


    
    


    