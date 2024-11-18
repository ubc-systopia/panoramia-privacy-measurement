import os

def setup_attack_output_dir(config):
    """"
    Sets up output directory based on specific varying (ablation) parameters set.
    """
    config.attack.baseline.training_args.output_dir = os.path.join(
        config.attack.baseline.training_args.output_dir, 
        f'attack_num_train_{config.dataset.mia_num_train}',
        f'seed_{config.attack.baseline.training_args.seed}/'
    )
    config.attack.mia.training_args.output_dir = os.path.join(
        config.attack.mia.training_args.output_dir, 
        f'attack_num_train_{config.dataset.mia_num_train}',
        f'seed_{config.attack.mia.training_args.seed}/'
    )
    return config
