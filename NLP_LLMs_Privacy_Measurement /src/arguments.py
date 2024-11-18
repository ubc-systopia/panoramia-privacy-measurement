import argparse
from collections import defaultdict

def add_load_args(parser):
    parser.add_argument("--use_yml_config", action='store_true', default=False, help='load config from yaml')
    parser.add_argument("--path_yml_config", type=str, help='Path to experiment yaml config')
    
    return parser

def add_base_args(parser):
    parser.add_argument("--base_log_dir", type=str, help='Path to where the log file would be saved', default='logs/')
    parser.add_argument("--base_project_name", type=str, help='Project name for wandb logging', default='panoramia')
    
    parser.add_argument("--base_train_load_target", action="store_true", help="Train the target model, or loads it if there exists a model in the path provided by --audit_target_saving_dir argument")
    parser.add_argument("--base_train_load_generator", action="store_true", help="Train the generative model, or loads it if there exists a model in the path provided by --generator_train_saving_dir argument")
    parser.add_argument("--base_generate_samples", action="store_true", help="Generate synthetic samples, if there is no synthetic sample in the path provided by --generator_generation_saving_dir argument")
    parser.add_argument("--base_train_load_helper", action="store_true", help="Train the helper model, or loads if if there exists a model in the path provided by --audit_helper_saving_dir argument")
    parser.add_argument("--base_train_baseline", action="store_true", help="Train and evaluate the baseline classifier")
    parser.add_argument("--base_train_mia", action="store_true", help="Train and evaluate the MIA classifier")
    parser.add_argument("--base_evaluate_o1", action="store_true", help="Compute the scores (loss values) for the O(1) auditing")
    parser.add_argument("--base_full_pipeline", action="store_true", help="Run the full pipeline from start to finish")

    return parser

def add_dataset_args(parser):
    parser.add_argument('--dataset_path', type=str, help='', default='EleutherAI/wikitext_document_level')
    parser.add_argument('--dataset_name', type=str, help='', default='wikitext-103-raw-v1')
    parser.add_argument('--dataset_data_split_percentage', type=int, help='', default=16)
    parser.add_argument('--dataset_validation_size', type=float, help='', default=0.1)
    parser.add_argument('--dataset_test_size', type=float, help='', default=0.1)
    parser.add_argument('--dataset_num_chunks_keep', type=int, help='', default=50)
    parser.add_argument('--dataset_path_to_synthetic_data', type=str, help='')
    parser.add_argument('--dataset_synthetic_text_column_name', type=str, help='', default='text')
    parser.add_argument('--dataset_seed', type=int, help='', default=8)
    parser.add_argument('--dataset_do_shuffle', action='store_true', help='', default=True)
    parser.add_argument('--dataset_pretrained_model_name_or_path', type=str, help='', default='gpt2')
    parser.add_argument('--dataset_block_size', type=int, help='', default=64)
    parser.add_argument('--dataset_generator_train_percent', type=int, help='', default=35)
    parser.add_argument('--dataset_prompt_sampling_percent', type=int, help='', default=15)
    parser.add_argument('--dataset_target_model_percent', type=int, help='', default=45)
    parser.add_argument('--dataset_helper_model_percent', type=int, help='', default=100)
    parser.add_argument('--dataset_helper_model_train_data_mode', type=str, help='', default='syn')
    parser.add_argument('--dataset_syn_audit_percent', type=int, help='', default=45)
    parser.add_argument('--dataset_mia_num_train', type=int, help='', default=6000)
    parser.add_argument('--dataset_mia_num_val', type=int, help='', default=1000)
    parser.add_argument('--dataset_mia_num_test', type=int, help='', default=10000)
    parser.add_argument('--dataset_mia_seed', type=int, help='', default=10)
    parser.add_argument('--dataset_include_synthetic', action='store_true', help='', default=False)
    parser.add_argument('--dataset_audit_mode', type=str, help='', default='RMFN_fixed_test')
    parser.add_argument('--dataset_num_syn_canary', type=int, help='', default=2000)
    parser.add_argument('--dataset_game_seed', type=int, help='', default=10)
    # parser.add_argument('--dataset_include_auxilary', action='store_true', help='', default=False)
    # parser.add_argument('--dataset_num_aux_in', type=int, help='', default=10000)
    # parser.add_argument('--dataset_combine_wt2_test', action='store_true', help='', default=False)

    parser.add_argument('--dataset_extra_synthetic', action='store_true', help='', default=False)
    parser.add_argument('--dataset_path_to_extra_synthetic_data', type=str, help='', default='')
    parser.add_argument('--dataset_extra_m', type=int, help='', default=10000)
    
    return parser

def add_generator_args(parser):
    # generator training arguments
    parser.add_argument('--generator_train_pretrained_model_name_or_path', type=str, help='', default='gpt2')
    parser.add_argument('--generator_train_saving_dir', type=str, help='', default='outputs/generator/saved_model/')
    parser.add_argument('--generator_train_run_name', type=str, help='', default='generator-fine-tune')
    parser.add_argument('--generator_train_seed', type=int, help='', default=42)
    parser.add_argument('--generator_train_train_with_dp', action='store_true', help='', default=False)
    parser.add_argument('--generator_train_optimization_max_steps', type=int, help='', default=-1)
    parser.add_argument('--generator_train_optimization_per_device_batch_size', type=int, help='', default=64)
    parser.add_argument('--generator_train_optimization_epoch', type=int, help='', default=60)
    parser.add_argument('--generator_train_optimization_learning_rate', type=float, help='', default=2e-05)
    parser.add_argument('--generator_train_optimization_weight_decay', type=float, help='', default=0.01)
    parser.add_argument('--generator_train_optimization_warmup_steps', type=int, help='', default=100)
    parser.add_argument('--generator_train_optimization_gradient_accumulation_steps', type=int, help='', default=1)
    parser.add_argument('--generator_train_dp_per_example_max_grad_norm', type=float, help='', default=0.1)
    parser.add_argument('--generator_train_dp_target_epsilon', type=float, help='', default=3)
    

    # generator synthetic text generation arguments
    parser.add_argument('--generator_generation_saving_dir', type=str, help='', default='outputs/generator/saved_synthetic_data/')
    parser.add_argument('--generator_generation_syn_file_name', type=str, help='', default='syn_data.csv')
    parser.add_argument('--generator_generation_save_loss_on_target', action='store_true', help='', default=False)
    parser.add_argument('--generator_generation_seed', type=int, help='', default=42)
    parser.add_argument('--generator_generation_parameters_batch_size', type=int, help='', default=128)
    parser.add_argument('--generator_generation_parameters_prompt_sequence_length', type=int, help='', default=64)
    parser.add_argument('--generator_generation_parameters_max_length', type=int, help='', default=128)
    parser.add_argument('--generator_generation_parameters_top_k', type=int, help='', default=200)
    parser.add_argument('--generator_generation_parameters_top_p', type=int, help='', default=1)
    parser.add_argument('--generator_generation_parameters_temperature', type=int, help='', default=1)
    parser.add_argument('--generator_generation_parameters_num_return_sequences', type=int, help='', default=8)
    
    return parser

def add_audit_args(parser):
    # target model training arguments
    parser.add_argument('--audit_target_pretrained_model_name_or_path', type=str, help='', default='gpt2')
    parser.add_argument('--audit_target_saving_dir', type=str, help='', default='outputs/audit_model/target/')
    parser.add_argument('--audit_target_seed', type=int, help='', default=42)
    parser.add_argument('--audit_target_run_name', type=str, help='', default='target_train')
    parser.add_argument('--audit_target_train_with_DP', action='store_true', help='', default=False)
    parser.add_argument('--audit_target_embedding_type', type=str, help='', default='loss_seq')
    parser.add_argument('--audit_target_do_save_weight_initialization', action='store_true', help='', default=False)
    parser.add_argument('--audit_target_optimization_learning_rate', type=float, help='', default=2e-05)
    parser.add_argument('--audit_target_optimization_weight_decay', type=float, help='', default=0.01)
    parser.add_argument('--audit_target_optimization_warmup_steps', type=int, help='', default=100)
    parser.add_argument('--audit_target_optimization_batch_size', type=int, help='', default=64)
    parser.add_argument('--audit_target_optimization_epoch', type=int, help='', default=200)
    parser.add_argument('--audit_target_optimization_save_strategy', type=str, help='', default='steps')
    parser.add_argument('--audit_target_optimization_load_best_model_at_end', action='store_true', help='', default=False)
    parser.add_argument('--audit_target_optimization_save_total_limit', type=str, help='', default=None)
    parser.add_argument('--audit_target_optimization_gradient_accumulation_steps', type=int, help='', default=64)
    parser.add_argument('--audit_target_dp_per_example_max_grad_norm', type=float, help='', default=0.1)
    parser.add_argument('--audit_target_dp_target_epsilon', type=float, help='', default=3)

    # helper model training arguments
    parser.add_argument('--audit_helper_pretrained_model_name_or_path', type=str, help='', default='gpt2')
    parser.add_argument('--audit_helper_saving_dir', type=str, help='', default='outputs/audit_model/helper/')
    parser.add_argument('--audit_helper_seed', type=int, help='', default=42)
    parser.add_argument('--audit_helper_run_name', type=str, help='', default='helper_train')
    parser.add_argument('--audit_helper_embedding_type', type=str, help='', default='loss_seq')
    parser.add_argument('--audit_helper_do_save_weight_initialization', action='store_true', help='', default=False)
    parser.add_argument('--audit_helper_optimization_learning_rate', type=float, help='', default=2e-05)
    parser.add_argument('--audit_helper_optimization_weight_decay', type=float, help='', default=0.01)
    parser.add_argument('--audit_helper_optimization_warmup_steps', type=int, help='', default=100)
    parser.add_argument('--audit_helper_optimization_batch_size', type=int, help='', default=64)
    parser.add_argument('--audit_helper_optimization_epoch', type=int, help='', default=60)
    parser.add_argument('--audit_helper_optimization_save_strategy', type=str, help='', default='epoch')
    parser.add_argument('--audit_helper_optimization_load_best_model_at_end', action='store_true', help='', default=True)
    parser.add_argument('--audit_helper_optimization_save_total_limit', type=int, help='', default=1)
    
    return parser

def add_attack_args(parser):
    # mia attack arguments
    parser.add_argument('--attack_mia_net_type', type=str, help='', default='mix')
    parser.add_argument('--attack_mia_distinguisher_type', type=str, help='', default='GPT2Distinguisher')
    parser.add_argument('--attack_mia_run_name', type=str, help='', default='RMFN_main_table')
    parser.add_argument('--attack_mia_training_args_seed', type=int, help='', default=0)
    parser.add_argument('--attack_mia_training_args_output_dir', type=str, help='', default='outputs/attacks/mia/')
    parser.add_argument('--attack_mia_training_args_which_test', type=str, help='', default='test')
    parser.add_argument('--attack_mia_training_args_max_steps', type=int, help='', default=6000)
    parser.add_argument('--attack_mia_training_args_batch_size', type=int, help='', default=64)
    parser.add_argument('--attack_mia_training_args_warmup_steps', type=int, help='', default=500)
    parser.add_argument('--attack_mia_training_args_weight_decay', type=float, help='', default=0.01)
    parser.add_argument('--attack_mia_training_args_learning_rate', type=float, help='', default=3e-05)
    parser.add_argument('--attack_mia_training_args_reg_coef', type=int, help='', default=0)
    parser.add_argument('--attack_mia_training_args_phase1_max_steps', type=int, help='', default=1500)
    parser.add_argument('--attack_mia_training_args_phase1_batch_size', type=int, help='', default=64)
    parser.add_argument('--attack_mia_training_args_phase1_learning_rate', type=float, help='', default=0.003)
    parser.add_argument('--attack_mia_training_args_phase1_reg_coef', type=int, help='', default=1)
    parser.add_argument('--attack_mia_training_args_logging_steps', type=int, help='', default=10)
    parser.add_argument('--attack_mia_training_args_save_strategy', type=str, help='', default='no')
    parser.add_argument('--attack_mia_training_args_evaluation_strategy', type=str, help='', default='epoch')
    parser.add_argument('--attack_mia_training_args_overwrite_output_dir', action='store_true', help='', default=True)
    parser.add_argument('--attack_mia_training_args_max_fpr', type=float, help='', default=0.1)
    parser.add_argument('--attack_mia_training_args_evaluate_every_n_steps', type=int, help='', default=100)
    parser.add_argument('--attack_mia_training_args_metric_for_best_model', type=str, help='', default='eps')

    # baseline attack arguments
    parser.add_argument('--attack_baseline_net_type', type=str, help='', default='mix')
    parser.add_argument('--attack_baseline_distinguisher_type', type=str, help='', default='GPT2Distinguisher')
    parser.add_argument('--attack_baseline_run_name', type=str, help='', default='RMFN_main_table')
    parser.add_argument('--attack_baseline_training_args_seed', type=int, help='', default=0)
    parser.add_argument('--attack_baseline_training_args_output_dir', type=str, help='', default='outputs/attacks/baseline/')
    parser.add_argument('--attack_baseline_training_args_which_test', type=str, help='', default='test')
    parser.add_argument('--attack_baseline_training_args_max_steps', type=int, help='', default=6000)
    parser.add_argument('--attack_baseline_training_args_batch_size', type=int, help='', default=64)
    parser.add_argument('--attack_baseline_training_args_warmup_steps', type=int, help='', default=500)
    parser.add_argument('--attack_baseline_training_args_weight_decay', type=float, help='', default=0.01)
    parser.add_argument('--attack_baseline_training_args_learning_rate', type=float, help='', default=3e-05)
    parser.add_argument('--attack_baseline_training_args_reg_coef', type=int, help='', default=0)
    parser.add_argument('--attack_baseline_training_args_phase1_max_steps', type=int, help='', default=1500)
    parser.add_argument('--attack_baseline_training_args_phase1_batch_size', type=int, help='', default=64)
    parser.add_argument('--attack_baseline_training_args_phase1_learning_rate', type=float, help='', default=0.003)
    parser.add_argument('--attack_baseline_training_args_phase1_reg_coef', type=int, help='', default=1)
    parser.add_argument('--attack_baseline_training_args_logging_steps', type=int, help='', default=10)
    parser.add_argument('--attack_baseline_training_args_save_strategy', type=str, help='', default='no')
    parser.add_argument('--attack_baseline_training_args_evaluation_strategy', type=str, help='', default='epoch')
    parser.add_argument('--attack_baseline_training_args_overwrite_output_dir', action='store_true', help='', default=True)
    parser.add_argument('--attack_baseline_training_args_max_fpr', type=float, help='', default=0.1)
    parser.add_argument('--attack_baseline_training_args_evaluate_every_n_steps', type=int, help='', default=100)
    parser.add_argument('--attack_baseline_training_args_metric_for_best_model', type=str, help='', default='eps')

    parser.add_argument('--attack_o1_output_dir', type=str, help='', default='outputs/o1/')
    return parser

def init_args():
    parser = argparse.ArgumentParser()
    parser = add_load_args(parser)
    parser = add_base_args(parser)
    parser = add_dataset_args(parser)
    parser = add_generator_args(parser)
    parser = add_audit_args(parser)
    parser = add_attack_args(parser)
    args = parser.parse_args()
    return args


def args_to_nested_dict(args):
    """
    convert the args.Namespace object into a nested dictionary
    """
    nested_config = {
        "base": {},
        "dataset": {},
        "generator": {
            "train": {
                "optimization": {},
                "dp": {}
            },
            "generation": {
                "parameters": {}
            }
        },
        "audit": {
            "target": {
                "optimization": {},
                "dp": {}
            },
            "helper": {
                "optimization": {}
            }
        },
        "attack":{
            "mia": {
                "training_args": {}
            },
            "baseline": {
                "training_args": {}
            },
            "o1": {}
        }
    }
    for key, value in args.__dict__.items():
        key_parts = key.split('_')
        if key.startswith("base"):
            nested_config['base']["_".join(key_parts[1:])] = value
        elif key.startswith("dataset"):
            nested_config['dataset']["_".join(key_parts[1:])] = value
        elif key.startswith("generator_train"):
            if key.startswith("generator_train_optimization"):
                nested_config['generator']['train']['optimization']["_".join(key_parts[3:])] = value
            elif key.startswith("generator_train_dp"):
                nested_config['generator']['train']['dp']["_".join(key_parts[3:])] = value
            else:
                nested_config['generator']['train']["_".join(key_parts[2:])] = value
        elif key.startswith("generator_generation"):
            if key.startswith("generator_generation_parameters"):
                nested_config['generator']['generation']['parameters']["_".join(key_parts[3:])] = value
            else:
                nested_config['generator']['generation']["_".join(key_parts[2:])] = value
        elif key.startswith("audit_target"):
            if key.startswith("audit_target_optimization"):
                nested_config['audit']['target']['optimization']["_".join(key_parts[3:])] = value
            elif key.startswith("audit_target_dp"):
                nested_config['audit']['target']['dp']["_".join(key_parts[3:])] = value
            else:
                nested_config['audit']['target']["_".join(key_parts[2:])] = value
        elif key.startswith("audit_helper"):
            if key.startswith("audit_helper_optimization"):
                nested_config['audit']['helper']['optimization']["_".join(key_parts[3:])] = value
            else:
                nested_config['audit']['helper']["_".join(key_parts[2:])] = value
        elif key.startswith("attack_mia"):
            if key.startswith("attack_mia_training_args"):
                nested_config['attack']['mia']['training_args']["_".join(key_parts[4:])] = value
            else:
                nested_config['attack']['mia']["_".join(key_parts[2:])] = value
        elif key.startswith("attack_baseline"):
            if key.startswith("attack_baseline_training_args"):
                nested_config['attack']['baseline']['training_args']["_".join(key_parts[4:])] = value
            else:
                nested_config['attack']['baseline']["_".join(key_parts[2:])] = value
        elif key.startswith("attack_o1"):
            nested_config['attack']['o1']["_".join(key_parts[2:])] = value
    return nested_config
        
        
