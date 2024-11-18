import logging
import os

from easydict import EasyDict
import torch
import pandas as pd
from transformers import set_seed

from src.datasets.datamodule import PANORAMIADataModule
from src.generator.utils import check_length_to_block_size

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

def generate_synthetic_samples(
    config: EasyDict, 
    dm: PANORAMIADataModule,
    model: torch.nn.Module
):
    # moving model to the available device. torch.nn.module.to happens in-place
    model.to(device)

    # where and file name to save the generated synthetic data
    output_dir_path = config.generator.generation.saving_dir

    output_file_path = os.path.join(output_dir_path, config.generator.generation.syn_file_name)

    # deleting the previous generated fake samples, if existed
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    # creating the directories
    os.makedirs(output_dir_path)

    # loading the generation parameters
    config_gen_params = config.generator.generation.parameters

    # setting seed for reproducibility
    set_seed(config.generator.generation.seed)


    # getting the prompt dataloader
    prompt_dataloader = dm.get_generator_prompt_dataloaders(batch_size=config_gen_params.batch_size)

    logging.info(f"Total number of prompt samples: {len(prompt_dataloader.dataset)}")

    # set the model to evaluation mode
    model.eval()

    # synthetic data structure
    synthetic_data = {
    "Prompt": [],
    "text": [],
    "Length-Generated": []
    }

    # initializing the dataframe consisting the syn data
    df = pd.DataFrame(synthetic_data)

    # loading length of sequence of the prompt
    prompt_sequence_length = config_gen_params.prompt_sequence_length


    for batch in prompt_dataloader:
        # taking a prefix of the prompt of size prompt_sequence_length and moving to the available device
        inputs_input_ids = batch["input_ids"][:, :prompt_sequence_length].to(device) #shape: (batch_size, prompt_sequence_length)
        inputs_attention_mask = batch["attention_mask"][:, :prompt_sequence_length].to(device)
        
        # no gradients
        with torch.no_grad():
            
            # generating texts with the given parameters
            outputs = model.generate(
                inputs_input_ids,
                attention_mask=inputs_attention_mask,
                max_length=config_gen_params.max_length, 
                do_sample=True,
                top_k=config_gen_params.top_k,
                top_p=config_gen_params.top_p,
                temperature=config_gen_params.temperature,
                num_return_sequences=config_gen_params.num_return_sequences
            )            
        

        output_text = dm.tokenizer.batch_decode(
            outputs[:, prompt_sequence_length:].contiguous().view(-1, dm.block_size), #extracting the generated part from the input prompt
            skip_special_tokens=False, 
            clean_up_tokenization_spaces=False
        )
        
        # attention_mask = torch.ones_like(outputs[:, prompt_sequence_length:config_gen_params.max_length])
        
        if config.generator.generation.save_loss_on_target:
            # Also save the loss values of the generated samples on the target and helper model
            raise NotImplementedError
        
        # appending the generated samples to the dataframe
        for i in range(len(batch['input_ids']) ): # the last batch might not be equal to batch_size
            for j in range(config_gen_params.num_return_sequences):
                for k in range((config_gen_params.max_length - prompt_sequence_length)//dm.block_size):

                    row = {
                        "Prompt":  [dm.tokenizer.decode(batch["input_ids"][i, :prompt_sequence_length])],
                        "text": [output_text[config_gen_params.num_return_sequences * ((config_gen_params.max_length - prompt_sequence_length)//dm.block_size) * i +  (((config_gen_params.max_length - prompt_sequence_length)//dm.block_size) * j) + k]],
                        "Length-Generated": [len(outputs[i])] #the length of generated text is not necessarily deterministic and might vary
                    }

                    df = pd.concat([df, pd.DataFrame.from_dict(row)], ignore_index=True)
    
    # decoding and ecoding back are not necessarily inverse of each other. Hence, we check that samples would be of length after encoding.
    df = check_length_to_block_size(
        df,
        dm.tokenizer,
        config.dataset.block_size
    )

    # shuffling the rows (if num_return_sequences > 1)
    df = df.sample(frac=1).reset_index(drop=True)

    df.to_csv(output_file_path)
