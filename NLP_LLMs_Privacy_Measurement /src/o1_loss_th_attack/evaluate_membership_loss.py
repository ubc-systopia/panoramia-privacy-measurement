import os

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    
def compute_and_save_membership_loss(dm, audit_model, output_dir, batch_size=32):
    _, _, test_dataset = dm.get_mia_datasets()

    # Move the model to the specified device
    audit_model.model.to(device)

    # Create a DataLoader for the test dataset
    test_dataloader = DataLoader(
        test_dataset, 
        sampler=SequentialSampler(test_dataset),  # Pull out batches sequentially.
        batch_size=batch_size
    )
    
    scores = []

    # Iterate through the batches in the test dataloader
    for batch in test_dataloader:
        b_input_ids = batch["input_ids"].to(device)
        b_masks = batch["attention_mask"].to(device)
        b_labels = batch["labels"].to(device)

        # Get the audit loss or score for the batch
        audit_loss_batch = audit_model.get_embedding(b_input_ids, b_masks, b_input_ids).cpu().numpy().tolist()
        scores += audit_loss_batch
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    

    labels = test_dataset['labels'].numpy()

    # Separate scores based on labels
    member_loss_values = scores[labels == 1.]
    non_member_loss_values = scores[labels == 0.]

    # Save the separated scores to .npy files
    np.save(
        os.path.join(output_dir, "O(1)_members_loss.npy"),
        np.array(member_loss_values)
    )

    np.save(
        os.path.join(output_dir, "O(1)_nonmembers_loss.npy"),
        np.array(non_member_loss_values)
    )

    
    return scores