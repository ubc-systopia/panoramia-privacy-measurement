import argparse
import numpy as np
import torch
import math
import csv
import random
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset, Subset
import torch.nn as nn
import os
from os.path import join, basename, dirname, exists
from time import time, sleep
#import wandb
from sklearn.model_selection import train_test_split
from scipy.stats import bernoulli
import scipy

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=42, type=int)
parser.add_argument("--output_file", type=str, default="o1_results.csv", help='Output CSV file name') # new arg for O(1) logging
args = parser.parse_args()

def bernoulli_sample_datasets(single_dataset, p=0.5):
    """
    Sample datasets using Bernoulli trials to determine inclusion in the member or non-member dataset.
    
    Supports both PyTorch datasets and NumPy arrays.
    
    Args:
        single_dataset (Dataset or tuple): The original dataset from which to sample.
                                           If a tuple, it should be (data, targets) where both are NumPy arrays.
        p (float, optional): Probability of a data point being included in the member dataset. Default is 0.5.

    Returns:
        tuple: (sampled_member_dataset, sampled_non_member_dataset) where:
            - sampled_member_dataset (Subset or tuple): Subset of the original dataset where data points were included based on Bernoulli trials.
            - sampled_non_member_dataset (Subset or tuple): Subset of the original dataset where data points were excluded based on Bernoulli trials.
    """
    
    # Handle NumPy arrays by converting them to tensors
    if isinstance(single_dataset, tuple):
        data, targets = single_dataset
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets)
        single_dataset = torch.utils.data.TensorDataset(data, targets)
    
    # Total number of data points in the original dataset
    total_length = len(single_dataset)
    
    # Create Bernoulli trials for each data point
    bernoulli_trials = torch.bernoulli(torch.full((total_length,), p))

    # Separate indices based on Bernoulli trials
    member_indices = [i for i in range(total_length) if bernoulli_trials[i]]
    non_member_indices = [i for i in range(total_length) if not bernoulli_trials[i]]
    
    # Create subsets based on the indices
    sampled_member_dataset = Subset(single_dataset, member_indices)
    sampled_non_member_dataset = Subset(single_dataset, non_member_indices)
    
    return sampled_member_dataset, sampled_non_member_dataset
    
def p_value_DP_audit(m, r, v, eps, delta=0):
    """
    Args:
        m = number of examples, each included independently with probability 0.5
        r = number of guesses (i.e. excluding abstentions)
        v = number of correct guesses by auditor
        eps,delta = DP guarantee of null hypothesis
    Returns:
        p-value = probability of >=v correct guesses under null hypothesis
    """
    assert 0 <= v <= r <= m
    assert eps >= 0
    assert 0 <= delta <= 1
    q = 1/(1+math.exp(-eps)) # accuracy of eps-DP randomized response
    beta = scipy.stats.binom.sf(v-1, r, q) # = P[Binomial(r, q) >= v]
    alpha = 0
    sum = 0 # = P[v > Binomial(r, q) >= v - i]
    for i in range(1, v + 1):
       sum = sum + scipy.stats.binom.pmf(v - i, r, q)
       if sum > i * alpha:
           alpha = sum / i
    p = beta  #+ alpha * delta * 2 * m
    # print("p", p)
    return min(p, 1)

def get_eps_audit(m, r, v,  p, delta):
    """
    Args:
        m = number of examples, each included independently with probability 0.5
        r = number of guesses (i.e. excluding abstentions)
        v = number of correct guesses by auditor
        p = 1-confidence e.g. p=0.05 corresponds to 95%
    Returns:
        lower bound on eps i.e. algorithm is not (eps,delta)-DP
    """
    assert 0 <= v <= r <= m
    assert 0 <= delta <= 1
    assert 0 < p <= 1
    eps_min = 0 # maintain p_value_DP(eps_min) < p
    eps_max = 1 # maintain p_value_DP(eps_max) >= p

    while p_value_DP_audit(m, r, v, eps_max, delta) < p: eps_max = eps_max + 1
    for _ in range(30): # binary search
        eps = (eps_min + eps_max) / 2
        if p_value_DP_audit(m, r, v, eps, delta) < p:
            eps_min = eps
        else:
            eps_max = eps
    return eps_min

def find_O1_pred(member_loss_values, non_member_loss_values, delta = 0.):
    """
    Args:
        member_loss_values: NumPy array containing member loss values
        non_member_loss_values: NumPy array containing non_member loss values
    Returns:
     best_eps: largest audit (epsilon) value that can be returned for a particular p value
    """
    
    # Create labels for real and generated loss values
    member_labels = np.ones_like(member_loss_values)
    non_member_labels = np.zeros_like(non_member_loss_values)

    # Concatenate loss values and labels
    all_losses = np.concatenate((member_loss_values, non_member_loss_values))
    all_labels = np.concatenate((member_labels, non_member_labels))
    
    # Step 1: Find t_pos that maximizes precision for positive predictions
    best_precision = 0
    best_t_pos = 0
    threshold_range = np.arange(np.min(all_losses), np.max(all_losses) + 0.01, 0.01)
    results, recall = [], []
    best_accuracy = 0
    best_t_neg = 0
    total_predictions = 0
    correct_predictions = 0
    best_eps = 0
    p = 0.05
    for t_pos in threshold_range:
        positive_predictions = all_losses[all_losses <= t_pos]
        if len(positive_predictions) == 0:
            continue

        true_positives = np.sum(all_labels[all_losses <= t_pos] == 1)
        
        eps = get_eps_audit(len(all_labels), len(positive_predictions), true_positives, p, delta)
        precision = true_positives / len(positive_predictions)
        if eps > best_eps:
            print("EPSILON UPDATE:", eps)
            best_eps = eps
            best_t_pos = t_pos
        recalls = true_positives / np.sum(all_labels == 1)
        recall.append(recalls)
        
        # Step 2: With t_pos fixed, find t_neg that maximizes overall accuracy
        for t_neg in reversed(threshold_range):
            if t_neg <= best_t_pos:
                break
            confident_predictions = all_losses[(all_losses <= best_t_pos) | (all_losses >= t_neg)]
            r = len(confident_predictions)
            mask_pos = (confident_predictions <= best_t_pos) & (all_labels[(all_losses <= best_t_pos) | (all_losses >= t_neg)] == 1)
            mask_neg = (confident_predictions >= t_neg) & (all_labels[(all_losses <= best_t_pos) | (all_losses >= t_neg)] == 0)

            v = np.sum(np.logical_or(mask_pos, mask_neg))

            if r > 0:
                accuracy = v / r
                eps = get_eps_audit(len(all_labels), r, v, p, delta)
                if eps > best_eps:
                    best_eps = eps
                    best_t_neg = t_neg
                    total_predictions = r
                    correct_predictions = v
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
            
            results.append({
                't_pos': t_pos,
                't_neg': best_t_neg,
                'best_precision': precision,
                'best_accuracy': best_accuracy,
                'recall': recall,
                'total_predictions': r,
                'correct_predictions': v
            })
    print(f"Best eps: {best_eps} with thresholds (t_neg, t_pos): ({best_t_neg}, {best_t_pos})")
    print(f"Best precision for t_pos: {best_precision} with t_pos: {best_t_pos}")
    print(f"Best accuracy: {best_accuracy} with thresholds (t_neg, t_pos): ({best_t_neg}, {best_t_pos})")
    
    # Save results to CSV file
    output_csv_path = args.output_file
    with open(output_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['t_pos', 't_neg', 'best_precision', 'best_accuracy', 'recall', 'total_predictions', 'correct_predictions']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)

    return total_predictions, correct_predictions, len(all_losses)
    
