import os
import random
import math

import numpy as np
import torch
import scipy
from sklearn.metrics import confusion_matrix

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def enable_full_determinism(seed: int):
    """
    inspired by https://github.com/huggingface/transformers/blob/v4.36.1/src/transformers/trainer_utils.py#L58
    """
    set_seed(seed)

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)

    # Enable CUDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_wand_group(config, train_baseline):
    attack_num_train = config.dataset.mia_num_train
    if train_baseline:
        return f"baseline_k_{attack_num_train}"
    else:
        return f"mia_k_{attack_num_train}"


# TODO: this is temporary. Should be moved to somewhere more reasonable than here

#m = number of examples, each included independently with probability 0.5
#r = number of guesses (i.e. excluding abstentions)
#v = number of correct guesses by auditor
#eps,delta = DP guarantee of null hypothesis
#output: p-value = probability of >=v correct guesses under null hypothesis
def p_value_DP_audit(m, r, v, eps, delta):
    assert 0 <= v <= r <= m
    assert eps >= 0
    assert 0 <= delta <= 1
    q = 1/(1+math.exp(-eps)) # accuracy of eps-DP randomized response
    beta = scipy.stats.binom.sf(v-1, r, q) # = P[Binomial(r, q) >= v]
    alpha = 0
    sum = 0 # = P[v > Binomial(r, q) >= v - i]
    # for i in range(1, v + 1):
    #    sum = sum + scipy.stats.binom.pmf(v - i, r, q)
    #    if sum > i * alpha:
    #        alpha = sum / i
    p = beta # + alpha * delta * 2 * m
    return min(p, 1)


#m = number of examples, each included independently with probability 0.5
#r = number of guesses (i.e. excluding abstentions)
#v = number of correct guesses by auditor
#p = 1-confidence e.g. p=0.05 corresponds to 95%
#output: lower bound on eps i.e. algorithm is not (eps,delta)-DP
def get_eps_audit(m, r, v, delta, p):
    assert 0 <= v <= r <= m
    assert 0 <= delta <= 1
    assert 0 < p < 1
    eps_min = 0 # maintain p_value_DP(eps_min) < p
    eps_max = 1 # maintain p_value_DP(eps_max) >= p
    while p_value_DP_audit(m, r, v, eps_max, delta) < p: 
        eps_max = eps_max + 1
    
    for _ in range(30): # binary search
        eps = (eps_min + eps_max) / 2
        if p_value_DP_audit(m, r, v, eps, delta) < p:
            eps_min = eps
        else:
            eps_max = eps
    return eps_min

def get_max_eps_validation(preds: np.array, labels: np.array, dataset_size: int, delta = 0, audit_CI = 0.05):
    eps_lbs = []
    thresholds = np.linspace(0, 1, 1000)
    for th in thresholds:
        hard_preds = (preds > th).astype('float')
        tn, fp, fn, tp = confusion_matrix(labels, hard_preds).ravel()
        eps_lb = get_eps_audit(dataset_size, fp+tp, tp, delta, audit_CI/2)
        eps_lbs.append(eps_lb)
    return np.max(np.array(eps_lbs))
    