import scipy
import pandas as pd
import matplotlib.pyplot as plt
import math
import csv
import numpy as np

# reference: Privacy Auditing with One (1) Training Run (Thomas Steinke, Milad Nasr, Matthew Jagielski)
# arXiv:2305.08846 
# m = number of examples, each included independently with probability 0.5
# r = number of guesses (i.e. excluding abstentions) (only positive guesses for Panoramia)
# v = number of correct guesses by auditor (only positive guesses for Panoramia)
# eps,delta = DP guarantee of null hypothesis
# output: p-value = probability of >=v correct guesses under null hypothesis
def p_value_DP_audit(m, r, v, eps, delta):
    assert 0 <= v <= r <= m
    assert eps >= 0
    assert 0 <= delta <= 1
    q = 1/(1+math.exp(-eps)) # accuracy of eps-DP randomized response
    beta = scipy.stats.binom.sf(v-1, r, q)/r # = P[Binomial(r, q) >= v]
    alpha = 0
    sum = 0 # = P[v > Binomial(r, q) >= v - i]
    #for i in range(1, v + 1):
    #    sum = sum + scipy.stats.binom.pmf(v - i, r, q)
    #    if sum > i * alpha:
    #        alpha = sum / i
    p = beta # + alpha * delta * 2 * m
    return min(p, 1)


# m = number of examples, each included independently with probability 0.5
# r = number of guesses (i.e. excluding abstentions)
# v = number of correct guesses by auditor
# p = 1-confidence e.g. p=0.05 corresponds to 95%
# output: lower bound on eps i.e. algorithm is not (eps,delta)-DP
def get_eps_audit(m, r, v,  p, delta):
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


def main(csv_file, m, r, delta, p):
    data_ep = pd.read_csv(csv_file)
    epsilons = []
    recalls, mia_TP, mia_FP, total_guesses = [], [], [], []

    with open(csv_file, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            recalls.append(float(row['Recall']))
            mia_TP.append(float(row['TP']))
            mia_FP.append(float(row['FP']))
            total_guesses.append(float(row['Number of Total Guesses']))

    for i in range(len(total_guesses)):
        v = total_guesses[i]
        eps_min = get_eps_audit(m, r, v, p, delta)
        epsilons.append(eps_min * 2)

    max_epsilon = max(epsilons)

    plt.figure(figsize=(10, 8))
    plt.plot(recalls, epsilons, linestyle='-', color='red', linewidth=2.5, label='PANORAMIA: MultiLabel CNN-E100')
    plt.axhline(max_epsilon, color='red', linestyle='--')
    plt.plot(np.NaN, np.NaN, 'black', linewidth=2.5, linestyle='--', label='empirical maximum value')

    plt.xlim((0), (1))
    plt.ylim((0), (4))
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.tick_params(axis='both', which='both', length=5)
    plt.xlabel('Recall', fontsize='28')
    plt.ylabel('Measurement of $c_{lb}$ or $\tilde\epslion$', fontsize='28')
    plt.legend(fontsize='24', loc='upper right')
    plt.tight_layout()
    plt.showfig()
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DP Audit Calculation")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to the CSV file")
    parser.add_argument("--m", type=int, required=True, help="Number of examples")
    parser.add_argument("--r", type=int, required=True, help="Number of guesses")
    parser.add_argument("--delta", type=float, default=0, help="Delta for DP guarantee")
    parser.add_argument("--p", type=float, default=0.05, help="Confidence level")
    
    args = parser.parse_args()
    
    main(args.csv_file, args.m, args.r, args.delta, args.p)
