import scipy
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--delta', type=float, default=0.)
parser.add_argument('--gamma', type=float, default=0.)

args = parser.parse_args()
DELTA = args.delta
GAMMA = args.gamma


# m = number of examples, each included independently with probability 0.5
# r = number of guesses (i.e. excluding abstentions)
# v = number of correct guesses by auditor
# eps,delta = DP guarantee of null hypothesis
# output: p-value = probability of >=v correct guesses under null hypothesis
def p_value_DP_audit(m, r, v, eps, delta, gamma):
    assert 0 <= v <= r <= m
    assert eps >= 0
    assert 0 <= delta <= 1
    q = 1/(1+math.exp(-eps)) # accuracy of eps-DP randomized response
    beta = scipy.stats.binom.sf(v-1, r, q) # = P[Binomial(r, q) >= v]
    if delta == 0:
        p = beta # + alpha * delta * 2 * m
    else:
        alpha = 0
        sum = 0 # = P[v > Binomial(r, q) >= v - i]
        for i in range(1, v + 1):
           sum = sum + scipy.stats.binom.pmf(v - i, r, q)
           if sum > i * alpha:
               alpha = sum / i
        if gamma == 0:
            p = beta + alpha * delta * 2 * m
        else:
            p = beta + alpha * 2 * m * (gamma + delta - gamma * delta)
    return min(p, 1)


# m = number of examples, each included independently with probability 0.5
# r = number of guesses (i.e. excluding abstentions)
# v = number of correct guesses by auditor
# p = 1-confidence e.g. p=0.05 corresponds to 95%
# output: lower bound on eps i.e. algorithm is not (eps,delta)-DP
def get_eps_audit(m, r, v, p, delta, gamma):
    assert 0 <= v <= r <= m
    assert 0 <= delta <= 1
    assert 0 < p <= 1
    eps_min = 0 # maintain p_value_DP(eps_min) < p
    eps_max = 1 # maintain p_value_DP(eps_max) >= p

    while p_value_DP_audit(m, r, v, eps_max, delta, gamma) < p: eps_max = eps_max + 1
    for _ in range(30): # binary search
        eps = (eps_min + eps_max) / 2
        if p_value_DP_audit(m, r, v, eps, delta, gamma) < p:
            eps_min = eps
        else:
            eps_max = eps
    return eps_min
