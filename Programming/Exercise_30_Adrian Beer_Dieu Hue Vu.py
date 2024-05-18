# GROUP QF 26
# NAMES: Adrian Beer, Dieu Hue Vu
# EXERCISE 30

import numpy as np
import scipy
from scipy.stats import norm
from functools import partial

'''
Calculating the Delta/Hedge of a European option via MC methods, infinitesimal perturbation
'''

def get_phi1_t(St, r, sigma, g, T, t, N, K):
    return norm.cdf((np.log(St/K)+r*(T-t)+sigma**2/2*(T-t))/(sigma*np.sqrt(T-t)))


def EuOptionHedge_BS_MC_IP(St, r, sigma, g, T, t, N):
    X = np.random.normal(0, 1, N)

    def Z(s, x):
        Vts = np.exp(-r*(T-t)) * \
        g(s * np.exp((r - sigma**2/2)*(T-t) + sigma*np.sqrt(T-t)*x))
        return np.mean(Vts)

    Ds_Z = [scipy.misc.derivative(func=partial(Z, x=x), x0=St) for x in X]
    print(f"Estimate 95% CI radius: {1.96*np.std(Ds_Z)/N**0.5:.4f}")
    return np.mean(Ds_Z)


K=100
def g(x):
    return np.maximum(0, x-K)
p = dict(t=0, St=110, r=0.03, sigma=0.2, T=1, N=10000, g=g)


phi1_t_hat = EuOptionHedge_BS_MC_IP(**p)
print(f"Estimates phi1 {phi1_t_hat:.3f}")

phi1_t = get_phi1_t(**p, K=K)
print(f"True phi1 {phi1_t:.3f}")
print("Pretty close...")
