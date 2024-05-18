# GROUP QF 26
# NAMES: Adrian Beer, Dieu Hue Vu
# EXERCISE 30

import numpy as np
from matplotlib import pyplot as plt

np.random.seed(0)

'''
Valuation of European options in the Heston model using the Euler method
'''

def generate_Heston_path(S0, r, gamma0, kappa, lambda_, sigma_tilde, T, m, *args, **kwargs):
    Y = np.ndarray(shape=(m,))
    Y[0] = S0
    gamma = gamma0
    dt = T/m
    # Using two different Brownian motions, one for the stock, one for the stochastic volatility
    dW = np.random.normal(0, dt**0.5, m)
    dW_tilde = np.random.normal(0, dt**0.5, m)
    for i in range(1,m):
        # Calculate gamma realization first
        gamma = gamma + (kappa - lambda_*gamma)*dt + gamma**0.5 *sigma_tilde*dW_tilde[i]
        # If gamma happens to be negative we just set it to zero
        gamma = max([0, gamma])
        # Then calculate stock realization
        Y[i] = Y[i-1] + Y[i-1]*r*dt + Y[i-1]* gamma**0.5 *dW[i]
    return Y


def Heston_EuCall_MC_Euler (g, M, *args, **kwargs):
    payoffs = np.ndarray(shape=(M,))
    T = kwargs['T']
    r = kwargs['r']
    # Generate M different Heston paths and estimate option value with them
    for i in range(M):
        payoffs[i] = g(generate_Heston_path(*args, **kwargs)[-1])
    V0 = np.exp(-r*T)*np.mean(payoffs)

    # Calculate confidence intervals
    std = np.std(payoffs)
    cutoff = 1.96
    radius = cutoff*std/np.sqrt(M)
    CIl, CIr = V0 - radius, V0 + radius
    return V0, CIl, CIr


def g(x): return np.maximum(0, x-100)
p = dict(S0=100, r=0.05, gamma0=0.2**2, kappa=0.5,
         lambda_=2.5, sigma_tilde=1, T=1, g=g, M=10000, m=250)

V0_hat, CIl, CIr =  Heston_EuCall_MC_Euler(**p)
print(f"Estimate: {V0_hat:.3f} with 95% CI: ({CIl:.3f}, {CIr:.3f})")

