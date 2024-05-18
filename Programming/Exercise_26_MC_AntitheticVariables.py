# GROUP QF 26
# NAMES: Adrian Beer, Dieu Hue Vu
# EXERCISE 21

import numpy as np
from scipy.stats import norm
#np.random.seed(123123)

"""
Using antithetic variables to reduce the variance of MC-estimators
"""

def BlackScholes_EUCall(S0, r, sigma, T, K, *args, **kwargs):
    t=0
    d1 = (np.log(S0/K) + (r + sigma**2/2)*(T-t))/(sigma*(np.sqrt(T-t)))
    d2 = d1 - sigma*(np.sqrt(T-t))
    C = S0*(norm.cdf(d1)) - K*np.exp(-r*(T-t))*norm.cdf(d2)
    return C


# Function from Ex 21
def Eu_Option_BS_MC(S0, r, sigma, T, M, g, K=None):
    X = np.random.normal(loc=0, scale=1, size=M)
    S_T = S0*np.exp((r-sigma**2/2)*T + sigma*np.sqrt(T)*X)
    assert len(S_T) == M
    f_S_T = list(map(g, S_T))
    avg_f_S_T = np.mean(f_S_T)
    V_0 = np.exp(-r*T)*avg_f_S_T

    # Calculate confidence intervals
    alpha = 0.95
    sigma_hat = np.std(f_S_T, ddof=1)
    radius = 1.96*np.sqrt(sigma_hat**2/M)
    c_lower, c_upper = avg_f_S_T-radius, avg_f_S_T+radius
    c_lower = np.exp(-r*T)*c_lower
    c_upper = np.exp(-r*T)*c_upper
    return V_0, (c_lower, c_upper)


# Function using Antithetic variables

def BS_EuOption_MC_AV (S0, r, sigma, T, K=None, M=None, g=None):
    X = np.random.normal(loc=0, scale=1, size=M)
    X = np.concatenate([X, -X])  # Antithetic variables
    S_T = S0*np.exp((r-sigma**2/2)*T + sigma*np.sqrt(T)*X)
    f_S_T = list(map(g, S_T))
    print(f"Cov(V_hat, V_minus) = {np.cov(f_S_T[:M], f_S_T[M:])[0,1]/M:.4f}")
    avg_f_S_T = np.mean(f_S_T)
    V_0 = np.exp(-r*T)*avg_f_S_T
    # It's the same.
    #print(1/(2*M)*sum((f_S_T-np.mean(f_S_T))**2))
    #print(1/(2*M)*np.sum((f_S_T[:M]+f_S_T[M:]-np.mean(f_S_T[:M]+f_S_T[M:]))**2))

    # Calculate confidence intervals
    alpha = 0.95
    sigma_hat = np.std(f_S_T, ddof=1)
    radius = 1.96*np.sqrt(sigma_hat**2/len(f_S_T))
    c_lower, c_upper = avg_f_S_T-radius, avg_f_S_T+radius
    c_lower = np.exp(-r*T)*c_lower
    c_upper = np.exp(-r*T)*c_upper
    return V_0, (c_lower, c_upper)

K = 100
def g(x): return max([x-K, 0])
params = dict(S0=110, r=0.03, sigma=0.2, T=1,  M=100000, g=g)

V0, (c_l, c_u) = BS_EuOption_MC_AV(**params)
print(f"Antithetic MC Simulation price: {V0:.4f} with 95% conf interval: ({c_l:.4f}, {c_u:.4f}) (width: {(c_u - c_l):.4f}), sample size: {2*params['M']}")

params2 = dict(S0=110, r=0.03, sigma=0.2, T=1,  M=200000, g=g)  # Doubling the number of simulations
V0, (c_l, c_u) = Eu_Option_BS_MC(**params2)
print(f"Standard* MC Simulation price: {V0:.4f} with 95% conf interval: ({c_l:.4f}, {c_u:.4f}) (width: {(c_u - c_l):.4f}), sample size {2*params['M']}")

V0, (c_l, c_u) = Eu_Option_BS_MC(**params)
print(f"Standard MC Simulation price: {V0:.4f} with 95% conf interval: ({c_l:.4f}, {c_u:.4f}) (width: {(c_u - c_l):.4f}), sample size {params['M']}")

BS_price = BlackScholes_EUCall(**params, K=K)
print(f"BS price: {BS_price:.4f}")

print('''\n
We see that the correlation between V_hat and V_minus is negligible (still negative though), hence there is only a very
weak reduction in the variance of the estimate in the Antithetic MC simulation compared to the Standard* MC approach, 
which has the same sample size.
''')