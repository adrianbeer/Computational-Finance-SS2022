import numpy as np
from scipy.stats import norm

# K is redundant... f is already defined and K isn't an input
# parameter to f (here at least).
def Eu_Option_BS_MC(S0, r, sigma, T, M, f, K=None):
    X = np.random.normal(loc=0, scale=1, size=M)
    S_T = S0*np.exp((r-sigma**2/2)*T + sigma*np.sqrt(T)*X)
    assert len(S_T) == M
    f_S_T = list(map(f, S_T))
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


def BlackScholes_EUCall(S0, r, sigma, T, K, *args, **kwargs):
    t=0
    d1 = (np.log(S0/K) + (r + sigma**2/2)*(T-t))/(sigma*(np.sqrt(T-t)))
    d2 = d1 - sigma*(np.sqrt(T-t))
    C = S0*(norm.cdf(d1)) - K*np.exp(-r*(T-t))*norm.cdf(d2)
    return C

K = 100
params = dict(K=100, f=lambda x: max([x-K, 0]), S0=110, r=0.03, sigma=0.2, T=1, M=10000)
MC_price, (c_lower, c_upper) = Eu_Option_BS_MC(**params)
BS_price = BlackScholes_EUCall(**params)
print(f"MC price: {MC_price:.4f} with 95% conf interval: ({c_lower:.4f}, {c_upper:.4f})")
print(f"BS price: {BS_price:.4f}" )
# The option prices differ slightly.
