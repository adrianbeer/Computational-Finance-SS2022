import numpy as np
from scipy.stats import norm
from cmath import exp

# GROUP QF 26
# NAMES: Adrian Beer, Dieu Hue Vu
# EXERCISE 21


def Eu_Option_Quanto_MC(S0, r, sigma, T, M, f, K=None, X=None):
    if not X is None: X = np.random.normal(loc=0, scale=1, size=M)
    S_T = np.exp(-r*T) * S0 * np.exp((r - sigma ** 2 / 2) * T + sigma * np.sqrt(T) * X)
    assert len(S_T) == M
    f_S_T = np.array(list(map(f, S_T)))
    V_0 = np.mean(f_S_T)

    # Calculate confidence intervals
    var_hat = np.var(f_S_T, ddof=1)/M
    radius = 1.96*np.sqrt(var_hat)  # alpha = 0.95
    c_lower, c_upper = V_0-radius, V_0+radius
    return V_0, (c_lower, c_upper)


def BlackScholes_EUCall(S0, r, sigma, T, K, *args, **kwargs):
    t = 0
    d1 = (np.log(S0/K) + (r + sigma**2/2)*(T-t))/(sigma*(np.sqrt(T-t)))
    d2 = d1 - sigma*(np.sqrt(T-t))
    C = S0*(norm.cdf(d1)) - K*exp(-r*(T-t))*norm.cdf(d2)
    return C


def BS_EuOption_MC_CV(S0, r, sigma, T, K, M, X=None):
    if X is None: X = np.random.normal(loc=0, scale=1, size=M)
    def y(x): return max([x - K, 0])

    S_T = np.exp(-r*T) * S0 * np.exp((r - sigma ** 2 / 2) * T + sigma * np.sqrt(T) * X)
    Y = np.array(list(map(y, S_T)))
    f_X = Y * S_T
    assert len(f_X) == M

    beta = np.cov(f_X, Y)[0,1]/np.var(Y)

    EY = np.real(BlackScholes_EUCall(S0, r, sigma, T, K))
    V_CV = np.mean(f_X - beta*Y) + beta*EY

    # Calculate confidence intervals
    var_hat_V_CV = np.var(f_X - beta*Y, ddof=1)/M
    radius = 1.96*np.sqrt(var_hat_V_CV)  # alpha = 0.95
    c_lower, c_upper = V_CV-radius, V_CV+radius
    return V_CV, (c_lower, c_upper)

# Some thing is wrong here since often one estimated price isn't even in the confidence band of the other.

M= 100000
X = np.random.normal(loc=0, scale=1, size=M)
params = dict(S0=110, r=0.03, sigma=0.2, T=1,  M=M, K=100, X=X)
V0, (c_l, c_u) = BS_EuOption_MC_CV(**params)

print(f"CV-MC Simulation price of a self-quanto call: "
      f"\n{V0:.4f} with 95% conf interval: ({c_l:.4f}, {c_u:.4f}) (width: {(c_u - c_l):.4f})\n")

V1, (c_l, c_u) = Eu_Option_Quanto_MC(**params, f=lambda x: max([x-params['K'], 0])*x)
print(f"Plain MC Simulation price of a self-quanto call: "
      f"\n{V1:.4f} with 95% conf interval: ({c_l:.4f}, {c_u:.4f}) (width: {(c_u - c_l):.4f})")
