# GROUP QF 26
# NAMES: Adrian Beer, Dieu Hue Vu
# EXERCISE 29

import numpy as np
from scipy.stats import norm
import scipy
import math
from matplotlib import pyplot as plt

''' Pricing a deep out-of-the-money European call option by Monte-Carlo with
importance sampling.'''

def BlackScholes_EuCall(t, S_t, r, sigma, T, K, *args, **kwargs):
    d_1 = (math.log(S_t / K) + (r + 1 / 2 * math.pow(sigma, 2)) * (T - t)) / (sigma * math.sqrt(T - t))
    d_2 = d_1 - sigma * math.sqrt(T - t)
    phi = scipy.stats.norm.cdf(d_1)
    C = S_t * phi - K * math.exp(-r * (T - t)) * scipy.stats.norm.cdf(d_2)
    return C


def BS_EuCall_MC_IS(S0, r, sigma, K, T, mu, N, alpha):
    Y = np.random.normal(mu, 1, N)

    # Estimate option value at time 0 using importance sampling with variable Y
    V0s = np.exp(-r*T-Y*mu+mu**2/2)*\
          np.maximum(0, S0*np.exp((r-sigma**2/2)*T + sigma*np.sqrt(T)*Y) - K)
    V0 = np.mean(V0s)

    # Compute confidence intervals
    std = np.std(V0s)
    Cr_norm = norm.ppf(1-(1-alpha)/2)
    radius = Cr_norm*std/np.sqrt(N)
    CIl, CIr = V0 - radius, V0 + radius
    return V0, CIl, CIr


p = dict(S0=110, r=0.03, sigma=0.2, K=220, T=1, N=10_000, alpha=0.95)
d = (np.log(p['K']/p["S0"]) - (0.03-0.2**2/2))/(0.2)

C = BlackScholes_EuCall(**p, t=0, S_t=p['S0'])

mus = np.linspace(start=0, stop=d*2, num=50)
estimates = [BS_EuCall_MC_IS(**p, mu=mu) for mu in mus]
#print(estimates)
plt.plot(mus, np.array(estimates))
plt.xlabel("$\mu$")
plt.title("Option Price Estimates With Different $\mu$s")
plt.axhline(C, color='red')
plt.axvline(d, color='black')
plt.gca().legend(('$\hat V0$', 'CI lower','CI upper', "True V0 (Black-Scholes)", "d"))

plt.show()
