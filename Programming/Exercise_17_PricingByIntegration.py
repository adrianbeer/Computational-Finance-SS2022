import math
import numpy as np
import scipy.integrate as integrate
from matplotlib import pyplot as plt

# compute Black-Scholes price by integration
def BS_Price_Int(S0, r, sigma, T, f):
    # define integrand as given in the exercise
    def integrand(x):
        return 1 / math.sqrt(2 * math.pi) * f(
            S0 * math.exp((r - 0.5 * math.pow(sigma, 2)) * T + sigma * math.sqrt(T) * x)) * math.exp(-r * T) * math.exp(
            -1 / 2 * math.pow(x, 2))

    # perform integration
    I = integrate.quad(integrand, -np.inf, np.inf)
    # return value of the integration
    return I[0]


# a)
def BS_Greeks_num(r, sigma, S0, T, g ,eps):
    def f(x): return BS_Price_Int(S0=x, r=r, sigma=sigma, T=T, f=g)
    delta = (f(S0+eps*S0) - f(S0)) / (eps*S0)
    gamma = (f(S0+eps*S0) - 2*f(S0) + f(S0-eps*S0)) / (eps*S0)**2

    def f(x): return BS_Price_Int(S0=S0, r=r, sigma=x, T=T, f=g)
    vega = (f(sigma+eps*sigma) - f(sigma)) / (eps*sigma)

    return [delta, vega, gamma]


# b)
def g(x): return max([x-110, 0])
params = dict(r=0.05, sigma=0.3, T=1, eps=0.001, g=g)
S0_list = range(60, 140+1)
delta_list = [BS_Greeks_num(**params, S0=x)[0] for x in S0_list]

f, ax = plt.subplots(1,1)
ax.plot(S0_list, delta_list, label='Delta')
ax.axvline(x=110, color='red', label='K')
ax.legend()
ax.set_title("Delta of a European Call Option")
ax.set_xlabel("$S(0)$")
plt.show()