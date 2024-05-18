import math
import numpy as np
import scipy.integrate as integrate
from matplotlib import pyplot as plt
import scipy
import cmath

def BS_EuCall_Laplace(S0, r, sigma, T, K, R):
    def chi(u): return cmath.exp(complex(0, u)*(np.log(S0)+r*T)-complex(u**2, u)*sigma**2*T/2)
    def f_tilde(z): return K**(1-z)/(z*(z-1))  # Is complex exponent handled correctly?
    def integrand(u): return np.real(f_tilde(complex(R, u))*chi(complex(u, -R)))
    V0, error = scipy.integrate.quad(func=integrand, a=0, b=np.inf)
    return V0


params = dict(r=0.03, sigma=0.2, T=1, K=110, R=10)  # Choose some R > 1
S0_list = range(50, 150+1)
call_prices = [BS_EuCall_Laplace(**params, S0=x) for x in S0_list]

f, ax = plt.subplots(1,1)
ax.plot(S0_list, call_prices, label='Call_prices')
ax.axvline(x=params["K"], color='red', label='K')
ax.legend()
ax.set_title("Call Prices of European Call Option via Laplace")
ax.set_ylabel("$V(0)$")
ax.set_xlabel("$S(0)$")
ax.grid()
plt.show()
