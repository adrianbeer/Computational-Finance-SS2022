import math
import numpy as np
import scipy.stats as ss
import scipy.integrate as integrate
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import cmath

# computes the price of a put in the Black-Scholes model
def BlackScholes_EuPut(t, S_t, r, sigma, T, K):
    d_1 = (math.log(S_t / K) + (r + 1 / 2 * math.pow(sigma, 2)) * (T - t)) / (sigma * math.sqrt(T - t))
    d_2 = d_1 - sigma * math.sqrt(T - t)
    phi = ss.norm.cdf(-d_1)
    C = K * math.exp(-r * (T - t)) * ss.norm.cdf(-d_2) -S_t * phi
    return C

# part a)
def ImplVol(S0, r, T, K, V):
    def function(sigma):
        Squared_error = (V - BlackScholes_EuPut(0, S0, r, sigma, T, K))**2
        return Squared_error
    initial_guess = 1
    res = minimize(function, initial_guess, bounds=((0, None),), method='Powell')
    return res.x[0]

# Test it with a Blackscholes price:
S0 = 100
r = 0.05
T = 1
K = 100
#sigma = 0.4
V0 = 6.09

#BS_price = BlackScholes_EuPut(0, S0, r, sigma, T, K)

#print("The price of this option in the Blackscholes model with sigma " + str(sigma) + " is " + str(BS_price))
print("The implied volatility is " + str(ImplVol(S0, r, T, K, V0)))


# part b)
def Heston_EuPut_Laplace(S0, r, nu0, kappa, lmbda, sigma_tilde, T, K, R):
    # Laplace transform of the function f(x) = (e^(xp) - K)^+ (cf. (4.6))
    def f_tilde(z):
        if np.real(z) < 0:
            return np.power(K, 1 - z) / (z * (z - 1))
        else:
            print('Error')

    # Characteristic function of log(S(T)) in the Heston model (cf. (4.8))
    def chi(u):
        d = cmath.sqrt(
            math.pow(lmbda, 2) + math.pow(sigma_tilde, 2) * (complex(0, 1) * u + cmath.exp(2 * cmath.log(u))))
        n = cmath.cosh(d * T / 2) + lmbda * cmath.sinh(d * T / 2) / d
        z1 = math.exp(lmbda * T / 2)
        z2 = (complex(0, 1) * u + cmath.exp(2 * cmath.log(u))) * cmath.sinh(d * T / 2) / d
        v = cmath.exp(complex(0, 1) * u * (math.log(S0) + r * T)) * cmath.exp(
            2 * kappa / math.pow(sigma_tilde, 2) * cmath.log(z1 / n)) * cmath.exp(-nu0 * z2 / n)
        return v

    # integrand for the Laplace transform method (cf. (4.9))
    def integrand(u):
        return math.exp(-r * T) / math.pi * (f_tilde(R + complex(0, 1) * u) * chi(u - complex(0, 1) * R)).real

    # integration to obtain the option price (cf. (4.9))
    V0 = integrate.quad(integrand, 0, 50)
    return V0[0]

S0 = 100
r = 0.05
nu0 = 0.3**2
kappa = 0.3**2
lambd = 2.5
sigma_tilde = 0.2
T = 1
K = range(50, 151, 1)
R = -1
V0 = np.empty(101, dtype=float)

for i in range(0,len(K)):
    V0[i] = Heston_EuPut_Laplace(S0,r,nu0, kappa, lambd, sigma_tilde,T,K[i],R)
plt.clf()
plt.plot(K, V0)
plt.xlabel('K')
plt.ylabel('V0')

plt.show()

# part c)
def ImplVol_Heston(S0, r, nu0, kappa, lambd, sigma_tilde, T, K, R):
    V0 = np.empty(101, dtype=float)

    for i in range(0, len(K)):
        V0[i] = Heston_EuPut_Laplace(S0, r, nu0, kappa, lambd, sigma_tilde, T, K[i], R)

    implVol = np.empty(101, dtype=float)
    for i in range(0, len(K)):
        implVol[i] = ImplVol(S0, r, T, K[i], V0[i])

    return implVol

implVol = ImplVol_Heston(S0, r, nu0, kappa, lambd, sigma_tilde, T, K, R)
plt.clf()
plt.plot(K, implVol)
plt.xlabel('K')
plt.ylabel('implVol')

plt.show()
