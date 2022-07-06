import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# GROUP QF 26
# NAMES: Adrian Beer, Dieu Hue Vu
# EXERCISE 37

"Valuation of a European Call using the Crank-Nicolsen finite difference scheme"

def BlackScholes_EUCall(S0, r, sigma, T, K):
    t = 0
    d1 = (np.log(S0/K) + (r + sigma**2/2)*(T-t))/(sigma*(np.sqrt(T-t)))
    d2 = d1 - sigma*(np.sqrt(T-t))
    C = S0*(norm.cdf(d1)) - K*np.exp(-r*(T-t))*norm.cdf(d2)
    return C

def BS_EuCall_FiDi_CN(r, sigma, a, b, m, nu_max, T, K):
    q = 2*r/sigma**2
    dt_tild  = sigma**2*T/(2*nu_max)
    dx_tild = (b-a)/m
    x_tild = a + np.arange(m+1)*dx_tild
    lambda_ = dt_tild/dx_tild**2
    w = np.maximum(0, np.exp(x_tild/2*(q+1))-np.exp(x_tild/2*(q-1)))

    A = -(lambda_/2)*(np.tri(m + 1, m + 1, 1) + np.tri(m + 1, m + 1, 1).transpose() - 1) + (1 + 3/2 * (lambda_)) * np.identity(m + 1)
    B = -(-lambda_/2)*(np.tri(m + 1, m + 1, 1) + np.tri(m + 1, m + 1, 1).transpose() - 1) + (1 + 3/2 * (-lambda_)) * np.identity(m + 1)
    A_inv = np.linalg.inv(A)

    def r_2(nu): return np.exp(1 / 2 * (q + 1) * b + 1 / 4 * (q + 1) ** 2 * dt_tild * nu) - np.exp(
        1 / 2 * (q - 1) * b + 1 / 4 * (q - 1) ** 2 * dt_tild * nu)

    d = np.ndarray((m + 1, nu_max))
    d[-1, :] = [lambda_ / 2 * (r_2(nu) + r_2(nu + 1)) for nu in range(nu_max)]

    for i in range(nu_max):
        w[-1] = r_2(i)
        w = np.dot(A_inv, np.dot(B, w))
    w[-1] = r_2(i)

    S = K*np.exp(x_tild)
    v0 = K*w*np.exp(-x_tild/2*(q-1) - 1/2*sigma**2*T*((q-1)**2/4+q))
    return (S, v0)


r = 0.05
sigma = 0.2
a = -0.7
b = 0.4
m = 100
nu_max = 2000
T = 1
K = 100

(S, v0_hat) = BS_EuCall_FiDi_CN(r, sigma, a, b, m, nu_max, T, K)

v0 = BlackScholes_EUCall(S, r, sigma, T, K)

plt.plot(S, v0_hat, label="Approx price")
plt.plot(S, v0, label="BS price")
plt.legend()
plt.title("Crank-Nicolson Approximation")
plt.show()
