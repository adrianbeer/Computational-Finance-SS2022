import numpy as np
import matplotlib.pyplot as plt

import math

# GROUP QF 26
# NAMES: Adrian Beer, Dieu Hue Vu
# EXERCISE 39

"Valuation of an American Put using the explicit finite difference scheme"

# computes the price of american and european puts in the CRR model
def CRR_AmEuPut(S_0, r, sigma, T, M, K, EU):
    # compute values of u, d and q
    delta_t = T / M
    alpha = math.exp(r * delta_t)
    beta = 1 / 2 * (1 / alpha + alpha * math.exp(math.pow(sigma, 2) * delta_t))
    u = beta + math.sqrt(math.pow(beta, 2) - 1)
    d = 1 / u
    q = (math.exp(r * delta_t) - d) / (u - d)

    # allocate matrix S
    S = np.empty((M + 1, M + 1))

    # fill matrix S with stock prices
    for i in range(1, M + 2, 1):
        for j in range(1, i + 1, 1):
            S[j - 1, i - 1] = S_0 * math.pow(u, j - 1) * math.pow(d, i - j)

    # V will contain the put prices
    V = np.empty((M + 1, M + 1))
    # compute the prices of the put at time T
    V[:, M] = np.maximum(K - S[:, M], 0)

    # define recursion function for european and american options
    if (EU == 1):
        def g(k):
            # martingale property in the european case
            return math.exp(-r * delta_t) * (q * V[1:k + 1, k] + (1 - q) * V[0:k, k])
    else:
        def g(k):
            # snell envelope in the american case
            return np.maximum(K - S[0:k, k - 1], math.exp(-r * delta_t) * (q * V[1:k + 1, k] + (1 - q) * V[0:k, k]))

    # compute put prices at t_i
    for k in range(M, 0, -1):
        V[0:k, k - 1] = g(k)

    # return the price of the put at time t_0 = 0
    return V[0, 0]


def BS_AmPut_FiDi_Explicit(r,sigma,a,b,m,nu_max,T,K):
    q = 2 * r / sigma ** 2
    dx_tilde = (b - a) / m
    dt_tilde = sigma ** 2 * T / (2 * nu_max)
    lambda_ = dt_tilde / dx_tilde ** 2
    x_tilde = np.arange(a, b + dx_tilde, dx_tilde)
    t_tilde = np.arange(0, sigma ** 2 * T / 2 + dt_tilde, dt_tilde)


    g= np.transpose(np.asmatrix(np.maximum(np.exp(x_tilde*(q-1)/2)-np.exp(x_tilde*(q+1)/2),0)))* \
       np.asmatrix(np.exp((q + 1) ** 2 * t_tilde / 4))

    # r1, r2 for boundaries
    r1 = np.transpose(g[0,:])
    r2: int = 0
    # w at t_tilde=0
    w = g[:,0]

    for j in range(0, nu_max):
        w[0] = r1[j]
        w[-1] = r2
        b = np.empty([m + 1,1])
        for i in range(2, m - 1):
            b[i] = w[i] + lambda_ * (w[i + 1] - 2 * w[i] + w[i - 1])
        b[1] = w[1] + lambda_ * (w[2] - 2 * w[1] +g[0,j] )
        b[m - 1] = w[m - 1] + lambda_ * (g[m,j]- 2 * w[m - 1] + w[m - 2])
        w = np.maximum(b, g[:, j + 1])

    S = K * np.exp(x_tilde)
    V = np.empty(m+1)
    for i in range(0,m+1):
        V[i] = K * w[i] * np.exp(-x_tilde[i] / 2 * (q - 1) - 1 / 2 * sigma ** 2 * T * ((q - 1) ** 2 / 4 + q))

    return S,V


r = 0.05
sigma = 0.2
a = -0.7
b = 0.4
m = 100
nu_max = 2000
T = 1
K = 95

(S, v0_hat) = BS_AmPut_FiDi_Explicit(r, sigma, a, b, m, nu_max, T, K)

v0 = [CRR_AmEuPut(s, r, sigma, T, 500, K, EU=0) for s in S]

plt.plot(S, v0_hat, label="Finite Differences price")
plt.plot(S, v0, label="CRR price")
plt.legend()
plt.title("Price of an American Put option with strike K=95")
plt.show()
