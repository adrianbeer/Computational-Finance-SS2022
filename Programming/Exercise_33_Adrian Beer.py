import math
import numpy as np
import scipy
from sklearn.linear_model import LinearRegression
from fractions import Fraction
import matplotlib.pyplot as plt

# Solution from Exercise 09
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


def generate_geom_B_path(S0, r, sigma, T, m):
    m=m+1
    Y = np.ndarray(shape=(m,))
    Y[0] = S0
    dt = T/m
    dW = np.random.normal(0, dt**0.5, m)
    for i in range(1,m):
        # Then calculate stock realization
        Y[i] = Y[i-1] + (Y[i-1]*r)*dt + (Y[i-1]*sigma)*dW[i]
    return Y


def BS_AmPut_LSM(S0, r, sigma, T, K, m, N, l):
    dt = T/m

    # Simulate N paths of geometric Brownian motion:
    paths = np.array([generate_geom_B_path(S0, r, sigma, T, m) for i in range(N)])

    # Let Atau = tau/dt and tau = Atau*dt
    Atau = np.ones(shape=(N,), dtype=int)*m
    g = np.maximum(K - paths, 0)

    for i in reversed(range(m)):
        y = np.exp(-r * (Atau*dt - i*dt)) * g[range(N), Atau]
        x = paths[:, i]

        # Compute coefficients a_hat by linearly regressing y on the laguerre polynomials
        X = np.array([scipy.special.laguerre(k)(x) for k in range(l)]).transpose()
        a_hats = LinearRegression(fit_intercept=False).fit(y.reshape(-1, 1), X).coef_
        # Update tau only if current payoff g is larger than expected payoff in the future
        # and the option is in the money (x <= K).
        mask = (g[:, i] >= np.dot(X, a_hats).flatten()) * (x <= K)
        Atau[mask] = i

    #debugging
    # plt.plot(g[:50, :].transpose())
    # plt.scatter(Atau[0:50], g[range(50), Atau[0:50]], color='black')
    # print(g[range(N), Atau])
    # plt.show()
    #

    immediate_payoff0 = max([K-S0, 0])
    expected_option_value = 1/N*sum(np.exp(-r * (Atau*dt)) * g[range(N), Atau])
    V0 = max([immediate_payoff0, expected_option_value])
    return V0


S0 = 100
r = 0.05
sigma = 0.2
T = 1
K = 90
m = 1000
N = 2000
K = 90
l = 5

s = BS_AmPut_LSM(S0, r, sigma, T, K, m, N, l)
print(f"Approximated option value : {s}")

s = CRR_AmEuPut(S0, r, sigma, T, m, K, EU=0)
print(f"Theoretically correct option value: {s}")