# Sample Solution for C-Exercise 33, SS 2022
# Pricing an American option with the Longstaff-Schwartz method

import math
import numpy as np
import scipy.special as ss
import scipy.stats


def BS_AmPut_LSM(S0, r, sigma, T, K, m, N, l):
    Delta_t = T / m
    M = len(l)
    # Simulate N paths of geometric Brownian motion (using the exact method), saving all intermediate results
    Delta_W = np.random.normal(0, math.sqrt(Delta_t), (m, N))
    S = np.empty((m + 1, N))
    S[0, :] = S0 * np.ones(N)
    for i in range(0, m):
        S[i + 1, :] = S[i, :] * np.exp((r - math.pow(sigma, 2) / 2) * Delta_t + sigma * Delta_W[i, :])
    # set tau and g
    tau = T * np.ones(N)
    g = np.maximum(K - S[-1, :], 0)
    # t-loop
    for i in range(m - 1, -1, -1):
        # define subset I_i
        I = np.where(K - S[i, :] > 0)[0]
        if len(I) > 0:
            # compute x_n
            x = np.empty((len(I), M))
            for j in range(0, M):
                x[:, j] = l[j](S[i, I])
            # compute y_n
            y = np.exp(-r * (tau[I] - i * Delta_t)) * g[I]
            # linear regression
            a = np.linalg.lstsq(x, y, rcond=None)[0]
            # update tau and g
            exercise_check = I[np.where(np.maximum(K - S[i, I], 0) >= (x * a).sum(axis=1))]
            tau[exercise_check] = i * Delta_t
            g[exercise_check] = np.maximum(K - S[i, exercise_check], 0)
    # compute V0 and error
    samplesPayoff = np.exp(-r * tau) * g
    V0 = max(np.maximum(K - S0, 0), np.mean(samplesPayoff))
    sigma_hat = np.std(samplesPayoff)
    epsilon = scipy.stats.norm.ppf(1 - (1 - 0.95) / 2, 0, 1) * sigma_hat / math.sqrt(N)
    return V0, epsilon


# from C-Exercise 09
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


if __name__ == '__main__':
    # test parameters
    S0 = 100
    r = 0.05
    sigma = 0.2
    T = 1
    K = 90
    m = 1000
    N = 10000

    # construct basis of Laguerre-Polynomials
    M = 3


    def Laguerre_basis(k, K):
        def basis_func(x):
            return ss.eval_laguerre(k, x) * np.exp(-x / 5)

        return basis_func


    l = [Laguerre_basis(k, K) for k in range(M + 1)]

    V0, epsilon = BS_AmPut_LSM(S0=S0, r=r, sigma=sigma, T=T, K=K, m=m, N=N, l=l)
    V0_CRR = CRR_AmEuPut(S_0=S0, r=r, sigma=sigma, T=T, M=500, K=K, EU=0)

    print(
        "The Monte-Carlo approximation with Longstaff-Schwartz to the price of the American Put option is given by " + str(
            V0) + ";\n   radius of 95% confidence interval: " + str(epsilon))
    print("The price approximated by the CRR model is: " + str(V0_CRR))
