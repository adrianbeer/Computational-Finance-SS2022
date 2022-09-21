# Sample Solution for C-Exercise 40, SS 2022
# Pricing an American option with the Longstaff-Schwartz method and Richardson extrapolation

import math
import numpy as np
import scipy.special as ss
import scipy.stats
import matplotlib.pyplot as plt


def BS_AmPut_LSM_Richardson(S0, r, sigma, T, K, m, N, l):
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

    # stepsize for coarse grid
    stepsize = range(2, 10, 1)
    # array for results:
    V0_normal = np.zeros(len(stepsize)+1)
    V0_Richardson = np.zeros(len(stepsize))
    # normal LSM
    Y_dt = np.exp(-r * tau) * g
    V0_normal[0] = max(np.maximum(K - S0, 0), np.mean(Y_dt))
    sigma_hat = np.std(Y_dt)
    epsilon = scipy.stats.norm.ppf(1 - (1 - 0.95) / 2, 0, 1) * sigma_hat / math.sqrt(N)

    for s in stepsize:
        # save stock price on coarse grid
        S_coarse = S[::s, :]
        # second t-loop for coarse grid
        tau_coarse = T * np.ones(N)
        g_coarse = np.maximum(K - S[-1, :], 0)
        # t-loop
        for i in range(len(S_coarse[:, 0]) - 1, -1, -1):
            # define subset I_i
            I = np.where(K - S_coarse[i, :] > 0)[0]
            if len(I) > 0:
                # compute x_n
                x = np.empty((len(I), M))
                for j in range(0, M):
                    x[:, j] = l[j](S_coarse[i, I])
                # compute y_n
                y = np.exp(-r * (tau_coarse[I] - s * i * Delta_t)) * g_coarse[I]
                # linear regression
                a = np.linalg.lstsq(x, y, rcond=None)[0]
                # update tau and g
                exercise_check = I[np.where(np.maximum(K - S_coarse[i, I], 0) >= (x * a).sum(axis=1))]
                tau_coarse[exercise_check] = s * i * Delta_t
                g_coarse[exercise_check] = np.maximum(K - S_coarse[i, exercise_check], 0)
        # compute estimates for Richardson extrapolation
        Y_idt = np.exp(-r * tau_coarse) * g_coarse
        Z = 2 * Y_dt - Y_idt

        # compute normal V0 and Richardson V0
        V0_normal[s-1] = max(np.maximum(K - S0, 0), np.mean(Y_idt))
        V0_Richardson[s-2] = max(np.maximum(K - S0, 0), np.mean(Z))
        sigma_hat_Richardson = np.std(Z)
        epsilon_Richardson = scipy.stats.norm.ppf(1 - (1 - 0.95) / 2, 0, 1) * sigma_hat_Richardson / math.sqrt(N)
    return V0_normal, epsilon, V0_Richardson, epsilon_Richardson


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
    N = 2000

    # construct basis of Laguerre-Polynomials
    M = 3


    def Laguerre_basis(k, K):
        def basis_func(x):
            return ss.eval_laguerre(k, x) * np.exp(-x / 5)

        return basis_func


    l = [Laguerre_basis(k, K) for k in range(M + 1)]

    V0, epsilon, V0_Richardson, epsilon_Richardson = BS_AmPut_LSM_Richardson(S0=S0, r=r, sigma=sigma, T=T, K=K, m=m, N=N, l=l)
    V0_CRR = CRR_AmEuPut(S_0=S0, r=r, sigma=sigma, T=T, M=500, K=K, EU=0)

    print(
        "The Monte-Carlo approximation with Longstaff-Schwartz to the price of the American Put option is given by " + str(
            V0) + ";\n   radius of 95% confidence interval: " + str(epsilon))
    print(
        "The Monte-Carlo approximation with Longstaff-Schwartz-Richardson to the price of the American Put option is given by " + str(
            V0_Richardson) + ";\n   radius of 95% confidence interval: " + str(epsilon_Richardson))
    print("The price approximated by the CRR model is: " + str(V0_CRR))



    V0_ref = CRR_AmEuPut(S_0=S0, r=r, sigma=sigma, T=T, M=1000, K=K, EU=0)
    num_simulations = 50
    MSE, MSE_Richardson = [], []
    V0s, V0s_Richardson, V0_test = [], [], []
    for j in range(num_simulations):
        V0, error, V0_trunc, error_trunc = BS_AmPut_LSM_Richardson(S0=S0, r=r, sigma=sigma, T=T, K=K, m=m, N=N,
                                                                          l=l)
        V0s.append(V0[0])
        V0s_Richardson.append(V0_trunc[0])
        plt.plot(V0)
        MSE.append((V0_ref - V0[0]) ** 2)
        MSE_Richardson.append((V0_ref - V0_trunc[0]) ** 2)
    improvement = (np.mean(MSE) - np.mean(MSE_Richardson)) / np.mean(MSE)

    plt.figure(figsize=(12, 8))
    plt.scatter(np.repeat(1, num_simulations), V0s, s=3)
    plt.scatter(np.repeat(2, num_simulations), V0s_Richardson, s=3)
    plt.hlines(y=V0_ref, xmin=0.5, xmax=2.5)
    plt.xlim(0.5, 2.5)
    plt.grid(alpha=0.3)
    plt.xticks([])
    plt.ylabel('Option Prices')
    plt.title(
        'Repeated Monte Carlo Simulation with and without Richardson Extrapolation. MSE Improvement: {:.2f}%'.format(
            improvement * 100))
    plt.legend(['Estimates without Extrapolation. Bias: {:.3f}'.format(np.mean(V0s) - V0_ref),
                'Estimates with Extrapolation. Bias: {:.3f}'.format(np.mean(V0s_Richardson) - V0_ref),
                'Reference Price'])



    # test wether linearity in m is given:
    V0_test = []
    test_range = range(100, 10000, 500)
    for i in test_range:
        V0 = BS_AmPut_LSM_Richardson(S0=S0, r=r, sigma=sigma, T=T, K=K, m=i, N=N, l=l)[0]
        V0_test.append(V0[0])
    plt.figure()
    plt.plot(test_range, V0_test)
    plt.show()