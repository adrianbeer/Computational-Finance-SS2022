import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def get_CRR_params(r, sigma,T, M):
    dt = T/M
    beta = 1/2*(math.exp(-r*dt) + math.exp((r+sigma**2)*dt))
    u = beta + np.sqrt(beta**2 - 1)
    d = 1/u
    q = (math.exp(r*dt) - d)/(u-d)
    return u, d, q, dt


def CRR_stock(S_0, r, sigma,T, M):
    assert min(S_0, r, sigma, T) > 0
    assert type(M) == int

    u, d, q, dt = get_CRR_params(r, sigma,T, M)
    spm = np.zeros(shape=(M+1, M+1), dtype=float)  # Stock Price Matrix

    first_row = np.power(d, np.arange(0, M+1, 1))
    spm[0, :] = first_row
    for j in range(1, M + 1):
        spm[j, :] = np.roll(first_row, j) * u**j
    return np.triu(spm)*S_0


def CRR_EuCall(S_0, r, sigma, T: int, M: int, K: int):
    u, d, q, dt = get_CRR_params(r, sigma,T, M)
    S: np.ndarray = CRR_stock(S_0, r, sigma,T, M)  # stock price matrix
    V: np.ndarray = np.empty(shape=(M+1, M+1), dtype=float)  # option price matrix
    V[:, -1] = np.maximum(0, S[:, -1] - K)

    for i in reversed(range(1, M+1)):
        for j in range(i):
            V[j, i-1] = math.exp(-r*dt)*(q*V[j+1, i] + (1-q)*V[j, i])
    return V[0,0]


# b)
def BlackScholes_EUCall(t, S_t, r, sigma, T, K):
    d1 = (np.log(S_t/K) + (r + sigma**2/2)*(T-t))/(sigma*(np.sqrt(T-t)))
    d2 = d1 - sigma*(np.sqrt(T-t))
    C = S_t*(norm.cdf(d1)) - K*np.exp(-r*(T-t))*norm.cdf(d2)
    return C


# c)
S_0 = S_t = 100
r = 0.03
sigma = 0.3
t=0
T = 1
M = 100
K_list = np.arange(70,201,1)

V_0_BS = BlackScholes_EUCall(t, S_t, r, sigma, T, K_list)
V_0_CRR = np.array([CRR_EuCall(S_0, r, sigma, T, M, K) for K in K_list])
error = V_0_CRR - V_0_BS
f, (ax1, ax2) = plt.subplots(2,1, sharex=True)
ax1.plot(K_list, error)
ax1.set_ylabel("abs. error")
ax1.axhline(0, color='r')

ax2.set_xlabel("K")
ax2.plot(K_list, error/V_0_BS)
ax2.set_ylabel("rel. error")
ax2.axhline(0, color='r')
plt.show()
