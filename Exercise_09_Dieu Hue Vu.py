import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Part a

def get_CRR_params(r, sigma,T, M):
    dt = T/M
    beta = 1/2*(math.exp(-r*dt) + math.exp((r+sigma**2)*dt))
    u = beta + np.sqrt(beta**2 - 1)
    d = 1/u
    q = (math.exp(r*dt) - d)/(u-d)
    return u, d, q, dt

def CRR_stock(S_0, M,u,d):
    S = np.empty((M + 1, M + 1))  # stock price matrix
    for i in range(0, M + 1):
        for j in range(0, i + 1):
            S[j, i] = S_0 * (u ** j) * (d ** (i - j))

    return S

def CRR_AmEuPut(S_0,r,sigma,T,M,K,EU):
    assert min(S_0, r, sigma, T) > 0
    assert EU in [0,1]

    u, d, q, dt = get_CRR_params(r, sigma, T, M)
    S = CRR_stock(S_0, M, u, d)
    V = np.empty((M + 1, M + 1))  # price of option
    V[:, M] = np.maximum(K - S[:, M], 0)

    if EU==1:
        for i in range(M - 1, -1, -1):
            for j in range(i + 1):
                V[j, i] = math.exp(-r * dt) * (q * V[j + 1, i + 1] + (1 - q) * V[j, i + 1])
    else:
        for i in range(M - 1, -1, -1):
            for j in range(i + 1):
                V[j, i] = max(math.exp(-r * dt) * (q * V[j + 1, i + 1] + (1 - q) * V[j, i + 1]), K - S[j, i], 0)
    return V

# part b
def BlackScholes_EuPut(t,S_t,r,sigma,T,K):
    d1 = (math.log(S_t / K) + (r + 1 / 2 * (sigma ** 2)) * (T - t)) / (sigma * math.sqrt(T - t))
    d2 = d1 - sigma * math.sqrt(T - t)
    V_0 = K * math.exp(-r * (T - t)) * norm.cdf(-d2) - S_t * norm.cdf(-d1)
    return V_0


S_0,r,sigma,T,K,t,EU =100,0.05,0.3,1,120,0,1
M=range(10, 501, 1)
BS_price = np.empty(491, dtype=float)
CRR_price = np.empty(491, dtype=float)
for i in range(491):
    CRR_price[i] = CRR_AmEuPut(S_0, r, sigma, T, M[i], K, EU)[0,0]
    BS_price[i] = BlackScholes_EuPut(t,S_0,r,sigma,T,K)


plt.plot(M, CRR_price, 'b')
plt.plot(M,BS_price,'r')
plt.title('European put option prices')
plt.show()