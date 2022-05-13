import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# GROUP 26
# NAMES: Adrian Beer, Dieu Hue Vu
# EXERCISE 9


# auxiliary function
def get_CRR_params(r, sigma,T, M):
    dt = T/M
    beta = 1/2*(math.exp(-r*dt) + math.exp((r+sigma**2)*dt))
    u = beta + np.sqrt(beta**2 - 1)
    d = 1/u
    q = (math.exp(r*dt) - d)/(u-d)
    return u, d, q, dt


# auxiliary function
def CRR_stock(S_0, r, sigma,T, M):
    assert min(S_0, r, sigma, T) > 0

    u, d, q, dt = get_CRR_params(r, sigma,T, M)
    spm = np.zeros(shape=(M+1, M+1), dtype=float)  # Stock Price Matrix

    first_row = np.power(d, np.arange(0, M+1, 1))
    spm[0, :] = first_row
    for j in range(1, M + 1):
        spm[j, :] = np.roll(first_row, j) * u**j
    return np.triu(spm)*S_0


# a)
def CRR_AmEuPut(S_0, r, sigma, T: int, M: int, K: int, EU: bool):
    u, d, q, dt = get_CRR_params(r, sigma,T, M)
    S: np.ndarray = CRR_stock(S_0, r, sigma,T, M)  # stock price matrix
    V: np.ndarray = np.empty(shape=(M+1, M+1), dtype=float)  # option price matrix
    V[:, -1] = np.maximum(0, K - S[:, -1])

    for i in reversed(range(1, M+1)):
        for j in range(i):
            expected_future_value = math.exp(-r * dt) * (q * V[j + 1, i] + (1 - q) * V[j, i])
            if EU:
                V[j, i-1] = expected_future_value
            else:
                immediate_payoff = max([0, K - S(j, i-1)])
                V[j, i-1] = max([immediate_payoff, expected_future_value])
    return V[0,0]


# b)
def BlackScholes_EUCall(t, S_t, r, sigma, T, K):
    d1 = (np.log(S_t/K) + (r + sigma**2/2)*(T-t))/(sigma*(np.sqrt(T-t)))
    d2 = d1 - sigma*(np.sqrt(T-t))
    C = S_t*(norm.cdf(d1)) - K*np.exp(-r*(T-t))*norm.cdf(d2)
    return C


def BlackScholes_EuPut (t, S_t, r, sigma, T, K):
    # Using put-call-parity:
    # Call_T - Put_T = S_T - K => Put_t = Call_t - (S^ - K*S_t/S_T)
    return BlackScholes_EUCall(t, S_t, r, sigma, T, K) - S_t + K*np.exp(-r*(T-t))


# c)
S_0 = S_t = 100
r = 0.05
sigma = 0.3
T = 1
M_list = np.arange(10,501,1)
K = 120

V_0_BS = np.array([BlackScholes_EuPut(t=0, S_t=S_0, r=r, sigma=sigma, T=T, K=K) for M in M_list])
V_0_CRR = np.array([CRR_AmEuPut(S_0, r, sigma, T, M, K, 1) for M in M_list])

f, ax1 = plt.subplots(1,1, sharex=True)
ax1.plot(M_list, V_0_CRR, label="CRR_price")
ax1.plot(M_list, V_0_BS, label="BS_Price")
ax1.set_ylabel("price")
ax1.set_title("European Put Price")
f.show()
plt.show()

print(CRR_AmEuPut(S_0, r, sigma, T, 500, K, 1))
