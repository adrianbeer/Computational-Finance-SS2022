import math
import numpy as np


def get_CRR_params(r, sigma,T, M):
    dt = T/M
    beta = 1/2*(math.exp(-r*dt) + math.exp((r+sigma**2)*dt))
    u = beta + np.sqrt(beta**2 - 1)
    d = 1/u
    q = (math.exp(r*dt) - d)/(u-d)
    return u, d, q, dt

print(get_CRR_params(0.05, 0.3, 1, 500))

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


S_0, r, sigma, T, M = 100, 0.05, 0.3, 1, 500
S = CRR_stock(S_0, r, sigma,T, M)
#print(S)


def CRR_option_price(S_0, r, sigma, T: int, M: int, european: bool, K: int, call: bool):
    u, d, q, dt = get_CRR_params(r, sigma,T, M)
    S: np.ndarray = CRR_stock(S_0, r, sigma,T, M)  # stock price matrix
    V: np.ndarray = np.empty(shape=(M+1, M+1), dtype=float)  # option price matrix
    V[:, -1] = np.maximum(0, S[:, -1] - K) if (call is True) else np.maximum(0, K - S[:, -1])

    if european:
        for i in reversed(range(1, M+1)):
            for j in range(i):
                # TODO: This can be vectorized
                V[j, i-1] = math.exp(-r*dt)*(q*V[j+1, i] + (1-q)*V[j, i])
        return V


V = CRR_option_price(100, 0.05, 0.3, 1, 1000, True, 100, True)
print(V[0,0])