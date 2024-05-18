import numpy as np
import math


def CRR_stock(S_0, r, sigma,T, M):
    assert min(S_0, r, sigma, T) > 0
    assert type(M) == int

    dt = T/M
    beta = 1/2*(math.exp(-r*dt) + math.exp((r+sigma**2)*dt))
    u = beta + np.sqrt(beta**2 - 1)
    d = 1/u
    spm = np.zeros(shape=(M+1, M+1), dtype=float)  # Stock Price Matrix

    first_row = np.power(d, np.arange(0, M+1, 1))
    spm[0, :] = first_row
    for j in range(1, M + 1):
        spm[j, :] = np.roll(first_row, j) * u**j
    return np.triu(spm)*S_0


S_0, r, sigma, T, M = 100, 0.05, 0.3, 1, 500
S = CRR_stock(S_0, r, sigma,T, M)

np.set_printoptions(suppress=True)
print(np.round(S, 3))