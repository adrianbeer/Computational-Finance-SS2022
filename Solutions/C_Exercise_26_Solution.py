import math
import numpy as np
import scipy.stats

def Eu_Option_BS_MC(S0, r, sigma, T, K, M, f):
    # generate M samples
    X = np.random.normal(0, 1, M)

    # compute ST and Y for each sample
    ST = S0 * np.exp((r - 0.5 * math.pow(sigma, 2)) * T + sigma * math.sqrt(T) * X)
    Y = f(ST, K)

    # calculate V0
    VN_hat = np.mean(Y)
    V0 = math.exp(-r * T) * VN_hat

    # compute confidence interval
    epsilon = 1.96 * math.sqrt(np.var(Y) / M)
    c1 = (VN_hat - epsilon) * math.exp(-r * T)
    c2 = (VN_hat + epsilon) * math.exp(-r * T)
    return V0, c1, c2, epsilon

def BS_EuOption_MC_AV(S0, r, sigma, T, K, M, f):
    # generate M samples
    X = np.random.normal(0, 1, M)

    # compute ST and Y for each sample
    ST = S0 * np.exp((r - 0.5 * math.pow(sigma, 2)) * T + sigma * math.sqrt(T) * X)
    ST_neg = S0 * np.exp((r - 0.5 * math.pow(sigma, 2)) * T + sigma * math.sqrt(T) * -X)
    Y = (f(ST, K) + f(ST_neg, K))/2

    # calculate V0
    VN_hat = np.mean(Y)
    V0 = math.exp(-r * T) * VN_hat

    # compute confidence interval
    epsilon = 1.96 * math.sqrt(np.var(Y) / M)
    c1 = (VN_hat - epsilon) * math.exp(-r * T)
    c2 = (VN_hat + epsilon) * math.exp(-r * T)
    return V0, epsilon

# computes the price of a call in the Black-Scholes model
def EuCall_BlackScholes(t, S_t, r, sigma, T, K):
    d_1 = (math.log(S_t / K) + (r + 1 / 2 * math.pow(sigma, 2)) * (T - t)) / (sigma * math.sqrt(T - t))
    d_2 = d_1 - sigma * math.sqrt(T - t)
    phi = scipy.stats.norm.cdf(d_1)
    C = S_t * phi - K * math.exp(-r * (T - t)) * scipy.stats.norm.cdf(d_2)
    return C, phi

# test parameters
S0 = 110
r = 0.03
sigma = 0.2
T = 1
M = 100000
K = 100


# european call
def g(x, K):
    return np.maximum(x - K, 0)

V0 = EuCall_BlackScholes(0, S0, r, sigma, T, K)[0]
print('The exact price of an European Call in the Black Scholes model is ' + str(V0))

V0_AV, epsilon = BS_EuOption_MC_AV(S0, r, sigma, T, K ,M, g)
print('Price of European call by use of Monte-Carlo simulation with control variate: ' + str(V0) + ', radius of 95% confidence interval: '+ str(epsilon))

V0_plain, c1, c2, epsilon = Eu_Option_BS_MC(S0, r, sigma, T, K, M, g)
print('Price of European call by use of plain Monte-Carlo simulation: ' + str(V0) + ', radius of 95% confidence interval: ' + str(epsilon))