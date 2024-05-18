import math
import numpy as np
from scipy.stats import norm

def BackScholes_EuCall(t,S_t,r,sigma,T,K):

    d1 = (np.log(S_t/K) + (r+1/2*(sigma**2))*(T-t))/(sigma*np.sqrt(T-t))
    d2 = d1 - sigma*math.sqrt(T-t)
    V_0 = S_t*norm.cdf(d1) - K*math.exp(-r*(T-t))*norm.cdf(d2)
    return V_0

def BS_EuOption_MC_CV (S0, r, sigma, T, K, M):
    X = np.random.normal(0, 1, size=M)
    S = np.empty(M) # stock prices at T
    Y = np.empty(M) #discounted value of European Call payoff
    f_x= np.empty(M) #discounted value of European self- quanto payoff

    for i in range(0, M, 1):
        S[i] = S0 * np.exp((r - np.square(sigma) / 2) * T + sigma * np.sqrt(T) * X[i])
        Y[i] = np.exp(-r * T)*max(S[i] - K, 0)
        f_x[i] = S[i]*Y[i]

    cov = np.cov(f_x,Y)[0,1]
    var_y = np.var(Y)
    beta = cov/var_y

    E_Y = BackScholes_EuCall(t,S0,r,sigma,T,K) #exact value from BS model
    V_CV = beta*E_Y+np.mean(f_x-beta*Y)         # value of option based on Control variates method
    V_pMC = np.mean(f_x)                        # value of option based on plain MC simulation
    var_pMC= np.var(f_x)/M
    var_CV = var_pMC - cov**2/(var_y*M)
    print(var_CV)
    print(np.var(f_x- beta*Y)/M)

    return V_CV,np.sqrt(var_CV),V_pMC,np.sqrt(var_pMC)


t=0
S0 = 110
r = 0.03
sigma = 0.2
T = 1
K = 100
M = 100000

V_CV,sigma_CV,V_pMC,sigma_pMC=BS_EuOption_MC_CV(S0, r, sigma, T, K, M)
print('Results from Control variates method:')
print('V=',V_CV, 'CI=',[V_CV-1.96*sigma_CV,V_CV+1.96*sigma_CV])
print(' ')
print('Results from plain MC:')
print('V=',V_pMC, 'CI=',[V_pMC-1.96*sigma_pMC,V_pMC+1.96*sigma_pMC])