import math
import numpy as np
from scipy.stats import norm

def BackScholes_EuCall(t,S_t,r,sigma,T,K):
    d1 = (np.log(S_t/K) + (r+1/2*(sigma**2))*(T-t))/(sigma*np.sqrt(T-t))
    d2 = d1 - sigma*math.sqrt(T-t)
    V_0 = S_t*norm.cdf(d1) - K*math.exp(-r*(T-t))*norm.cdf(d2)
    return V_0


def Eu_Option_BS_MC(S0,r,sigma,T,K,M,f):
    X = np.random.normal(0,1,size=M)
    S = np.empty(M)
    payoff = np.empty(M)
    for i in range(0,M,1):
        S[i] = S0*np.exp((r-np.square(sigma)/2)*T + sigma*np.sqrt(T)*X[i])
        payoff[i] = f(S[i],K)
    V_0 = np.exp(-r*T)*np.mean(payoff)
    return V_0

def f(x,K):
    return max(x-K,0)

S0 = 110
r = 0.03
sigma = 0.2
T = 1
M = 10000
K = 100
t = 0

print('Price of European call option:')
print('-using Monte Carlo method:',Eu_Option_BS_MC(S0,r,sigma,T,K,M,f))
print('-using BS formula:', BackScholes_EuCall(t,S0,r,sigma,T,K))