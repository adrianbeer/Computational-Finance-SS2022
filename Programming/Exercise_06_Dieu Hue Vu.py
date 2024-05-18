import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, norm

# Task a: CRR model
def CRR_EuCall(S_0,r,sigma,T,M,K):
    assert min(S_0, r, sigma, T,K) > 0
    assert type(M) == int

# create empty matrix and arrays for stock prices, probabilities and payoffs at M
    S= np.empty((M+1,M+1))   # stock price matrix
    Q_M= np.empty(M+1)       # probability of each branch at M
    payoff = np.empty(M+1)   # payoff at M

    dt = T/M
    beta = 1/2*(math.exp(-r * dt) + math.exp((r + sigma ** 2) * dt))
    u = beta + math.sqrt(beta ** 2 - 1)
    d = 1 / u
    q = (math.exp(r*dt)-d)/(u-d)

    for i in range(1,M+1):
        for j in range(0,i+1):
            S[j,i]=S_0*(u**j)*(d**(i-j))

    for j in range(0,M+1):
        payoff[j]=max(0,S[j,M]-K)
        Q_M[j]=binom.pmf(j,M,q)
    return np.dot(payoff,Q_M)


S_0, r, sigma, T, M,K = 1,0.05,math.sqrt(0.3),3,3,1.2
print(CRR_EuCall(S_0,r,sigma,T,M,K))

# Task b: Black- Scholes Model

def BackScholes_EuCall(t,S_t,r,sigma,T,K):
    d1=(math.log(S_t/K)+(r+1/2*(sigma**2))*(T-t))/(sigma*math.sqrt(T-t))
    d2=d1-sigma*math.sqrt(T-t)
    V_0=S_t*norm.cdf(d1)-K*math.exp(-r*(T-t))*norm.cdf(d2)
    return V_0

S_0, r, sigma, T, M,t = 100,0.03,0.3,1,100,0
diff=np.empty(131)
K=range(70,201,1)
for i in range(131):
    diff[i]=CRR_EuCall(S_0,r,sigma,T,M,K[i])-BackScholes_EuCall(t,S_0,r,sigma,T,K[i])
plt.plot(K,diff)
plt.xlabel('K')
plt.ylabel('Error of CRR model against BS price')
plt.show()

