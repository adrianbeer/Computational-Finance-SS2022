
import numpy as np
from scipy import linalg
from scipy.stats import norm
import matplotlib.pyplot as plt

def BS_EuCall_FiDi_CN(r,sigma,a,b,m,nu_max,T,K):
    q = 2 * r / sigma ** 2
    dx_tilde = (b - a) / m
    dt_tilde = sigma ** 2 * T / (2 * nu_max)
    lambda_ = dt_tilde / dx_tilde ** 2
    x_tilde = np.arange(a, b + dx_tilde, dx_tilde)
    t_tilde = np.arange(0, sigma ** 2 * T / 2 + dt_tilde, dt_tilde)

    # w at t_tilde=0
    w = np.maximum(np.exp(x_tilde*(q+1)/2)-np.exp(x_tilde*(q-1)/2),0)


    # create matrices A and B.Then invert A

    A = np.empty([m+1,m+1])
    B = np.empty([m+1,m+1])
    A[0, 0] = 1 + lambda_
    B[0, 0] = 1 - lambda_
    for i in range(1,m+1):
        A[i, i] = 1 + lambda_
        A[i,i-1] = -lambda_/2
        A[i-1,i] = -lambda_/2
        B[i, i] = 1 - lambda_
        B[i, i-1] = lambda_ / 2
        B[i-1, i] = lambda_ / 2
    A_inv = linalg.inv(A)

    # r1, r2 for boundaries
    r1 = 0
    r2 = np.exp(1/2*(q+1)*b+1/4*(q+1)**2*t_tilde) - np.exp(1/2*(q-1)*b+1/4*(q-1)**2*t_tilde)

    for i in range(0,nu_max):
        w[0]= r1
        w[-1]= r2[i]
        w = np.dot(A_inv,np.dot(B,w))

    w[-1]=r2[-1]

    S = K * np.exp(x_tilde)
    V = K * w * np.exp(-x_tilde/2*(q-1)-1/2*sigma**2*T*((q-1)**2/4+q))

    return S,V
def BlackScholes_EUCall(t, S_t, r, sigma, T, K):
    d1 = (np.log(S_t/K) + (r + sigma**2/2)*(T-t))/(sigma*(np.sqrt(T-t)))
    d2 = d1 - sigma*(np.sqrt(T-t))
    C = S_t*(norm.cdf(d1)) - K*np.exp(-r*(T-t))*norm.cdf(d2)
    return C


r = 0.05
sigma= 0.2
a = -0.7
b = 0.4
m = 100
nu_max = 2000
T = 1
K = 100

S,V= BS_EuCall_FiDi_CN(r,sigma,a,b,m,nu_max,T,K)

V2=np.empty(m+1)
for i in range(0,m+1):
    V2[i]= BlackScholes_EUCall(0, S[i], r, sigma, T, K)


plt.plot(S,V,'b',label=" Crank-Nicolson scheme")
plt.plot(S,V2,'r',label="BS formula")
plt.xlabel("initial stock price")
plt.ylabel("option value at 0")
plt.legend()
plt.show()