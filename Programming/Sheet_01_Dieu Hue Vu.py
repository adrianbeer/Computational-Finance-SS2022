import math
import numpy as np

def bond_value(V0,r,n,M,c):
    if r>0 and c==1:
        Vn=V0*math.exp(r*n)
    elif r>0 and c==0:
        Vn=V0*((1+r/M)**(n*M))
    else:
        print('Please check the inputs')
    return Vn
print(str(bond_value(1000,0.05,10,4,0)))

def CRR_stock(S_0,r,sigma,T,M):
    S=np.empty((M+1,M+1))
    S[0,0]=S_0
    beta = (math.exp(-r * T / M) + math.exp((r + sigma ** 2) * T / M))/2
    u = beta + math.sqrt(beta ** 2 - 1)
    d = 1 / u
    if S_0<0 or r<0 or sigma<0 or T<0:
        print('Please check the inputs')
    else:
        for i in range(1,M+1):
            for j in range(0,i+1):
                S[j,i]=S_0*(u**j)*(d**(i-j))
    return S
print(np.round(CRR_stock(100,0.05,0.3,1,500),0))



