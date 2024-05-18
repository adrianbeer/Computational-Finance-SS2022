import math


def bond_value(V0, r, n, M, c):
    assert r > 0
    assert type(n) == int
    assert c in [0, 1]

    if c==1: # Continuous compounding
        return V0*math.exp(r*n)
    elif c==0: # Discrete compounding
        return V0*(1+r/M)**(n*M)
    else:
        raise Exception("c not in {0, 1]")


V0, r, n, M, c = 1000, 0.05, 10, 4, 0
print(bond_value(V0, r, n, M, c))
