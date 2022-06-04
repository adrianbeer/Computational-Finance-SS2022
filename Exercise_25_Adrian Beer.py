# GROUP QF 26
# NAMES: Adrian Beer, Dieu Hue Vu
# EXERCISE 23

import numpy as np
import  matplotlib.pyplot as plt


def f(x):
    # Wigner semicircle distribution with R=1
    if np.abs(x) <= 1:
        return 2*np.sqrt(1-x**2)/np.pi
    else:
        return 0


# The max of the semi-cirlce density f(x) is f(0) = 1/(2*pi).
# We can bound the semi-circle density by  1/2 * I{x in [-1,1]} = g(x) ~ Y ~ U(-1,1).
# Y = 2*(U-1/2) = 2*U - 1 where U ~ U(0,1).
# f(x) <= 2/pi <= 1/2 * 4/pi = g(x) * C.
# Using Acceptance/Rejection method: Accept simulation of Y, i.e. y, U <= if f(Y) / (C*g(Y)).


def Sample_dist_AR(N):
    C = 4/np.pi
    samples = []
    while len(samples) < N:
        U = np.random.uniform(0, 1, 1)[0]
        Y = np.random.uniform(0, 1, 1)[0]*2 - 1
        if U <= f(Y)/(C*1/2): # g(Y) = 1/2 is a constant...
            samples.append(Y)
    return samples


sim_samples = Sample_dist_AR(10000)
fig, ax = plt.subplots(1,1)
print(sim_samples)
ax.hist(x=sim_samples, bins=15, density=True, histtype='step')

x_grid = np.linspace(-1, 1, 100)
ax.plot(x_grid, [f(x) for x in x_grid])

plt.show()
# Simulated values seem to follow the semicircle distribution...
