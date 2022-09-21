# C-Exercise 25, SS 2022

import math
import numpy as np
import matplotlib.pyplot as plt


# pdf of truncated exponential distribution
def density(x):
    if -1 <= x and x <= 1:
        return 2*math.sqrt(1-x**2)/math.pi
    else:
        return 0


def Sample_dist_AR(N):
    # compute C as the maximum of f(x)/g(x), such that f(x) <= C*g(x)
    C = density(0) * 2

    # function to generate a single sample of the distribution
    def SingleSample():
        # set run parameter for while-loop
        success = False
        while ~success:
            # generate two U([0,1]) random variables
            U = np.random.uniform(size=(2, 1))
            # scale one of them to the correct interval
            Y = U[0] * 2 - 1
            # check for rejection/acceptance
            # when the sample gets rejected the while loop will generate a new sample and will check again
            success = ((C * U[1] / 2) <= density(Y))
        return Y

    # use function SingleSample N times to generate N samples
    X = np.empty(N, dtype=float)
    for i in range(0, N):
        X[i] = SingleSample()

    return X


# test parameters
N = 10000

X = Sample_dist_AR(N)
# plot histogram
plt.hist(X, 50, density=True)

# plot exact pdf
x = np.linspace(-1 - 1 / 20, 1 + 1 / 20, 1000)
pdf_vector = np.zeros(len(x))
for i in range(0, len(x)):
    pdf_vector[i] = density(x[i])
plt.plot(x, pdf_vector)
plt.show()
