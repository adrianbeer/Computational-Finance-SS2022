import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib import cm
import functools

"Valuation of a European Call using the explicit finite difference scheme"

def BlackScholes_EUCall(S0, r, sigma, T, K):
    t = 0
    d1 = (np.log(S0/K) + (r + sigma**2/2)*(T-t))/(sigma*(np.sqrt(T-t)))
    d2 = d1 - sigma*(np.sqrt(T-t))
    C = S0*(norm.cdf(d1)) - K*np.exp(-r*(T-t))*norm.cdf(d2)
    return C

def BS_EuCall_FiDi_Explicit(r, sigma, a, b, m, nu_max, T, K):
    q = 2*r/sigma**2
    dt_tild  = sigma**2*T/(2*nu_max)
    dx_tild = (b-a)/m
    x_tild = a + np.arange(m+1)*dx_tild
    lambda_ = dt_tild/dx_tild**2
    w = np.maximum(0, np.exp(x_tild/2*(q+1))-np.exp(x_tild/2*(q-1)))

    A = lambda_*(np.tri(m+1, m+1, 1) + np.tri(m+1, m+1, 1).transpose() - 1) + (1 - 3*lambda_)*np.identity(m+1)
    A_pow = np.linalg.matrix_power(A, nu_max)

    ### 3D GRAPH WITHOUT BOUNDARY CONDITIONS
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #print(np.array([np.dot(np.linalg.matrix_power(A, i), w) for i in range(nu_max)])[5, :])
    X = K*np.exp(x_tild)
    Y = range(nu_max)
    X, Y = np.meshgrid(X, Y)
    #print(X)
    surf = ax.plot_surface(X, Y, np.array([K*np.dot(np.linalg.matrix_power(A, i), w)**np.exp(-x_tild/2*(q-1) - 1/2*sigma**2*T*((q-1)**2/4+q)) for i in range(nu_max)]), cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_xlabel("S")
    ax.set_ylabel("$\\nu$")
    ax.set_zlabel("option price")
    ax.set_title("Without boundary conditions...")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    ###

    # High probability that there is an off-by-one error here somewhere
    d = np.ndarray((m+1, nu_max))
    def r_2(nu): return np.exp(1/2*(q+1)*b+1/4*(q+1)**2*dt_tild*nu)-np.exp(1/2*(q-1)*b+1/4*(q-1)**2*dt_tild*nu)
    d[-1, :] = [lambda_/2*(r_2(nu)+r_2(nu+1)) for nu in range(nu_max)]
    d = np.array(d)
    d_cum = functools.reduce(lambda d_nu_i, d_nu_ip1: np.dot(A, d_nu_i) + d_nu_ip1, d.transpose().tolist())

    w = np.dot(A_pow, w) + d_cum

    S = K*np.exp(x_tild)
    v0 = K*w*np.exp(-x_tild/2*(q-1) - 1/2*sigma**2*T*((q-1)**2/4+q))
    return (S, v0)


r = 0.05
sigma = 0.2
a = -0.7
b = 0.4
m = 100
nu_max = 2000
T = 1
K = 100

(S, v0_hat) = BS_EuCall_FiDi_Explicit(r, sigma, a, b, m, nu_max, T, K)

v0 = BlackScholes_EUCall(S, r, sigma, T, K)

plt.plot(S, v0_hat, label="Approx price")
plt.plot(S, v0, label="BS price")
plt.legend()
plt.title("With boundary conditions...")
plt.show()
