import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from scipy import integrate
from cmath import exp
import  matplotlib.pyplot as plt

def BlackScholes_EUCall(S0, r, sigma, T, K, *args, **kwargs):
    t = 0
    d1 = (np.log(S0/K) + (r + sigma**2/2)*(T-t))/(sigma*(np.sqrt(T-t)))
    d2 = d1 - sigma*(np.sqrt(T-t))
    C = S0*(norm.cdf(d1)) - K*exp(-r*(T-t))*norm.cdf(d2)
    return C

# a)
def ImplVol(S0, r, T, K, V):
    def loss_f(sigma):
        return np.square(V - BlackScholes_EUCall(S0, r, sigma, T, K))
    sigma_hat = minimize_scalar(loss_f, method='brent')
    return sigma_hat.x

params = dict(S0=100, r=0.05, T=1, K=100, V=6.09)

res = ImplVol(**params)
print(f"Estimated implied volatility: {res:.4f}")
print(f"BlackScholes Price: {BlackScholes_EUCall(**params, sigma=res)}")


# b)
def Heston_EuPut_Laplace(S0, r, nu0, kappa, _lambda, sigma_tilde, T, K, R):
    def d(u): return np.sqrt(_lambda**2 + complex(u**2, u)*sigma_tilde**2)

    # cosh(x) and sinh(x) explode for large Re(x), but char_func(complex(x, -R))dx gives us very large x.
    def denom_hyperb(u):
        return np.cosh(d(u)*T/2) + _lambda*np.sinh(d(u)*T/2)/d(u)

    def numer_hyperb(u): return complex(u**2, u)*np.sinh(d(u)*T/2)/d(u)

    def char_func(u):
        return exp(complex(0, u)*(np.log(S0) + r*T)) * \
        np.power(exp(_lambda*T/2)/denom_hyperb(u), 2*kappa/sigma_tilde**2) * \
        exp(-nu0*numer_hyperb(u)/denom_hyperb(u))

    def f_tilde(z):
        return np.power(K, (1-z)) / (z**2-z)

    def integrand(u): return np.real(f_tilde(complex(R, u)) * char_func(complex(u, -R)))

    # Can only integrate to about 900, otherwise the terms explode
    # and can't be evaluated properly anymore
    integral = integrate.quad(func=integrand, a=0, b=900)[0]
    V = exp(-r*T)/np.pi*integral
    return V

K_list = list(range(50, 150+1))
params = dict(S0=100, r=0.05, nu0=0.3**2, kappa=0.3**2, _lambda=2.5, sigma_tilde=0.2,
              T=1, R=-0.001)
# We need to choose a negative R, because the Laplace transform of the put payoff function
# is only defined for Re(z) < 0, i.e. R < 0.
laplace_prices = [Heston_EuPut_Laplace(**params, K=K) for K in K_list]
plt.plot(K_list, [laplace_prices])
plt.grid()
plt.show()

def ImplVol_Heston(S0, r, nu0, kappa, _lambda, sigma_tilde, T, K, R):
    euput_heston_price = Heston_EuPut_Laplace(S0, r, nu0, kappa, _lambda, sigma_tilde, T, K, R)
    eucall_heston_price = euput_heston_price + S0 - K*exp(-r*T)
    # Because ImplVol calculates the implied vola for a **call** we need to
    # to get the call prices from the heston put prices (using put call parityship).
    impl_vol_heston = ImplVol(S0, r, T, K, eucall_heston_price)
    return impl_vol_heston

laplace_impl_vols = [ImplVol_Heston(**params, K=K) for K in K_list]
plt.plot(K_list, laplace_impl_vols)
plt.show()
