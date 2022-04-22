import numpy as np
import matplotlib.pyplot as plt

# a)
def log_returns(data: np.ndarray):
    l =  np.diff(np.log(data))
    assert l.shape[0] == data.shape[0] - 1
    return l

# b)
def ann_emp_mean(l: np.ndarray):
    return 250/(len(l)-1)*sum(l)

def ann_emp_sd(l: np.ndarray):
    mu_hat = ann_emp_mean(l)
    return np.sqrt(250/(len(l)-2)*sum(np.square(l - mu_hat/250)))

s = np.genfromtxt("time_series_dax_2022.csv", delimiter=';', usecols=4, skip_header=True)
l = log_returns(s)
mu_hat = ann_emp_mean(l)
sig_hat = ann_emp_sd(l)

# plotting
f, ax = plt.subplots(1,1)
ax.plot(l, label='observed', color='b', linewidth=0.5)
ax.set_ylabel("log-return")
ax.set_xlabel("day")
f.suptitle("$\hat{\mu}=%0.2f, \hat{\sigma}=%0.2f$" % (mu_hat, sig_hat))

# c)
l_sim = np.random.normal(mu_hat/250, sig_hat/np.sqrt(250), len(l)) # divide by 250 to reverse annualization
ax.plot(l_sim, label="simulated", color='r', linewidth=0.5)
ax.legend()
plt.show()

# d)
''' Observed time series (ts) of log-returns has periods of noticeable higher volatility than the simulated ts. 
These higher volatilities occur in clusters => observed log-returns are heteroskedastic and autoregressive; in contrast
to the simulated ts which is homoskedastic and i.i.d.
'''