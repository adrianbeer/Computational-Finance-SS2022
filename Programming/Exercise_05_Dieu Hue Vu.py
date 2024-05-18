import math
import numpy as np
import matplotlib.pyplot as plt

# task a: Function to calculate time series of log returns
def log_returns(data):
    N=len(data)
    data=np.log(data)
    ln=np.empty(N-1)
    for i in range(0,N-1):
        ln[i]=data[i+1]-data[i]
    return ln

# task b

# import data, flip data and remove NaN
dax = np.genfromtxt('time_series_dax_2022.csv',delimiter=';',skip_header=1,usecols=4) #close column
dax = np.flip(dax)
dax=dax[np.logical_not(np.isnan(dax))]

# calculate time series of log returns
ln=log_returns(dax)

# visualize
plt.plot(ln)
plt.ylabel('Log returns of DAX index')
plt.show()

# compute annualized empirical mean and standard deviation
AE_mean=250*np.mean(ln)
print('Annualized empirical mean of log-returns: ',str(AE_mean))

AE_sd=np.sqrt(250*np.var(ln,ddof=1))
print('Annualized empirical standard deviation of log-returns: ',str(AE_sd))

#compare to normal distributed time series
normal_distributed=np.random.normal(AE_mean/250,AE_sd/math.sqrt(250),len(ln))
plt.plot(normal_distributed,'r',label='Normal distribution')
plt.plot(ln,'b',label='DAX data')
plt.title('Compare log returns of DAX to Normal distributed series')
plt.legend()
plt.show()
"""comments: 2 time series are quite close. 
However, the large daily price changes happen much more often in the DAX data than in the normal distributed data.
"""

