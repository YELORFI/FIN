import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


from svensson import *

#------------------------------------------------------------------------------

def vasicek(r0, theta, kappa, sigma, T = 1., N = 100, seed = 0):
    if seed != 0:
        np.random.seed(seed)

    dt = T/N
    rates = [r0]
    for i in range(N):
        dr = kappa*(theta-rates[-1])*dt + sigma*np.random.normal(size = 1, scale = np.sqrt(dt))
        dr = float(dr)
        rates.append(rates[-1] + dr)
    
    return(pd.DataFrame(data = rates, index = [x*dt for x in range(N+1)] ))

#------------------------------------------------------------------------------

def vasicek_mean(r0, theta, kappa, sigma, T = 1., N = 100):
    t = range(N+1)
    dt = T/N
    t = [x*dt for x in t]
    y = [np.exp(-kappa*x)*r0+theta*(1-np.exp(-kappa*x)) for x in t]
    y = np.array(y)
    return(pd.DataFrame(data = y, index = t))

#------------------------------------------------------------------------------

def vasicek_sd(r0, theta, kappa, sigma, T = 1., N = 100, alpha = 0.90):
    dt = T/N
    alpha = 1-alpha
    t = range(N+1)
    t = [x*dt for x in t]
 
    y = np.sqrt(np.array([sigma**2/2/kappa*(1-np.exp(-2*kappa*x)) for x in t]))

    means = vasicek_mean(r0, theta, kappa, sigma, T, N)
    lower = pd.DataFrame(data = y*norm.cdf(1-alpha/2), index = means.index)
    upper = means + lower
    lower = means - lower
        
    return(lower, upper)

#------------------------------------------------------------------------------

def vasicek_B(kappa, T, t = 0.):
    if t > 0:
        T = T-t
    return( (1-np.exp(-kappa*T))/kappa )

def vasicek_B_prime(kappa, T):
    return(1 - kappa*vasicek_B(kappa, T))

def vasicek_B_2prime(kappa, T):
    return(-kappa + kappa**2*vasicek_B(kappa, T))


#------------------------------------------------------------------------------
    
def vasicek_A(theta, kappa, sigma, T, t = 0.):
    if t > 0:
        T = T-t
    tmp = -(theta-sigma**2/2/kappa**2)*(vasicek_B(kappa, T, t) - T + t)
    tmp += sigma**2/4/kappa * vasicek_B(kappa, T, t)**2
    return(tmp)

def vasicek_A_prime(kappa, theta, sigma, T):
    return(kappa*theta*vasicek_B(kappa, T) - sigma**2/2*vasicek_B(kappa, T))

def vasicek_A_2prime(kappa, theta, sigma, T):
    tmp = kappa*theta*vasicek_B_prime(kappa, T)
    tmp -= sigma**2*vasicek_B(kappa, T)*vasicek_B_prime(kappa, T)
    return(tmp)

#------------------------------------------------------------------------------
