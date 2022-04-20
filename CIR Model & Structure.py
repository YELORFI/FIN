import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#------------------------------------------------------------------------------

def CIR_process(r0, theta, kappa, sigma, T = 1., N = 100, seed = 0):
    if seed != 0:
        np.random.seed(seed)

    dt = T/N
    rates = [r0]
    for i in range(N):
        dr = kappa*(theta-rates[-1])*dt + sigma*np.sqrt(rates[-1])*np.random.normal(size = 1, scale = np.sqrt(dt))
        rates.append(rates[-1] + dr)
    
    return(pd.DataFrame(data = rates, index = [x*dt for x in range(N+1)] ))

#------------------------------------------------------------------------------

def CIR_A(theta, kappa, sigma, T):
    gamma = np.sqrt(kappa**2+2*sigma**2)
    tmp = -2*kappa*theta/sigma**2
    tmp = tmp*np.log( 2*gamma*np.exp((gamma+kappa)*T/2) / ( (gamma+kappa)*(np.exp(gamma*T)-1) +2*gamma ))
    return(tmp)

#------------------------------------------------------------------------------

def CIR_B(kappa, sigma, T):
    gamma = np.sqrt(kappa**2+2*sigma**2)
    tmp = 2*(np.exp(gamma*T)-1) /((gamma+kappa)*(np.exp(gamma*T)-1) +2*gamma )
    return(tmp)
