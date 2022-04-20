import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------

def hw_process_spline(r0, kappa, sigma, forward_spl, T = 5., N = 100, seed = 0):
    
    if seed != 0:
        np.random.seed(seed)
    dt = T/N
    dW = np.random.normal(size = N, scale = np.sqrt(dt))
    t = 0.
    rates = [r0]
    for i in range(N):
        dr = kappa*( theta_hw(forward_spl, kappa, sigma, t) / kappa - rates[-1] )*dt
        dr += sigma*dW[i]
        dr = float(dr)
        t += dt
        rates.append(rates[-1] + dr)
    return(pd.DataFrame(data = rates, index = [x*dt for x in range(N+1)] ))


def theta_hw(forward_spl, kappa, sigma, t):
    tmp = forward_spl.derivative()(t)
    tmp += forward_spl(t)*kappa
    tmp += sigma**2/2/kappa*(1-np.exp(-2*kappa*t))
    return(tmp)


def hw_B(kappa, T, t=0.):
    if t > 0:
        T = T-t
    tmp = 1 - np.exp(-kappa*T)
    tmp = tmp / kappa
    return(tmp)
    

def hw_A(kappa, sigma, forward_spl, T, t = 0.):
    P0T = np.exp(-forward_spl.integral(0,T))
    P0t = np.exp(-forward_spl.integral(0,t))
    B = hw_B(kappa, T, t)
    tmp = B*forward_spl(t) 
    tmp = tmp - sigma**2/4/kappa*B**2*(1-np.exp(-2*kappa*t))
    A = P0T/P0t
    A = A*np.exp(tmp)
    return(A)
    
    
def hw_P(kappa, sigma, forward_spl, r0, T, t = 0.):
    return( float(hw_A(kappa, sigma, forward_spl, T, t) * np.exp( -hw_B(kappa, T, t) * r0 ) ) ) 
