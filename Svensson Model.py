import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------

def svensson_yields(beta0, beta1, beta2, beta3, tau1, tau2, T):
    T1 = T/tau1
    T2 = T/tau2
    tmp = beta0
    tmp += beta1 * (1-np.exp(-T1))/T1
    tmp += beta2 * ( (1-np.exp(-T1))/T1 - np.exp(-T1) )
    tmp += beta3 * ( (1-np.exp(-T2))/T2 - np.exp(-T2) )
    
    return(tmp)
