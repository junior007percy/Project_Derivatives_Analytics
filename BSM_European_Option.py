# %%
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt 
mpl.rcParams['font.family'] = 'serif'
from scipy.integrate import quad

# %%
# Helper Functions

# %%
def dN(x):
    '''Probability density function of standard normal random variable x.'''
    return math.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)

def N(d):
    '''Cumulative density function of standard normal random variables x.'''
    return quad(lambda x: dN(x), -20, d, limit=50)[0]

def d1f(St, K, t, T, r, sigma):
    '''Black-Scholes-Merton d1 function.
       Parameters see e.g. BSM_call_value function.'''
    d1 = (math.log(St / K) + (r + 0.5 * sigma ** 2)
            * (T-t)) / (sigma * math.sqrt(T - t))
    return d1

# %%
# Valuation Functions

# %%
def BSM_call_value(St, K, t, T, r, sigma):
    '''Calculate Black-Scholes-Merton European call option value.
       Parameters
       ==========
       St: float
           stock/index level at time t
       K: float
          strike price
       t: float
          valuation date
       T: float
          date of maturity/time-to-maturity if t = 0; T > t
       r: float
          constant, risk-less short rate
       sigma: float
              volatility
              
       Returns
       =======
       call_value: float
           European call present value at t
    '''
    
    d1 = d1f(St, K, t, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T - t)
    call_value = St * N(d1) - math.exp(-r * (T -t)) * K * N(d2)
    return call_value

def BSM_put_value(St, K, t, T, r, sigma):
    '''Calculate Black-Scholes-Merton European call option value.
       Parameters
       ==========
       St: float
           stock/index level at time t
       K: float
          strike price
       t: float
          valuation date
       T: float
          date of maturity/time-to-maturity if t = 0; T > t
       r: float
          constant, risk-less short rate
       sigma: float
              volatility
              
       Returns
       =======
       put_value: float
           European put present value at t
    '''
    
    put_value = BSM_call_value(St, K, t, T, r, sigma) - St + math.exp(-r * (T -t)) * K
    return put_value


