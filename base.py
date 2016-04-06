
import numpy as np 
from numpy.linalg import solve
from scipy.stats import norm 
from scipy.stats import poisson
from scipy.optimize import minimize
class Normal:
    def __init__(self,mu,sd,cdist,pc): 
        self.mu = mu
        self.sd = sd 
        self.pc = pc 
        self.cdist = cdist
class Poisson: 
    def __init__(self,mu,dist,pc): 
        self.mu = mu
        self.pc = pc 
        self.dist = dist


def func1 ( ww, ee, ll ):
    pc = 1/sum([1/ww,1/ee,1/ll])
    x  = [0, 0, 0] 
    x[0] = 1/sum([1,ww/ee,ww/ll]) 
    x[1] = 1/sum([ee/ww,1,ee/ll]) 
    x[2] = 1/sum([ll/ww,ll/ee,1]) 
    x = np.cumsum(x) 
    sd = 1/(norm.ppf(x[1])-norm.ppf(x[0]))
    mu = -((norm.ppf(x[0])+norm.ppf(x[1]))*sd)/2 
    res = Normal(mu,sd,x,pc) 
    return res


def func2( x ): 
    x = 1.0/np.array(x)
    pc = 1.0/sum(x)
    x = pc*x 
    nn = len(x)-1
    def func(ll):
        y = range(0,nn)
        z = y[-1]
        y = np.append( poisson.pmf(y, ll) , [poisson.sf(z, ll)] )
        return y 
    def fun(ll):
        y = func(ll)
        y -= x 
        return sum(y*y)
    mu = minimize(fun, 1, tol=1e-20 ).x[0] 
    res = Poisson(mu,x,pc) 
    return  res


