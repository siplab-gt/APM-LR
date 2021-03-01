#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from enum import Enum
import numpy as np
import scipy.special as sc
import scipy.stats as st


"""
acquisition functions for active LR
"""


class Utility(Enum):
    RANDOM = 0
    INFOGAIN = 1
    UNCERTAINTY = 2
    APMLR = 3
    BALD = 4
    MAXVAR = 5

def get_utilfun(utility):
    """
    returns corresponding utility function for given method
    """
    if utility == Utility.INFOGAIN:
        return infogain
    elif utility == Utility.UNCERTAINTY:
        return uncertainty
    elif utility == Utility.APMLR:
        return apmlr
    elif utility == Utility.BALD:
        return bald
    elif utility == Utility.MAXVAR:
        return maxvar
    else:
        raise ValueError('Specify valid utility method')
    

def select_example(X, LRobj, utility, method_type, metadata=None, doplot=False):
    """
    X: (Npoints,d) array of Npoints number of d-dimensional examples
    LRobj: logistic regression object
    utility: utility function type (e.g. Utility.APMLR)
    method_type: specify method type
    metadata: data and classifier statistics used in active selection
    doplot: enable methods plotting
    """

    if utility == Utility.RANDOM:
        return np.random.randint(len(X)), -1, {'plot_name':'Random'}

    idx_best = -1
    best_val = -np.inf

    for ii in range(len(X)):

        x = X[ii]
        utilfun = get_utilfun(utility)

        util,_ = utilfun(x=x, LRobj=LRobj, method_type=method_type, metadata=metadata, doplot=False)

        if util > best_val:
            best_val = util
            idx_best = ii

    assert idx_best >= 0
         
    _,options = utilfun(x=X[idx_best], LRobj=LRobj, method_type=method_type,
                     metadata=metadata, doplot=doplot)
    
    return idx_best, best_val, options


def infogain(x, LRobj, method_type, metadata=None, doplot=False):
    """
    InfoGain acquisition function
    
    x: d-dimensional example
    LRobj: logistic regression object
    
    method_type: specify method type
        0: standard infogain
        1: accurate infogain
        
    metadata: data and classifier statistics used in active selection
    doplot: enable methods plotting
    """
    
    if method_type==0:
        npos=100
        plotname = 'InfoGain'
    elif method_type==1:
        npos=1000
        plotname = 'InfoGain - accurate'
    else:
        raise ValueError('Invalid infogain method type')
        
    likelihood = np.zeros(npos)
    entropy = np.zeros(npos)

    pos_samp = LRobj.sample_posterior(npos, full=True)
    likelihood = LRobj.likelihood(x, 1, pos_samp)
    entropy = binary_ent(likelihood)

    mean_like = np.mean(likelihood)
    ent_like = binary_ent(mean_like)
    mean_ent = np.mean(entropy)
    info = ent_like - mean_ent
    
    options = {'npos':npos, 'plot_name':plotname, 'likelihood':mean_like,
               'ent':ent_like, 'cond_ent':mean_ent}
        
    return info, options


def uncertainty(x, LRobj, method_type=0, metadata=None, doplot=False):
    """
    Uncertainty acquisition function
    
    x: d-dimensional example
    LRobj: logistic regression object
    
    method_type: specify method type
        0: standard closest to hyperplane
        
    metadata: data and classifier statistics used in active selection
    doplot: enable methods plotting
    """
    
    options = {'plot_name':'Closest to hyperplane'}
    
    weight,bias = LRobj.get_classifier()
    
    if bias:
        util = -np.abs(np.dot(weight,x) - bias)
    else:
        util = -np.abs(np.dot(weight,x))
        
    return util, options
    

def apmlr(x, LRobj, method_type, metadata, doplot=False):
    """
    APM-LR acquisition function
    
    x: d-dimensional example
    LRobj: logistic regression object
    
    method_type: specify method type
        0: exploit and explore
        1: exploit only
        2: explore only
        
    metadata: data and classifier statistics used in active selection
    doplot: enable methods plotting
    """
    
    # unpack metadata
    Wmean = metadata['Wmean']
    Wcov = metadata['Wcov']
    B = metadata['B']

    # unpack covariance
    d = len(x)
    
    if len(Wmean) == d:
        xp = x
        xmax_sqnorm = B**2
    elif len(Wmean) == d+1:
        xp = np.append(x,-1)
        xmax_sqnorm = B**2 + 1
    else:
        raise ValueError('LRobj has incorrect dimension')
         
    # calculate marginals
    VL = (xp.T).dot(Wcov).dot(xp)
    EL = xp.dot(Wmean)
    
    # switch APM method
    if method_type==0:
        P = xmax_sqnorm*metadata['maxeig']
        plotname = 'APM-LR'
    elif method_type==1:
        P = xmax_sqnorm*metadata['maxeig']
        plotname = 'APM-LR (exploit)'
    elif method_type == 2:
        P = xmax_sqnorm*metadata['maxeig']
        plotname = 'APM-LR (explore)'
    else:
        raise ValueError('Invalid APM method type')

    options = {'P':P, 'plot_name':plotname,
               'EL':EL, 'VL':VL}
    
    if method_type==0:
        util = -(EL**2 + (np.sqrt(VL) - np.sqrt(2/np.pi*P))**2)
    elif method_type==1:
        util = -EL**2
    elif method_type==2:
        util = -(np.sqrt(VL) - np.sqrt(2/np.pi*P))**2
        
    return util, options


def bald(x, LRobj, method_type, metadata, doplot=False):
    """
    BALD acquisition function
    
    x: d-dimensional example
    LRobj: logistic regression object
    
    method_type: specify method type
        0: standard BALD approximation
        
    metadata: data and classifier statistics used in active selection
    doplot: enable methods plotting
    """
    
    # unpack metadata
    Wmean = metadata['Wmean']
    Wcov = metadata['Wcov']

    # unpack covariance
    d = len(x)
    
    if len(Wmean) == d:
        xp = x
    elif len(Wmean) == d+1:
        xp = np.append(x,-1)
    else:
        raise ValueError('LRobj has incorrect dimension')
    
    # calculate marginals
    VL = (xp.T).dot(Wcov).dot(xp)
    EL = xp.dot(Wmean)
    
    # constants
    k = np.sqrt(np.pi/8) # probit constant, Phi(k*a) for input a, chosen to match slope of logistic at a=0
    C = np.sqrt(np.pi*np.log(2)/2)
    
    EL_k = k*EL
    VL_k = k**2*VL
    
    VCden = VL_k + C**2
    
    util = (binary_ent(st.norm.cdf(EL_k/np.sqrt(VL_k + 1))) -
            C*np.exp(-EL_k**2/(2*VCden))/np.sqrt(VCden))
    
    options = {'plot_name':'BALD approximation', 'EL':EL, 'VL':VL}

    return util, options


def maxvar(x, LRobj, method_type, metadata, doplot=False):
    """
    MaxVar acquisition function
    
    x: d-dimensional example
    LRobj: logistic regression object
    
    method_type: specify method type
        0: maximize variance
        
    metadata: data and classifier statistics used in active selection
    doplot: enable methods plotting
    """
    
    # unpack metadata
    Wmean = metadata['Wmean']
    Wcov = metadata['Wcov']

    # unpack covariance
    d = len(x)
    
    if len(Wmean) == d:
        xp = x
    elif len(Wmean) == d+1:
        xp = np.append(x,-1)
    else:
        raise ValueError('LRobj has incorrect dimension')
    
    # calculate marginals
    VL = (xp.T).dot(Wcov).dot(xp)
    
    util = VL
    
    options = {'plot_name':'Maximum variance', 'VL':VL}

    return util, options


def binary_ent(p):
    # helper function
    return -np.log2(np.e)*(sc.xlogy(p,p) + sc.xlog1py(1-p,-p))
