from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, linalg
from fbpca import svd, pca
import time


def save_sparse_csr(filename,array, label=[]):
    np.savez(filename,data = array.data ,indices=array.indices,
          indptr =array.indptr, shape=array.shape, label = label )

def load_sparse_csr(filename):
    loader = np.load(filename)
    if 'label' in loader.keys():
        label = loader['label']
    else:
        label = []
    return csr_matrix((  np.log10(1.0+loader['data']), loader['indices'], loader['indptr']),
                      shape = loader['shape']), label

def NE_dn(A, type='ave'):
    m , n = A.shape
    diags = np.ones(m)
    diags[:] = abs(A).sum(axis=1).flatten()
    if type == 'ave':
        D = 1/(diags+np.finfo(float).eps)
        return A*D[:,np.newaxis]
    elif type == 'gph':
        D = 1/np.sqrt(diags+np.finfo(float).eps)
        return (A*D[:,np.newaxis])*D

def Hbeta(D,beta):
    D = (D-D.min())/(D.max() - D.min() + np.finfo(float).eps)
    P = np.exp(-D*beta)
    sumP = P.sum()
    H = np.log(sumP) + beta*sum(D*P)/sumP
    P = P / sumP
    return H, P



def umkl_bo(D, beta):
    tol = 1e-4
    u = 20
    logU = np.log(u)
    H, P = Hbeta(D,beta)
    betamin = -np.inf
    betamax = np.inf
    Hdiff = H - logU
    tries = 0
    while(abs(Hdiff)>tol)&(tries < 30):
        if Hdiff>0:
            betamin = beta
            if np.isinf(betamax):
                beta *= 2.0
            else:
                beta = .5*(beta + betamax)
        else:
            betamax = beta
            if np.isinf(betamin):
                beta /= 2.0
            else:
                beta = .5*(beta + betamin)
        H, P = Hbeta(D,beta)
        Hdiff = H - logU
        tries +=1
    return P

def euclidean_proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n,d = v.shape  # will raise ValueError if v is not 1-D
    # get the array of cumulative sums of a sorted (decreasing) copy of v

    v -= (v.mean(axis = 1)[:,np.newaxis]-1.0/d)
    u = -np.sort(-v)
    cssv = np.cumsum(u,axis = 1)
    # get the number of > 0 components of the optimal solution
    temp = u * np.arange(1,d+1) - cssv +s
    temp[temp<0] = 'nan'
    rho = np.nanargmin(temp,axis = 1)
    #rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[np.arange(n), rho] - s) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    w = (v - theta[:,np.newaxis]).clip(min=0)
    return w


def fast_pca(in_X, no_dim):
    (U, s, Va) = pca(csc_matrix(in_X), no_dim, True, 5)
    del Va
    U[:] = U*np.sqrt(np.abs(s))
    D = 1/(np.sqrt(np.sum(U*U,axis = 1)+np.finfo(float).eps)+np.finfo(float).eps)
    return U*D[:,np.newaxis]




