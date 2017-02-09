"""
    Functions for large scale SIMLR and accuracy checks

    ---------------------------------------------------------------------

    This module contains the following functions:

    save_sparse_csr
    save a sparse csr format input of single-cell RNA-seq data
    load_sparse_csr
    load a sparse csr format input of single-cell RNA-seq data
    nearest_neighbor_search
    Approximate Nearset Neighbor search for every cell
    NE_dn
    Row-normalization of a matrix
    mex_L2_distance
    A fast way to calculate L2 distance
    Cal_distance_memory
    Calculate Kernels in a memory-saving mode
    mex_multipleK
    A fast way to calculate kernels
    Hbeta
    A simple LP method to solve linear weight
    euclidean_proj_simplex
    A fast way to calculate simplex projection
    fast_pca
    A fast randomized pca with sparse input
    fast_minibatch_kmeans
    A fast mini-batch version of k-means
    SIMLR_Large
    A large-scale implementation of our SIMLR
    ---------------------------------------------------------------------

    Copyright 2016 Bo Wang, Stanford University.
    All rights reserved.
    """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from . import helper
import numpy as np
import sys
import os
from annoy import AnnoyIndex
import scipy.io as sio
from scipy.sparse import csr_matrix, csc_matrix, linalg
from fbpca import svd, pca
import time
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans, KMeans

class SIMLR_LARGE(object):
    """A class for large-scale SIMLR.

    Attributes:
        num_of_rank: The rank hyper-parameter in SIMLR usually set to number of clusters.
        num_of_neighbors: the number of neighbors kept for each cell to approximate full cell similarities
        mode_of_memory: an indicator to open the memory-saving mode. This is helpful for datasets of millions of cells. It will sacrify a bit speed though.
    """
    def __init__(self, num_of_rank, num_of_neighbor=30, mode_of_memory = False, max_iter = 5):
        self.num_of_rank = int(num_of_rank)
        self.num_of_neighbor = int(num_of_neighbor)
        self.mode_of_memory = False
        self.max_iter = int(max_iter)

    def nearest_neighbor_search(self, GE_csc):
        K = self.num_of_neighbor * 2
        n,d = GE_csc.shape
        t = AnnoyIndex(d)
        for i in xrange(n):
            t.add_item(i,GE_csc[i,:])
        t.build(d)
        t.save('test.ann')
        u = AnnoyIndex(d)
        u.load('test.ann')
        os.remove('test.ann')
        val = np.zeros((n,K))
        ind = np.zeros((n,K))
        for i in xrange(n):
            tmp, tmp1 = u.get_nns_by_item(i,K, include_distances=True)
            ind[i,:] = tmp
            val[i,:] = tmp1
        return ind.astype('int'), val




    def mex_L2_distance(self, F, ind):
        m,n = ind.shape
        I = np.tile(np.arange(m), n)
        temp = np.take(F, I, axis = 0) - np.take(F, ind.ravel(order = 'F'),axis=0)
        temp = (temp*temp).sum(axis=1)
        return temp.reshape((m,n),order = 'F')



    def Cal_distance_memory(self, S, alpha):
        NT = len(alpha)
        DD = alpha.copy()
        for i in xrange(NT):
            temp = np.load('Kernel_' + str(i)+'.npy')
            if i == 0:
                distX = alpha[0]*temp
            else:
                distX += alpha[i]*temp
            DD[i] = ((temp*S).sum(axis = 0)/(S.shape[0]+0.0)).mean(axis = 0)
        alphaK0 = umkl_bo(DD, 1.0/len(DD));
        alphaK0 = alphaK0/np.sum(alphaK0)
        return distX, alphaK0


    def mex_multipleK(self, val, ind):
        val *= val
        KK = self.num_of_neighbor
        ismemory = self.mode_of_memory
        m,n=val.shape
        sigma = np.arange(1,2.1,.25)
        allK = (np.arange(np.ceil(KK/2.0), min(n,np.ceil(KK*1.5))+1, np.ceil(KK/10.0))).astype('int')
        if ismemory:
            D_kernels = []
            alphaK = np.ones(len(allK)*len(sigma))/(0.0 + len(allK)*len(sigma))
        else:
            D_kernels = np.zeros((m,n,len(allK)*len(sigma)))
            alphaK = np.ones(D_kernels.shape[2])/(0.0 + D_kernels.shape[2])
        t = 0;
        for k in allK:
            temp = val[:,np.arange(k)].sum(axis=1)/(k+0.0)
            temp0 = .5*(temp[:,np.newaxis] + np.take(temp,ind))
            temp = val/temp0
            temp*=temp
            for s in sigma:
                temp1 =  np.exp(-temp/2.0/s/s)/np.sqrt(2*np.pi)/s/temp0
                temptemp = temp1[:, 0]
                temp1[:] = .5*(temptemp[:,np.newaxis] + temptemp[ind]) - temp1
                if ismemory:
                    np.save('Kernel_' + str(t), temp1 - temp1.min())
                else:
                    D_kernels[:,:,t] = temp1 - temp1.min()
                t = t+1

        return D_kernels, alphaK


    def fast_eigens(self, val, ind):
        n,d = val.shape
        rows = np.tile(np.arange(n), d)
        cols = ind.ravel(order='F')
        A = csr_matrix((val.ravel(order='F'),(rows,cols)),shape = (n, n)) + csr_matrix((val.ravel(order='F'),(cols,rows)),shape = (n, n))
        (d,V) = linalg.eigsh(A,self.num_of_rank,which='LM')
        d = -np.sort(-np.real(d))
        return np.real(V),d/np.max(abs(d))
    def fast_minibatch_kmeans(self, X,C):
        cls = MiniBatchKMeans(n_clusters=C, n_init = 100, max_iter = 100)
        return cls.fit_predict(X)

    def fit(self, X, beta = 0.8):
        K = self.num_of_neighbor
        is_memory = self.mode_of_memory
        c = self.num_of_rank
        NITER = self.max_iter
        n,d = X.shape
        if d > 500:
            print('SIMLR highly recommends you to perform PCA first on the data\n');
            print('Please use the in-line function fast_pca on your input\n');
        ind, val = self.nearest_neighbor_search(X)
        del X
        D_Kernels, alphaK = self.mex_multipleK(val, ind)
        del val
        if is_memory:
            distX,alphaK = self.Cal_distance_memory(np.ones((ind.shape[0], ind.shape[1])), alphaK)
        else:
            distX = D_Kernels.dot(alphaK)
        rr = (.5*(K*distX[:,K+2] - distX[:,np.arange(1,K+1)].sum(axis = 1))).mean()
        lambdar = rr
        S0 = distX.max() - distX
        S0[:] = helper.NE_dn(S0)
        F, evalues = self.fast_eigens(S0.copy(), ind.copy())
        F = helper.NE_dn(F)
        F *= (1-beta)*d/(1-beta*d*d);
        F0 = F.copy()
        for iter in range(NITER):
            FF = self.mex_L2_distance(F, ind)
            FF[:] = (distX + lambdar*FF)/2.0/rr
            FF[:] = helper.euclidean_proj_simplex(-FF)
            S0[:] = (1-beta)*S0 + beta*FF

            F[:], evalues = self.fast_eigens(S0, ind)
            F *= (1-beta)*d/(1-beta*d*d);
            F[:] = helper.NE_dn(F)
            F[:] = (1-beta)*F0 + beta*F
            F0 = F.copy()
            lambdar = lambdar * 1.5
            rr = rr / 1.05
            if is_memory:
                distX, alphaK0 = self.Cal_distance_memory(S0, alphaK)
                alphaK = (1-beta)*alphaK + beta*alphaK0
                alphaK = alphaK/np.sum(alphaK)

            else:
                DD = ((D_Kernels*S0[:,:,np.newaxis]).sum(axis = 0)/(D_Kernels.shape[0]+0.0)).mean(axis = 0)
                alphaK0 = helper.umkl_bo(DD, 1.0/len(DD));
                alphaK0 = alphaK0/np.sum(alphaK0)
                alphaK = (1-beta)*alphaK + beta*alphaK0
                alphaK = alphaK/np.sum(alphaK)
                distX = D_Kernels.dot(alphaK)

        if is_memory:
            for i in xrange(len(alphaK)):
                os.remove('Kernel_' + str(i) + '.npy')
        rows = np.tile(np.arange(n), S0.shape[1])
        cols = ind.ravel(order='F')
        val = S0
        S0 = csr_matrix((S0.ravel(order='F'),(rows,cols)),shape = (n, n)) + csr_matrix((S0.ravel(order='F'),(cols,rows)),    shape = (n, n))
        return S0, F, val, ind



