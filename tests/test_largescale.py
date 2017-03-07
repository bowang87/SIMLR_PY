import os
import sys
import scipy.io as sio
sys.path.insert(0,os.path.abspath('..'))
import time
import numpy as np
import SIMLR
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn.metrics.cluster import adjusted_rand_score as ari
from scipy.sparse import csr_matrix


filename = 'Zeisel.mat'
X = csr_matrix(sio.loadmat(filename)['X']) #loading single-cell RNA-seq data
X.data = np.log10(1+X.data) ##take log transform of gene counts. This is very important since it makes the data more gaussian
label = sio.loadmat(filename)['true_labs'] #this is ground-truth label for validation
c = label.max() # number of clusters
### if the number of genes are more than 500, we recommend to perform pca first!
print('Start to Run PCA on the RNA-seq data!\n')
start_main = time.time()
if X.shape[1]>500:
    X = SIMLR.helper.fast_pca(X,500)
else:
    X = X.todense()
print('Successfully Run PCA! PCA took %f seconds in total\n' % (time.time() -             start_main))
print('Start to Run SIMLR!\n')
start_main = time.time()
simlr = SIMLR.SIMLR_LARGE(c, 30, 0); ###This is how we initialize an object for SIMLR. the first input is number of rank (clusters) and the second input is number of neighbors. The third one is an binary indicator whether to use memory-saving mode. you can turn it on when the number of cells are extremely large to save some memory but with the cost of efficiency.
S, F,val, ind = simlr.fit(X)
print('Successfully Run SIMLR! SIMLR took %f seconds in total\n' % (time.time() -         start_main))
y_pred = simlr.fast_minibatch_kmeans(F,c)
print('NMI value is %f \n' % nmi(y_pred.flatten(),label.flatten()))
print('ARI value is %f \n' % ari(y_pred.flatten(),label.flatten()))

