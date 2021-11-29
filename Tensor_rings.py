# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 10:47:17 2021

@author: adminlocal
"""

import numpy as np
import tensorly as tl
import sys


chemin = ("C:/Users/adminlocal/Desktop/Code_n")
sys.path.append(chemin)
from MPS_rand import  matrix_product_state
from tensorly import tt_to_tensor


def TensorRing(X, TT_ranks):
    # The first dim of the first unfolding of X should be > r0r1, otherwise, counld not reshape C into (-1, r0r1) 
    d = len(X.shape)
    r0, r1 = TT_ranks[0], TT_ranks[1]
    C = X.reshape(X.shape[0], -1)
    U, S, V = tl.partial_svd(C, r0*r1)
    factors = [U.reshape(-1, r0, r1)]

    for k in range(2, d):
        C = np.reshape(np.dot(np.diag(S), V), (TT_ranks[k-1]*X.shape[k-1], -1))
        U, S, V = tl.partial_svd(C, TT_ranks[k])
        factor = U.reshape(TT_ranks[k-1], X.shape[k-1], TT_ranks[k])
        factors.append(factor)
        C = np.dot(np.diag(S), V)
        
    factors.append(C.reshape(TT_ranks[-1], X.shape[-1], TT_ranks[0]))
    #factors.append(C)
        
    return factors





def test_TR(TT_cores):
    Q = len(TT_cores)
    I1, I2, I3, I4 = TT_cores[0].shape[1], TT_cores[1].shape[1], TT_cores[2].shape[1], TT_cores[-1].shape[1]
    X = np.zeros((I1, I2, I3, I4))
    for q in range(Q):
        for i1 in range(I1):
            for i2 in range(I2):
                for i3 in range(I3):
                    for i4 in range(I4):
                        temp = np.dot(TT_cores[0][:,i1,:], TT_cores[1][:,i2,:])
                        X[i1,i2,i3,i4] = np.trace( np.dot( temp, np.dot(TT_cores[2][:,i3,:], TT_cores[-1][:,i4,:]) ) )
                        
    return X

if __name__ == '__main__':
    r0= r1= r2= r3 = 3
    TT_ranks = [r0, r1, r2, r3]
    I1 = 9
    I2 = 480
    I3= 640
    I4 = 4
    #Construire un tenseur a partir des TT-coeurs générés aléatoirement
    G0 = np.random.rand(r0, I1, r1)
    G1 = np.random.rand(r1, I2, r2)
    G2 = np.random.rand(r2, I3, r3)
    G3 = np.random.rand(r3, I4, r0)
    
    TT_coeurs = [G0, G1, G2, G3]    
    
    #X = test_TR(TT_coeurs)
    X = np.random.randn(I1, I2, I3, I4)
    factors = TensorRing(X, TT_ranks)
    #X_reconst = test_TR(TT_coeurs)