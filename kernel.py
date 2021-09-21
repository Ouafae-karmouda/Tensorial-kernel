# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 09:04:04 2021

@author: adminlocal
"""
import sys

chemin = ("C:/Users/adminlocal/Desktop/Code_n")
sys.path.append(chemin)

import tlinalg
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.metrics import  accuracy_score
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize


import time
#Fct : pseudo inverse of a tensor
#Fct : Compute projectors
#Fct : single tensorial kernel
#Fct : kernel matrix

#Possible normlisation ?

def pseudo_inv(G):# pseudo_inv of G
    temp = tlinalg.t_inv(  tlinalg.t_product(tlinalg.t_transpose(G), G) )
    
    return tlinalg.t_product(temp, tlinalg.t_transpose(G))

def projector(G):#G is a tensor of order 3
    G_t = tlinalg.t_transpose(G)
    pinv_t = pseudo_inv(G_t)
    
    return tlinalg.t_product(G_t, pinv_t)

def projector_matrix(M):#M is a matrix
    return np.dot(M, M.T)
    


def get_single_tensorial_kernel(factors_X, factors_Y, rank, gamma):
    
    #factors_X = matrix_product_state(X, rank, verbose=False, type_svd = 'partial_svd')
    #factors_Y = matrix_product_state(Y, rank, verbose=False, type_svd = 'partial_svd')
    Q = len(factors_X)
    
    M1 = np.squeeze(factors_X[0])
    M2 = np.squeeze(factors_Y[0])
    K1 = (np.linalg.norm(projector_matrix(M1) - projector_matrix(M2), 'fro'))**2
    
    M1 = np.squeeze(factors_X[-1])
    M2 = np.squeeze(factors_Y[-1])
    K2 = (np.linalg.norm(projector_matrix(M1) - projector_matrix(M2), 'fro'))**2
    
    prod = 1
    for q in range(1, Q-1):
        G1 = factors_X[q]
        G2 = factors_Y[q]
        prod = prod*(np.linalg.norm(projector(G1) - projector(G2)))**2
        
    return np.exp(-gamma*K1*K2*prod)
        
def normalise_list_data(X1_CP_data):
    
    X1_CP_data_norm = []
    for data in X1_CP_data:
         
         data_norm = [normalize(l, norm='l2', axis=0, copy=True, return_norm=False) for l in data]
         X1_CP_data_norm.append(data_norm)
     
    return  X1_CP_data_norm 

def kernel_mat(X1_data, X2_data, rank, gamma):#X1_data : list of data factors; X1_data[0]: factors of the 1st data

    #X1_data = normalise_list_data(X1_data)
    #X2_data = normalise_list_data(X2_data)
    K=[]
    for i, L1 in enumerate(X1_data):
        for j, L2 in enumerate(X2_data):
              K.append(get_single_tensorial_kernel(L1, L2, rank, gamma))
    K = np.asarray(K)
    return K.reshape((len(X1_data), len(X2_data)))
    
def cross_valid(Xtr, Xtst, ytr, ytst, TT_ranks):
    
    tuned_parameter_C = [2**k for k in range(-1,1)]
    tuned_parameter_gamma = [2**k for k in range(-1,1)]
    
    acc =[]
    c_op=[]
    g_op =[]
    
    for g in tuned_parameter_gamma:
        t0 = time.time()
        print("Compute K_train")
        K_train = kernel_mat(Xtr, Xtr, TT_ranks, gamma=g)
        t1 = time.time() - t0
        print(f"K_train computed in {t1}seconds")
        for c in tuned_parameter_C:
            t2 = time.time()
            clf = svm.SVC(kernel='precomputed', C=c, gamma=g)
            cv_results  = cross_val_score(clf, K_train, ytr, cv=2)
            t3 = time.time()-t2
            acc.append(np.mean(cv_results))
   
    acc = np.asarray(acc)
    print("acc val scores", acc)
    acc = acc.reshape((len(tuned_parameter_C), len(tuned_parameter_C)))
    indices = np.where(acc== acc.max())
    L = list(zip(indices[0], indices[1]))
    
    scores =[]    
    
    print(" choosinf the right params")
    for (ind_c, ind_g) in L:
        t00= time.time()
        c_op= tuned_parameter_C[ind_c]
        g_op = tuned_parameter_gamma[ind_g]
        
        clf = svm.SVC(kernel='precomputed', C=c_op, gamma=g_op)
        K_train = kernel_mat(Xtr, Xtr, TT_ranks, gamma=g_op)
        clf.fit(K_train, ytr)
        K_test = kernel_mat(Xtst, Xtr, TT_ranks, gamma=g_op)

        y_pred = clf.predict(K_test)
        scores.append(accuracy_score(ytst, y_pred))
    
        t11 = time.time()-t00
    print("The choice of c_opt et g_opt is done")
    
    return max(scores), t1+t3, t11, len(L)

def cross_valid1(Xtr, Xtst, ytr, ytst, TT_ranks):
    
    tuned_parameter_C = [2**k for k in range(-4,4)]
    tuned_parameter_gamma = [2**k for k in range(-4,4)]
    
    acc =[]
    scores = []
    
    for g in tuned_parameter_gamma:
        t0 = time.time()
        print("Compute K_train")
        K_train = kernel_mat(Xtr, Xtr, TT_ranks, gamma=g)
        t1 = time.time() - t0
        print(f"K_train computed in {t1}seconds")
        for c in tuned_parameter_C:
            t2 = time.time()
            clf = svm.SVC(kernel='precomputed', C=c, gamma=g)
            clf.fit(K_train, ytr)
            cv_results  = cross_val_score(clf, K_train, ytr, cv=2)
            t3 = time.time()-t2
            print("time of cross val", t3)
            acc.append(np.mean(cv_results))
            print("acc",acc)
            K_test = kernel_mat(Xtst, Xtr, TT_ranks, gamma=g)
            y_pred = clf.predict(K_test)
            scores.append(accuracy_score(ytst, y_pred))
            print(scores)
    
    return max(scores), t1, t3
    
def get_scores(list_data, list_labels, l, TT_ranks):#l: longueur du training set
    
    Xo, y = shuffle(list_data, list_labels)
    M = len(Xo)
    m = int(l*M)
    #Q = len(list_CP_h[0])
    X_train,   X_test   = Xo[:m],  Xo[m:]

    y_train,   y_test   = y[:m],  y[m:]      
    score, t1, t11 = cross_valid1(X_train, X_test, y_train, y_test, TT_ranks)

    return score, t1, t11