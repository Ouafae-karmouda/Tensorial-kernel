# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 09:00:51 2021

@author: adminlocal
"""
import numpy as np 
from sklearn.model_selection import cross_val_score
from sklearn import svm
import tlinalg
import sys 
import time

chemin = ("C:/Users/adminlocal/Desktop/Code_n")
sys.path.append(chemin)
#Pour chaque donnée, construire sa liste de projecteurs
#rectifier la fct noyeu pour 2 tensuers à oartie de deux  listes de projexteurs correspondant à deux données
#utiliset les listes de projecteurs pour le calcul de la matrice noyau
    

def pseudo_inv(G):# pseudo_inv of G
    temp = tlinalg.t_inv(  tlinalg.t_product(tlinalg.t_transpose(G), G) )
    
    return tlinalg.t_product(temp, tlinalg.t_transpose(G))

def projector1(G):#G is a tensor of order 3
    
    return tlinalg.t_product(G, pseudo_inv(G))
    
    
def projector(G):#G is a tensor of order 3
    G_t = tlinalg.t_transpose(G)
    pinv_t = pseudo_inv(G_t)
    
    return tlinalg.t_product(G_t, pinv_t)

def projector_matrix(M):#M is a matrix
    return np.dot(M, np.linalg.pinv(M))

#calcul des projecteurs pour les données
def construct_list_projectors(factors_X):#factors_X: liste de coeurs d'une donnée
    Q = len(factors_X)
    M = np.squeeze(factors_X[0])
    list_projectors = [projector_matrix(M)]
    
    for q in range(1, Q-1):
        list_projectors.append(projector(factors_X[q]))
        
    M = np.squeeze(factors_X[-1])
    list_projectors.append(projector_matrix(M))
    return list_projectors
        
def get_precomputed_kernel(list_proj_X, list_proj_Y):
    Q = len(list_proj_X)
    
    M1 = np.squeeze(list_proj_X[0])
    m1 = np.linalg.norm(M1)
    M2 = np.squeeze(list_proj_Y[0])
    m2 = np.linalg.norm(M2)
    
    M11 = np.squeeze(list_proj_X[-1])
    m11 = np.linalg.norm(M11)
    M22 = np.squeeze(list_proj_Y[-1])
    m22 = np.linalg.norm(M22)
    
    prod = (np.linalg.norm(M1/m1 - M2/m2, 'fro'))**2
    prod = prod * (np.linalg.norm(M11/m11 - M22/m22, 'fro'))**2

    
    for q in range(1, Q-1):
        n1 = np.linalg.norm(list_proj_X[q])
        n2 = np.linalg.norm(list_proj_Y[q])
        prod = prod*(np.linalg.norm(list_proj_X[q]/n1 - list_proj_Y[q]/n2))**2
    return prod
    
    
def precomputed_kernel_matrix(X1_data, X2_data):#X1_data=[list_proj(X0),..., list_proj(Xn)]
    K=[]
    for i, L1 in enumerate(X1_data):
        for j, L2 in enumerate(X2_data):
              K.append(get_precomputed_kernel(L1, L2))
    K = np.asarray(K)
    return K.reshape((len(X1_data), len(X2_data)))

def cross_valid_proj(Jtrain, ytr):#Jtrain : Pre-computed matrix of training
    
    tuned_parameter_C = [2**k for k in range(-9,9)]
    tuned_parameter_gamma = [2**k for k in range(-9,9)]
    
    acc =[]
    
    for g in tuned_parameter_gamma:
        t0 = time.time()
        K_train = np.exp(-g*Jtrain)
        t1 = time.time() - t0
        for c in tuned_parameter_C:
            t2 = time.time()
            clf = svm.SVC(kernel='precomputed', C=c, gamma=g)
            clf.fit(K_train, ytr)
            cv_results  = cross_val_score(clf, K_train, ytr, cv=2)
            t3 = time.time()-t2
            acc.append(np.mean(cv_results))
    
    return max(acc), t1, t3
    

def get_scores(list_data, list_labels, l):#l: longueur du training set, 
                                                    #list_data =[list_proj(X0),..., list_proj(Xn)]

    J_train =  precomputed_kernel_matrix(list_data, list_data)
    score, t1, t11 = cross_valid_proj(J_train, list_labels)

    return score, t1, t11
    