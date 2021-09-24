# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 09:00:51 2021

@author: adminlocal
"""
import numpy as np 
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
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
    temp = tlinalg.t_inv(tlinalg.t_product(tlinalg.t_transpose(G), G) )
    
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
    
    s = (np.linalg.norm(M1/m1 - M2/m2, 'fro'))**2
    s = s + (np.linalg.norm(M11/m11- M22/m22, 'fro'))**2

    
    for q in range(1, Q-1):
        n1 = np.linalg.norm(list_proj_X[q])
        n2 = np.linalg.norm(list_proj_Y[q])
        s = s+(np.linalg.norm(list_proj_X[q]/n1 - list_proj_Y[q]/n2))**2
    return s
    
    
def precomputed_kernel_matrix(X1_data, X2_data, kernel='subs_tens'):#X1_data=[list_proj(X0),..., list_proj(Xn)]
    K=[]
    for i, L1 in enumerate(X1_data):
        for j, L2 in enumerate(X2_data):
            if kernel == 'bats_kernel':
                K.append(single_kernel_bat(L1, L2))
            else:
                K.append(get_precomputed_kernel(L1, L2))
    K = np.asarray(K)
    return K.reshape((len(X1_data), len(X2_data)))

def single_kernel_bat(TT_cores_X, TT_cores_Y):
    
    [G1, G2, G3, G4] = TT_cores_X
    [G11, G22, G33, G44] = TT_cores_Y
    [_, R1, R2, R3, _]  = [1, TT_cores_X[0].shape[-1],  TT_cores_X[1].shape[-1], TT_cores_X[2].shape[-1],1  ]
    
    s=0 
    for r1 in range(R1):
        for r2 in range(R2):
            for r3 in range(R3):
                temp1 = np.exp(-(np.linalg.norm(G1[:,r1]/np.linalg.norm(G1[:,r1]) -G11[:,r1]/np.linalg.norm(G11[:,r1])))**2)
                temp2 = np.exp(-(np.linalg.norm(G2[r1,:,r2]/np.linalg.norm(G2[r1,:,r2]) -G22[r1,:,r2]/np.linalg.norm(G22[r1,:,r2]) ))**2)
                temp3 = np.exp(-(np.linalg.norm(G3[r2,:,r3]/np.linalg.norm(G3[r2,:,r3]) -G33[r2,:,r3]/np.linalg.norm(G33[r2,:,r3]) ))**2)
                temp4 = np.exp(-(np.linalg.norm(G4[r3,:]/np.linalg.norm(G4[r3,:]) -G44[r3,:]/np.linalg.norm(G44[r3,:])))**2)
                s = s+ temp1 + temp2 + temp3 + temp4
                
    return s
                
    

def cross_valid_proj(Jtrain, ytr,  kernel='subs_tens'):#Jtrain : Pre-computed matrix of training
    
    tuned_parameter_C = [2**k for k in range(-9,9)]
    tuned_parameter_gamma = [2**k for k in range(-9,9)]
    
    acc = 0
    stand = 0
    
    for g in tuned_parameter_gamma:
        if kernel == 'bats_kernel':
            K_train = Jtrain
        else:
            K_train = np.exp(-g*Jtrain)
        for c in tuned_parameter_C:
            clf = svm.SVC(kernel='precomputed', C=c, gamma=g)
            clf.fit(K_train, ytr)
            cv_results  = cross_val_score(clf, K_train, ytr, cv=2)
            if np.mean(cv_results) > acc:
                acc = np.mean(cv_results) 
                stand = np.std(cv_results)
            
    return acc, stand
    


    
def normalize_list_projectors(list_projectors):#list_projectors: The nth element is a list of projectors of the nth data
    
    list_proj_per_data = [] #the qth elem is a liste of projectors of all data for the qth factors
    Q = len(list_projectors[0])
    N = len(list_projectors)
    for q in range(Q):
        L = [list_projectors[n][q] for n in range(N) ]
        arr1 = np.asarray(L)
        print(arr1.shape)
        arr = arr1.reshape(arr1.shape[0],-1)
        
        scaler = StandardScaler()
        scaler.fit(arr)
        arr = scaler.transform(arr)
        arr = arr.reshape(arr1.shape)
        list_proj_per_data.append(list(arr))
        
    list_projectors_norm = [[list_proj_per_data[q][n] for q in range(Q)] for n in range(N) ]
    
    return list_projectors_norm

    
    