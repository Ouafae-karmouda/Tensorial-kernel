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
import tensorly as tl
from MPS_rand import  matrix_product_state, matrix_product_state_transposed

chemin = ("C:/Users/adminlocal/Desktop/Code_n")
sys.path.append(chemin)
#Pour chaque donnée, construire sa liste de projecteurs
#rectifier la fct noyeu pour 2 tensuers à oartie de deux  listes de projexteurs correspondant à deux données
#utiliset les listes de projecteurs pour le calcul de la matrice noyau
    
def test(G):
    
    return np.linalg.norm(tlinalg.t_product(projector(G), tlinalg.t_transpose(G)) - tlinalg.t_transpose(G))

def pseudo_inv(G):# pseudo_inv of G
    temp = tlinalg.t_inv(tlinalg.t_product(tlinalg.t_transpose(G), G) )
    
    return tlinalg.t_product(temp, tlinalg.t_transpose(G))

def pseudo_inv_t(G_t): #pseudo_inv of transpose of G
    G = tlinalg.t_transpose(G_t)
    temp = tlinalg.t_inv( tlinalg.t_product(G, G_t))
    
    return tlinalg.t_product(temp, G)
    

def projector1(G):#G is a tensor of order 3
    
    return tlinalg.t_product(pseudo_inv(G), G)
    
    
def projector(G):#G is a tensor of order 3
    G_t = tlinalg.t_transpose(G)
    pinv_t = pseudo_inv(G_t)
    
    return tlinalg.t_product(G_t, pinv_t)

def projector_matrix1(M):#M is a matrix
    return np.dot(M, np.linalg.pinv(M))

def projector_matrix(A):
    r = np.linalg.inv(np.dot(A,A.T))
    return np.dot(A.T,np.dot(r,A))

#calcul des projecteurs pour les données
def construct_list_projectors(factors_X):#factors_X: liste de coeurs d'une donnée
    Q = len(factors_X)
    M = np.squeeze(factors_X[0])
    list_projectors = [projector_matrix(M)]
    
    for q in range(1, Q-1):
        list_projectors.append(projector(factors_X[q]))
        
    M = np.squeeze(factors_X[-1])
    list_projectors.append(projector_matrix(M.T))
    return list_projectors
      
def get_precomputed_kernel(list_proj_X, list_proj_Y):
    Q = len(list_proj_X)
    
    M1 = list_proj_X[0]
    M2 = list_proj_Y[0]
    m1 = np.linalg.norm(M1)
    m2 = np.linalg.norm(M2)
    
    M11 = list_proj_X[-1]
    m11 = np.linalg.norm(M11)
    M22 = list_proj_Y[-1]
    m22 = np.linalg.norm(M22)
    
    s = (np.linalg.norm(M1/m1 - M2/m2, 'fro'))**2
    s = s + (np.linalg.norm(M11/m11- M22/m22, 'fro'))**2

    
    for q in range(1, Q-1):
        n1 = np.linalg.norm(list_proj_X[q])
        n2 = np.linalg.norm(list_proj_Y[q])
        s = s+(np.linalg.norm(list_proj_X[q]/n1 - list_proj_Y[q]/n2))**2
    return s

def get_precomputed_kernel_opt(list_proj_X, list_proj_Y):
    Q = len(list_proj_X)
    
    M1 = list_proj_X[0]
    M2 = list_proj_Y[0]
 
    
    M11 = list_proj_X[-1] 
    M22 = list_proj_Y[-1]
    
    s = (np.linalg.norm(M1 - M2, 'fro'))**2
    s = s + (np.linalg.norm(M11- M22, 'fro'))**2

    
    for q in range(1, Q-1):
        s = s+(np.linalg.norm(list_proj_X[q]- list_proj_Y[q]))**2
    return s
   
def precomputed_kernel_matrix_opt(X1_data, X2_data):#X1_data=[list_proj(X0),..., list_proj(Xn)]
    K=[]
    for i, L1 in enumerate(X1_data):
        for j, L2 in enumerate(X2_data):
                K.append(get_precomputed_kernel_opt(L1, L2))
    K = np.asarray(K)
    return K.reshape((len(X1_data), len(X2_data)))

def precomputed_kernel_matrix(X1_data, X2_data):#X1_data=[list_proj(X0),..., list_proj(Xn)]
    K=[]
    for i, L1 in enumerate(X1_data):
        for j, L2 in enumerate(X2_data):
                K.append(get_precomputed_kernel(L1, L2))
    K = np.asarray(K)
    return K.reshape((len(X1_data), len(X2_data)))

def precomputed_kernel_matrix_bats(X1_data, X2_data, gamma):#X1_data=[list_proj(X0),..., list_proj(Xn)]
    K=[]
    for i, L1 in enumerate(X1_data):
        for j, L2 in enumerate(X2_data):
                K.append(single_kernel_bats(L1, L2, gamma))
    
    K = np.asarray(K)
    return K.reshape((len(X1_data), len(X2_data)))

def single_kernel_bats(TT_cores_X, TT_cores_Y, gamma):
    
    [G1, G2, G3, G4] = TT_cores_X
    [G11, G22, G33, G44] = TT_cores_Y
    [_, R1, R2, R3, _]  = [1, TT_cores_X[0].shape[-1],  TT_cores_X[1].shape[-1], TT_cores_X[2].shape[-1],1  ]
    
    s=0 
    for r1 in range(R1):
        for r2 in range(R2):
            for r3 in range(R3):
                temp1 = np.exp(-gamma*(np.linalg.norm(G1[:,r1]/np.linalg.norm(G1[:,r1]) -G11[:,r1]/np.linalg.norm(G11[:,r1])))**2)
                temp2 = np.exp(-gamma*(np.linalg.norm(G2[r1,:,r2]/np.linalg.norm(G2[r1,:,r2]) -G22[r1,:,r2]/np.linalg.norm(G22[r1,:,r2]) ))**2)
                temp3 = np.exp(-gamma*(np.linalg.norm(G3[r2,:,r3]/np.linalg.norm(G3[r2,:,r3]) -G33[r2,:,r3]/np.linalg.norm(G33[r2,:,r3]) ))**2)
                temp4 = np.exp(-gamma*(np.linalg.norm(G4[r3,:]/np.linalg.norm(G4[r3,:]) -G44[r3,:]/np.linalg.norm(G44[r3,:])))**2)
                s = s+ temp1 + temp2 + temp3 + temp4
                
    return s
                
    



    
def normalize_list_projectors(list_projectors):#list_projectors: The nth element is a list of projectors of the nth data
    
    list_proj_per_data = [] #the qth elem is a liste of projectors of all data for the qth factors
    Q = len(list_projectors[0])
    N = len(list_projectors)
    for q in range(Q):
        
        L = [list_projectors[n][q] for n in range(N) ]
        arr1 = np.asarray(L)
        #print(arr1.shape)
        arr = arr1.reshape(arr1.shape[0],-1)
        
        scaler = StandardScaler()
        scaler.fit(arr)
        arr = scaler.transform(arr)
        arr = arr.reshape(arr1.shape)
        list_proj_per_data.append(list(arr))
        
    list_projectors_norm = [[list_proj_per_data[q][n] for q in range(Q)] for n in range(N) ]
    
    return list_projectors_norm

    
    
def normaliser(list_projectors):
    
    N = len(list_projectors)
    Q = len(list_projectors[0])
    
    for n in range(N):
        list_projectors[n][0] = (list_projectors[n][0])/np.linalg.norm(list_projectors[n][0], 'fro')
        list_projectors[n][-1] = list_projectors[n][-1]/np.linalg.norm(list_projectors[n][-1], 'fro')
        
        for q in range(1, Q-1):
            list_projectors[n][q] = list_projectors[n][q]/np.linalg.norm(list_projectors[n][q])
    return list_projectors
        
#### Projectors by left sigular bases   
def get_base_single(list_cores):#list_cores of one single data
    Q = len(list_cores)
    list_base = []
    
    I, R = list_cores[0].shape
    U, S, Vh = tl.partial_svd(list_cores[0], R)
    list_base.append(U)
    
    for q in range(1, Q-1):
        U, S, V = tlinalg.t_svd(list_cores[q], opt ='econ')
        list_base.append(U)
    
    R, I = list_cores[-1].shape
    U, S, Vh = np.linalg.svd(list_cores[-1], R)
    list_base.append(U)
    
    return list_base

def get_proj_single(list_base):# list_base:liste base singuliè-res gauches of one data
    
    Q = len(list_base)
    U = list_base[0]
    list_proj = [np.dot(U, U.T)]
    
    for q in range(1, Q-1):
        U = list_base[q]
        U_t = tlinalg.t_transpose(U)
        list_proj.append(tlinalg.t_product(U, U_t))
    U =list_base[-1]
    list_proj.append(np.dot(U, U.T))
    
    return list_proj

def get_base_data(list_cores):#list of list of cores
    
    list_base = []
    N = len(list_cores)
    for n in range(N):
        list_sing = get_base_single(list_cores[n])
        list_base.append(list_sing)
        
    return list_base
        

def get_proj_data(list_cores):
    list_proj = []
    N = len(list_cores)
    for n in range(N):
        list_base = get_base_single(list_cores[n])
        list_proj_sing = get_proj_single(list_base)
        list_proj.append(list_proj_sing)
    return list_proj

###Projectors by unfolding cores
    
def unfold(a):
    return np.vstack([a[:,:,k] for k in range(a.shape[2])])

def get_proj_data_unfold(list_cores):

    list_proj=[]
    N = len(list_cores)
    Q = len(list_cores[0])
    for n in range(N):
        U = list_cores[n][0]
        list_proj_single =[projector_matrix(U)]
        for q in range(1, Q-1):
            G = list_cores[n][q]
            list_proj_single.append(projector_matrix(unfold(G)))
        U = list_cores[n][-1]
        list_proj_single.append(projector_matrix(U.T))
        list_proj.append(list_proj_single)
    return list_proj
            
        

def test_proj(G, U):#test if the projectots constructed by singular bases are the same as projectors usong TT core
    
    return np.linalg.norm( tlinalg.t_product(G, pseudo_inv(G)) - tlinalg.t_product(U,  tlinalg.t_transpose(U) ) )

def test_proj_matrix(G, U):
    
    return np.linalg.norm(np.dot(G, np.linalg.pinv(G) ) - np.dot(U, np.linalg.pinv(U)))
    

def ambig_TT(u,a,v):
    a1 = np.zeros(a.shape)
    for k in range(a.shape[1]):
        a1[:,k,:] = np.dot(u,np.dot(a[:,k,:],v))
    return a1


def test_tsvd_amb():
    r1 = 3
    r2 = 2
    I = 10
    
    G = np.random.randn(r1,I,r2)
    M1 = np.random.randn(G.shape[0],G.shape[0])
    M2 = np.random.randn(G.shape[2],G.shape[2])
    
    G_tilde = ambig_TT(M1,G,M2)
    U,_,_ = tlinalg.t_svd(G, opt ='econ')
    Utilde,_,_ = tlinalg.t_svd(G_tilde, opt ='econ')
    
    UM1 = ambig_TT(M1,U,np.eye(G.shape[2],G.shape[2]))
    
    return U, Utilde, UM1
    

def test_projectors(X, TT_ranks):
    
    TT_cores = matrix_product_state(X, TT_ranks, verbose=False, type_svd = 'partial_svd')
    TT_cores1 = matrix_product_state(X, TT_ranks, verbose=False, type_svd = 'partial_svd')
    
    G, G1 = TT_cores[1], TT_cores1[1]
    U,_,_ = tlinalg.t_svd(G, opt ='econ')
    U1,_,_ = tlinalg.t_svd(G1, opt ='econ')
    
    Ut,_,_ = tlinalg.t_svd(tlinalg.t_transpose(G), opt ='econ')
    U1t,_,_ = tlinalg.t_svd(tlinalg.t_transpose(G1), opt ='econ')
    
    print("Projectors singular bases without transpose ",np.linalg.norm(projector(U)- projector(U1)))
    print("Projectors without  transpose ",np.linalg.norm(projector(G)- projector(G1)))
    
  
    
    
    
    




