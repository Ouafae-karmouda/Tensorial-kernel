# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 15:41:01 2021

@author: adminlocal
"""
#read file of videos
#compute TT-svd and save it
import sys
chemin = ("C:/Users/adminlocal/Desktop/Code_n")
sys.path.append(chemin)
import pickle
from MPS_rand import  matrix_product_state
import numpy as np
import time



with open("C:/Users/adminlocal/Desktop/Code_n/Video_jumping_walking", 'rb') as pickle_file:
    content = pickle.load(pickle_file)
    
#s'assurer que la 2e dim = 240, 3e dim = 320
def process(list_tensor):
    list_T=[]
    for T in list_tensor:
        if (T.shape[1]== 240 and T.shape[2] == 320):
            list_T.append(T)
            
    return list_T

#unifier la première dimension des tenseurs
def min_dim1(list_tensor):
    lengths = []
    for T in list_tensor:
        lengths.append(T.shape[0])
        
    return np.min(lengths)

def unif_dim1(list_tensor, dim1):
    list_T=[]
    for T in list_tensor:
        T = T[:dim1,:]
        list_T.append(T)
        
    return list_T


def process_data_videos(file): # file: nom du fichier à lire, N_T_c: Nbr de données par classe
    """
    return (list_data, list_labels)video tensors and labels from file for example ''Video_jumping_walking'
    list_data = classe 0+ classe1
    """
    file = open(file, 'rb')
    data = pickle.load(file)
    T_c0 = data[0]
    T_c1 = data[1]
    
    file.close()
       
    list_T_c0 =process(T_c0)
    list_T_c1 =process(T_c1)
    
    
    N0, N1 = min_dim1(list_T_c0), min_dim1(list_T_c1)
    dim1 = min(N0, N1)
    
    list_T_c0, list_T_c1 = unif_dim1(list_T_c0, dim1), unif_dim1(list_T_c1, dim1)
    N_T_c0, N_T_c1 = len(list_T_c0), len(list_T_c1)
    N_T_c = min(N_T_c0, N_T_c1)
    
    
    list_data = list_T_c0[:N_T_c] + list_T_c1[:N_T_c]
    
    
    labels0 = [0]*N_T_c
    labels1 = [1]*N_T_c
    list_labels = labels0+ labels1
    print("N_T_c",N_T_c)
    
    return list_data, list_labels



def get_TT_svd(list_T, TT_ranks): # Returns a list of CP factors
    return [  matrix_product_state(T.astype('f'), TT_ranks, verbose=False, type_svd = 'partial_svd') for T in list_T]

def Complex_TT_svd(list_data, TT_ranks):
    #get hosvd factors list_CP_j, time_via_j
    t0 = time.time()
    list_cores = get_TT_svd(list_data,TT_ranks)
    t1 = time.time() - t0
    
    return list_cores, t1
if __name__ == 'main':
    
    list_data, list_labels = process_data_videos("C:/Users/adminlocal/Desktop/Code_n/Video_jumping_walking")
    TT_ranks = [1,2,2,2,1]
    list_data_TT_cores =  get_TT_svd(list_data, TT_ranks)

"""
# open a file, where you ant to store the data
file = open('TT_factors', 'wb')# list [list_data_TT_cores, list_labels, TT_ranks]
data=[None]*3
data[0] = list_data_TT_cores
data[1] = list_labels
data[2] = TT_ranks
# dump information to that file
pickle.dump(data, file)
# dump information to that file
pickle.dump(list_data_TT_cores , file)
# close the file
file.close()
"""