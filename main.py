# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 10:49:51 2021

@author: adminlocal
"""
import sys
from Construct_TT_cores import  get_TT_svd, Complex_TT_svd
from sklearn.utils import shuffle

chemin = ("C:/Users/adminlocal/Desktop/Code_n")
sys.path.append(chemin)

import numpy as np
import os
from kernel_optimized import construct_list_projectors, normaliser, normalize_list_projectors, precomputed_kernel_matrix, cross_valid_proj
import tqdm
import pickle
import time
from svm_hosvd import Complex_hosvd



def get_projectors(name_file, TT_ranks):

        with open(name_file, 'rb') as pickle_file:
            content = pickle.load(pickle_file)
        
        list_data = content[0]
        list_labels = content[1]
        
        list_data, list_labels = shuffle(list_data, list_labels)
        #TT_ranks = [1,2,4,2,1]
        N_video = len(list_data)
        list_TT_cores, t1 =  compute_TT_cores(list_data, TT_ranks)
        list_projectors = []
        t0 = time.time()
        for  n in range(N_video):
            #construit le projecteur pour la video n
            proj = construct_list_projectors(list_TT_cores[n])
            #ajoute le projecteur et le label dans leurs listes respectives
            list_projectors.append(proj)
        t1 = time.time() - t0
        print(f"time construct projectors {t1}")
        t0 = time.time()
        list_projectors= normalize_list_projectors(list_projectors)
        t1 =time.time() -t0
        print(f"standardization projectors {t1}")
        return list_projectors, list_labels


def compute_TT_cores(list_data,TT_ranks):
    print("Get TT-svd")
    t0 = time.time()
    list_TT_cores =  get_TT_svd(list_data, TT_ranks)
    t1 = time.time()-t0
    print(f"time TT-svd {t1}")
    return list_TT_cores, t1


def get_scores(list_projectors, list_labels, kernel='subs_tens'):#valid for cores or projectors
    t0 = time.time()
    list_projectors = normalize_list_projectors(list_projectors)
    t1= time.time() - t0
    t0 = time.time()
    J_train =  precomputed_kernel_matrix(list_projectors, list_projectors, kernel = kernel)
    t1 = time.time()-t0
    print(f"time kernel matrix {t1}")
    
    Nrep = 10
    scores =  np.zeros(Nrep)
    std =  np.zeros(Nrep)
    
    for nrep in tqdm.tqdm(range(Nrep)):
        print("Début svms")
        sc, stand = cross_valid_proj(J_train, list_labels, kernel = kernel)
        std[nrep] = stand
        scores[nrep] = sc
    return np.mean(scores), np.mean(std)

def read_dataset_extended(file):
    data = pickle.load(file)
    file.close()

    T_c0 = list(data[0].reshape(16,9,480,640,4))
    T_c1 = list(data[1].reshape(16,9,480,640,4))
    T_c2 = list(data[2].reshape(16,9,480,640,4))
    
    data = T_c0 + T_c1 + T_c2
    labels = 16*[0] + 16*[1] + 16*[2]
    
    return data, labels

"""
print("kernel projectors videos")
TT_ranks = [1,2,2,3,1]
file_videos = 'videos_processed'
list_projectors, list_labels = get_projectors(file_videos, TT_ranks)
t0 = time.time()
list_projectors = normaliser(list_projectors)
t1 = time.time() - t0
print(f"time normaliser {t1}")
score_proj, std_proj = get_scores(list_projectors, list_labels)
print(f"Score for kernel with projectors {score_proj} {std_proj}")


print("Batselier kernel videos")
#file_videos = 'videos_processed'
#list_projectors, list_labels = get_projectors(file_videos, TT_ranks)

with open('videos_processed', 'rb') as pickle_file:
            content = pickle.load(pickle_file)
        
list_data = content[0]
list_labels = content[1]
list_cores, t1 = Complex_TT_svd(list_data, TT_ranks)
score, std = get_scores(list_cores, list_labels, kernel='bats_kernel')
print(f"Score for kernel{score} {std}")
"""
print("kernel projectors Extended")
TT_ranks = [1,3,3,3,1]
#file_extended = 'C:/Users/adminlocal/Desktop/Code_n/Tensors_Extended_Yale_3lasses'
with open('extended_processed', 'rb') as pickle_file:
            content = pickle.load(pickle_file)
        
list_data = content[0]
list_labels = content[1]
list_data, list_labels = shuffle(list_data, list_labels)

list_projectors, list_labels = get_projectors('extended_processed', TT_ranks)
score, std = get_scores(list_projectors, list_labels)
print(f"Score for kernel with projectors{score} {std}")


list_cores, t1 = Complex_TT_svd(list_data, TT_ranks)
score, std = get_scores(list_cores, list_labels, kernel='bats_kernel')
print(f"Score for kernel with Bats {score} {std}")





































"""    
# open a file, where you ant to store the data
file = open('videos_processed', 'wb')# list [list_data, labels]
data=[None]*2
data[0] = list_data
data[1] = labels
# dump information to that file
pickle.dump(data, file)
pickle.dump(labels , file)
# close the file
file.close()
"""

"""
complexity_TT_svd = []
complexity_hosvd = []
with open('videos_processed', 'rb') as pickle_file:
    content = pickle.load(pickle_file)
    
list_data = content[0]
list_labels = content[1]
    
TT_ranks = [1,2,4,2,1]
Nrep = 10
m_ranks = [2,2,2,2]
for n in range(Nrep):
    t0 = time.time()
    list_TT_cores, t1 =  Complex_TT_svd(list_data, TT_ranks)
    complexity_TT_svd.append(t1)
    
    list_ho, t1= Complex_hosvd(list_data, m_ranks)
    complexity_hosvd.append(t1)
"""


"""
with open(file_extended, 'rb') as pickle_file:
       list_data, list_labels = read_dataset_extended(pickle_file)
       
data=[None]*2
data[0] = list_data
data[1] = list_labels
      
with open('extended_processed','wb') as proj_file:
        pickle.dump(data , proj_file)
"""