# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 10:49:51 2021

@author: adminlocal
"""
import sys
import numpy as np
import os
from sklearn.utils import shuffle

import tlinalg
from Construct_TT_cores import  compute_TT_cores_transposed, compute_TT_cores
chemin = ("C:/Users/adminlocal/Desktop/Code_n")
sys.path.append(chemin)


from kernel_optimized import  construct_list_projectors, get_proj_single, get_proj_data_unfold, get_base_single, get_base_data, get_proj_data, test_proj, test_proj_matrix, normaliser
from kernel_optimized import precomputed_kernel_matrix_opt,  normalize_list_projectors, precomputed_kernel_matrix, precomputed_kernel_matrix_bats
import tqdm
import pickle
import time
from sklearn import svm
from sklearn.model_selection import cross_val_score



def bruiter_data(list_data, sigma):
    N = len(list_data)
    
    list_data_bruit= []
    for n in range(N):
        X = list_data[n].astype('f')    
        B = np.random.randn(X.shape[0], X.shape[1], X.shape[2], X.shape[3])
        norm_X = np.linalg.norm(X)
        norm_B = np.linalg.norm(B)
        Y= (X /norm_X) + sigma *( B / norm_B)
        list_data_bruit.append(Y)
    return list_data_bruit
    

def get_projectors(name_file, TT_ranks, bruit = 'False'):

        with open(name_file, 'rb') as pickle_file:
            content = pickle.load(pickle_file)
        
        list_data = content[0]
        list_labels = content[1]
        list_data, list_labels = shuffle(list_data, list_labels)
        
        if bruit == 'True':
            sigma = 0.03
            list_data = bruiter_data(list_data, sigma)
            
        N_video = len(list_data)
        list_TT_cores, time_TT =  compute_TT_cores(list_data, TT_ranks)
        list_projectors = []
        t0 = time.time()
        for  n in range(N_video):
            #construit le projecteur pour la video n
            proj = construct_list_projectors(list_TT_cores[n])
            #ajoute le projecteur et le label dans leurs listes respectives
            list_projectors.append(proj)
        time_proj = time.time() - t0
        
        list_projectors= normalize_list_projectors(list_projectors)
        
        return list_projectors, list_TT_cores, list_labels, time_TT, time_proj
    
def get_projectors_ameliorated(name_file, TT_ranks):
    
        with open(name_file, 'rb') as pickle_file:
            content = pickle.load(pickle_file)
        
        list_data = content[0]
        list_labels = content[1]
        
        list_data, list_labels = shuffle(list_data, list_labels)
        list_TT_cores, time_TT =  compute_TT_cores_transposed(list_data, TT_ranks)
        t0 = time.time()
        list_projectors = get_proj_data(list_TT_cores)
        time_proj = time.time()-t0
        
        return list_projectors, list_labels, time_TT, time_proj


def get_projectors_unfolded(name_file, TT_ranks):
    
        with open(name_file, 'rb') as pickle_file:
            content = pickle.load(pickle_file)
        
        list_data = content[0]
        list_labels = content[1]
        
        list_data, list_labels = shuffle(list_data, list_labels)
        list_TT_cores, time_TT =  compute_TT_cores(list_data, TT_ranks)
        t0 = time.time()
        list_projectors = get_proj_data_unfold(list_TT_cores)
        time_proj = time.time()-t0
        
        return list_projectors, list_labels, time_TT, time_proj
    

def get_scores(J_train, list_labels):#valid for cores or projectors

        
    tuned_parameter_C = [2**k for k in range(-9,9)]
    tuned_parameter_gamma = [2**k for k in range(-9,9)]
    
    Nrep = 1
    list_scores =  []
    list_std = []
    
    for g in tuned_parameter_gamma:
        K_train = np.exp(-g*J_train)
        for c in tuned_parameter_C:
            clf = svm.SVC(kernel='precomputed', C=c, gamma=g)
            clf.fit(K_train, list_labels)
            acc = []
            std = []
            for nrep in (range(Nrep)):
                cv_results  = cross_val_score(clf, K_train, list_labels, cv=2)
                acc.append(np.mean(cv_results) )
                std.append( np.std(cv_results))
            list_scores.append(np.mean(acc))
            list_std.append(np.mean(std))
                
    return np.max(list_scores), np.mean(list_std)


def get_scores_bats(list_cores, list_labels):#valid for cores or projectors

    tuned_parameter_C = [2**k for k in range(-9,9)]
    tuned_parameter_gamma = [2**k for k in range(-9,9)]
    

    Nrep = 1
    list_scores = []
    list_std =  []
    
    
    for g in tuned_parameter_gamma:
        K_train =  precomputed_kernel_matrix_bats(list_cores, list_cores, g)

        for c in tuned_parameter_C:
            clf = svm.SVC(kernel='precomputed', C=c, gamma=g)
            clf.fit(K_train, list_labels)
            
            acc = []
            std = []
            for nrep in (range(Nrep)):
                cv_results  = cross_val_score(clf, K_train, list_labels, cv=2)
                acc.append(np.mean(cv_results) )
                std.append( np.std(cv_results))
            list_scores.append(np.mean(acc))
            list_std.append(np.mean(std))
            
        
    return np.max(list_scores), np.std(list_std)
    

def read_dataset_extended(file):
    data = pickle.load(file)
    file.close()

    T_c0 = list(data[0].reshape(16,9,480,640,4))
    T_c1 = list(data[1].reshape(16,9,480,640,4))
    T_c2 = list(data[2].reshape(16,9,480,640,4))
    
    data = T_c0 + T_c1 + T_c2
    labels = 16*[0] + 16*[1] + 16*[2]
    
    return data, labels

list_scores_proj = []
list_scores_bats = []
list_time_TT = []
list_time_K_proj = []
list_time_K_bats =[]
list_time_const_proj = []
TT_ranks = [1,2,2,2,1]

"""
file_videos = 'videos_processed'
#repeat 10 times witeh recomputing TT-cores for UCF11

for n in tqdm.tqdm(range(10)):

    print("kernel projectors videos")
    list_projectors, list_labels, time_TT, time_proj= get_projectors_ameliorated(file_videos, TT_ranks)
    #list_projectors, list_TT_cores, list_labels, time_TT, time_proj= get_projectors(file_videos, TT_ranks, bruit = 'True')
    list_time_TT.append(time_TT)
    list_time_const_proj.append(time_proj)
    J_train =  precomputed_kernel_matrix(list_projectors, list_projectors)
    t0 = time.time()
    #J_train_opt =  precomputed_kernel_matrix_opt(list_projectors, list_projectors)
    time_K_proj = time.time()-t0
    list_time_K_proj.append(time_K_proj)
    score_proj, std_proj = get_scores(J_train, list_labels)
    list_scores_proj.append(score_proj)
    print(f"Score for kernel with projectors {score_proj} {std_proj}")
    print("Batselier kernel videos")
    t0 = time.time()
    #K_train =  precomputed_kernel_matrix_bats(list_TT_cores, list_TT_cores, 1)
    time_K_train_bats = time.time() - t0
    list_time_K_bats.append(time_K_train_bats)
    #score, std = get_scores_bats(list_TT_cores, list_labels)
    #list_scores_bats.append(score)
    #print(f"Score for kernel{score} {std}")
"""
 

#repeat 10 times witeh recomputing TT-cores for extended
file_extended = 'extended_processed'

for n in tqdm.tqdm(range(10)):

    print("kernel projectors Extended")
    #list_projectors, list_TT_cores, list_labels, time_TT, time_proj= get_projectors(file_extended, TT_ranks)
    #list_projectors, list_labels, time_TT, time_proj= get_projectors_ameliorated(file_extended, TT_ranks)
    list_projectors, list_labels, time_TT, time_proj = get_projectors_unfolded(file_extended, TT_ranks)
    list_time_TT.append(time_TT)
    list_time_const_proj.append(time_proj)
    J_train =  precomputed_kernel_matrix(list_projectors, list_projectors)
    t0 = time.time()
    J_train_opt =  precomputed_kernel_matrix_opt(list_projectors, list_projectors)
    time_K_proj = time.time()-t0
    list_time_K_proj.append(time_K_proj)
    score_proj, std_proj = get_scores(J_train, list_labels)
    list_scores_proj.append(score_proj)
    print(f"Score for kernel with projectors {score_proj} {std_proj}")
    print("Batselier kernel Extended")
    t0 = time.time()
    #K_train =  precomputed_kernel_matrix_bats(list_TT_cores, list_TT_cores, 1)
    #time_K_train_bats = time.time() - t0
    #list_time_K_bats.append(time_K_train_bats)
    #score, std = get_scores_bats(list_TT_cores, list_labels)
    #list_scores_bats.append(score)


"""
#repeat 10 times witeh recomputing TT-cores for extended
file_faces_96 = 'dataset_faces_96'

for n in tqdm.tqdm(range(10)):
    print("kernel projectors Extended")
    #list_projectors, list_TT_cores, list_labels, time_TT, time_proj= get_projectors(file_faces_96, TT_ranks)
    
    list_projectors, list_labels, time_TT, time_proj= get_projectors_ameliorated(file_faces_96, TT_ranks)
    list_time_TT.append(time_TT)
    list_time_const_proj.append(time_proj)
    J_train =  precomputed_kernel_matrix(list_projectors, list_projectors)
    t0 = time.time()
    J_train_opt =  precomputed_kernel_matrix_opt(list_projectors, list_projectors)
    time_K_proj = time.time()-t0
    list_time_K_proj.append(time_K_proj)
    score_proj, std_proj = get_scores(J_train, list_labels)
    list_scores_proj.append(score_proj)
    print(f"Score for kernel with projectors {score_proj} {std_proj}")
    print("Batselier kernel Extended")
    _, list_TT_cores, list_labels, _, _= get_projectors(file_faces_96, TT_ranks)
    t0 = time.time()
    K_train =  precomputed_kernel_matrix_bats(list_TT_cores, list_TT_cores, 1)
    time_K_train_bats = time.time() - t0
    list_time_K_bats.append(time_K_train_bats)
    score, std = get_scores_bats(list_TT_cores, list_labels)
    list_scores_bats.append(score)
    print(f"Score for kernel{score} {std}")
"""


"""
print("kernel projectors Extended")
TT_ranks = [1,2,2,2,1]
#file_extended = 'C:/Users/adminlocal/Desktop/Code_n/Tensors_Extended_Yale_3lasses'
with open('extended_processed', 'rb') as pickle_file:
            content = pickle.load(pickle_file)
        
list_data = content[0]
list_labels = content[1]
list_data, list_labels = shuffle(list_data, list_labels)

list_projectors, list_TT_cores, list_labels, time_TT, time_proj = get_projectors('extended_processed', TT_ranks, bruit = 'True')
J_train =  precomputed_kernel_matrix(list_projectors, list_projectors)
t0 = time.time()
J_train_opt =  precomputed_kernel_matrix_opt(list_projectors, list_projectors)
time_K_train_proj = time.time()-t0
score_proj, std_proj = get_scores(J_train, list_labels)
print(f"Score for kernel with projectors{score_proj} {std_proj}")
t0 = time.time()
K_train =  precomputed_kernel_matrix_bats(list_TT_cores, list_TT_cores, 2)
time_K_train_bats = time.time()-t0
score, std = get_scores_bats(list_TT_cores, list_labels)
print(f"Score for kernel with Bats {score} {std}")
"""



"""
Test projectors with singular bases
TT_ranks = [1,2,2,2,1]
file_faces_96 = 'dataset_faces_96'

list_projectors, list_TT_cores, list_labels, time_TT, time_proj= get_projectores_ameliorated(file_faces_96, TT_ranks)
bas_sing = get_base_single(list_TT_cores[10])
list_cores = list_TT_cores[10]
print(test_proj(list_cores[2], bas_sing[2]))
print(test_proj(list_cores[1], bas_sing[1]))
print(test_proj_matrix(list_cores[0], bas_sing[0]))
print(test_proj_matrix(list_cores[-1], bas_sing[-1]))
"""
























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