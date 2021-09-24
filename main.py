# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 10:49:51 2021

@author: adminlocal
"""
import sys
from Construct_TT_cores import  get_TT_svd
chemin = ("C:/Users/adminlocal/Desktop/Code_n")
sys.path.append(chemin)

import numpy as np
import os
from kernel_optimized import construct_list_projectors, get_scores

import tqdm
import pickle





# Check if projectors are computed
proj_file = "projectors_val3.pic"
if os.path.isfile(proj_file):
	with open(proj_file, 'rb') as proj_file:
    		data = pickle.load(proj_file)
    		list_projectors = data['projectors']
    		list_labels = data['labels']
 
else:
    
    with open('videos_processed', 'rb') as pickle_file:
        content = pickle.load(pickle_file)
    
    list_data = content[0]
    list_labels = content[1]
    TT_ranks = [1,2,4,2,1]
    N_video = len(list_data)
    print("Get TT-svd")
    list_TT_cores =  get_TT_svd(list_data, TT_ranks)
    
    list_projectors = []
    for  n in range(N_video):
        #construit le projecteur pour la video n
        proj = construct_list_projectors(list_TT_cores[n])
        #ajoute le projecteur et le label dans leurs listes respectives
        list_projectors.append(proj)
	# On enregistre tout ca dans un fichier pour la prochaine fois sous forme de dictionnaire
    with open(proj_file,'wb') as proj_file:
        pickle.dump({'projectors':list_projectors, 'labels':list_labels} , proj_file)










l=1
Nrep = 1
scores =  np.zeros(Nrep)
time_svm = []
for nrep in tqdm.tqdm(range(Nrep)):
    print("Début svms")
    #sc, t1, t11 =  get_scores_donn_re_valid(list_ho, list_labels, l)
    sc, t1, t11 = get_scores(list_TT_cores, list_labels, l)
    time_svm.append(t1+ t11)
    print("t1", t1)
    print("t11", t11)
    print("Sc", sc)
    scores[nrep] = sc

"""
#Dataset of Extended yale dataset
TT_ranks= [1,2,2,2,1]
list_data, labels = read_dataset_extended()
print("get TT cores")
list_TT_cores =  get_TT_svd(list_data, TT_ranks)
"""

"""    
list_data, labels = process_data_videos("C:/Users/adminlocal/Desktop/Code_n/Video_jumping_walking")

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