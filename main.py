# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 10:49:51 2021

@author: adminlocal
"""
import sys
from sklearn.preprocessing import StandardScaler
from Construct_TT_cores import  process_data_videos, get_TT_svd
chemin = ("C:/Users/adminlocal/Desktop/Code_n")
sys.path.append(chemin)

import numpy as np
from kernel import get_scores

import tqdm
import pickle

"""
with open('videos_processed', 'rb') as pickle_file:
    content = pickle.load(pickle_file)
    
list_data = content[0]
labels = content[1]
TT_ranks = [1,2,2,2,1]
print("Get TT-svd")
list_TT_cores =  get_TT_svd(list_data, TT_ranks)
"""

l=0.8
Nrep = 1
scores =  np.zeros(Nrep)
time_svm = []
for nrep in tqdm.tqdm(range(Nrep)):
    print("Début svms")
    #sc, t1, t11 =  get_scores_donn_re_valid(list_ho, list_labels, l)
    sc, t1, t11 = get_scores(list_TT_cores, labels, l, TT_ranks)
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