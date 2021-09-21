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


def Normalize(x):
    """
        argument
            - x: input image data in numpy array [32, 32, 3]
        return
            - normalized x 
    """
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x


def standardize(list_data):
    list_data_normalized=[]
    for data in list_data:
        temp = (data-np.mean(data))/(np.std(data))
        list_data_normalized.append(temp)
    return list_data_normalized
        
def standardize_data(list_data):#list_data: chaque elem est une liste de TT_cores
    standardized_data=[]
    
    for X in list_data:
        temp = standardize(X)
        standardized_data.append(temp)
        
    return standardized_data

def preprocess_dataset(list_data, labels):
    #list_data, labels = process_data_videos("C:/Users/adminlocal/Desktop/Code_n/Video_jumping_walking")
    #arr_dataset = np.asarray(list_data)
    #data = np.asarray(list_data).reshape(216, -1)
    scaler = StandardScaler()
    scaler.fit(np.asarray(list_data).reshape(216, -1))
    scaler.transform(data)
    #data = data.reshape(216, 93, 240, 320, 3)
    return data.reshape(216, 93, 240, 320, 3), labels

def preprocess_dataset1(list_data, labels):
    data = np.asarray(list_data).reshape(216, -1)
    
    for i in range(data.shape[1]):
        data[:,i] = (data[:,i]-np.mean(data[:,i]))/(np.std(data[:,i]))
        
    return data.reshape(216, 93, 240, 320, 3), labels
    
#list_TT_cores = standardize_data(list_data_TT_cores)
#list_data, labels = process_data_videos("C:/Users/adminlocal/Desktop/Code_n/Video_jumping_walking")
#list_data = standardize(list_data)
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

with open('videos_processed', 'rb') as pickle_file:
    content = pickle.load(pickle_file)
    
list_data = content[0]
labels = content[1]


TT_ranks = [1,2,2,2,1]
print("Preprocess dataset")
data, labels = preprocess_dataset1(list_data, labels)
#list_data = list(list_data)
print("Get TT-svd")
list_TT_cores =  get_TT_svd(list(data), TT_ranks)


def read_dataset_extended():
    file = open('C:/Users/adminlocal/Desktop/Code_n/Tensors_Extended_Yale_3lasses', 'rb')
    data = pickle.load(file)
    file.close()

    T_c0 = list(data[0].reshape(16,9,480,640,4))
    T_c1 = list(data[1].reshape(16,9,480,640,4))
    T_c2 = list(data[2].reshape(16,9,480,640,4))
    
    data = T_c0 + T_c1 + T_c2
    labels = 16*[0] + 16*[1] + 16*[2]
    
    return data, labels
"""
#Dataset of Extended yale dataset
TT_ranks= [1,2,2,2,1]
list_data, labels = read_dataset_extended()
print("get TT cores")
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
