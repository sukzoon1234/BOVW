import numpy as np
import pandas as pd
import cv2 as cv2
from scipy.cluster.vq import vq
from tqdm import tqdm
import itertools

train_data = np.load('./bow_data/numpy_file/train_data.npy')

codebook = np.load('./bow_data/numpy_file/dense_codebook.npy')
#아마 k,128...? (200,128)

train_features = np.load('./bow_data/numpy_file/dense_train_features.npy')
#  (3133440, 128)
k = train_features.shape[1]
train_features = np.reshape(train_features, (3060,-1,k))
# 3060장,1024,128

sift = cv2.xfeatures2d.SIFT_create()

train_histogram = np.empty((0))


sift_size=8

########## VLAD  ###########

VLAD = []
for feature in train_features:
    code, dist = vq(feature, codebook) #x(1024),128  200,128       
    # ===> x(1024) 길이의 nparray  // codeword의 index를 담고있음(0~199?)
    #각각의 descriptor 들이 어떤 codeword에 가까운지에 대한 index 정보

    near_codeword = []
    for i in range(200):#각 codeword 별로 가장 가까운 descriptor의 index를 추가하기
        near_codeword.append((np.where(code == i))[0].tolist())
    
    for i, codeword in enumerate(codebook): #각 codeword로 부터 가장 가까운 descriptor 들만 더하기
        vector_sum = np.array((0))
        for near_idx in near_codeword[i]:
            vector_sum = vector_sum + (feature[near_idx] - codeword)
        if vector_sum.size==1:
            vector_sum = np.zeros((k))
        VLAD.append(vector_sum.tolist())

VLAD = np.reshape(np.array(VLAD), (3060,200,k))

np.save('./bow_data/numpy_file/VLAD', VLAD)