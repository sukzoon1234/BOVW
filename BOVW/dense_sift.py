import numpy as np
import pandas as pd
import cv2 as cv2
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA


train_data = np.load('./bow_data/numpy_file/train_data.npy')

sift = cv2.xfeatures2d.SIFT_create()

train_features = []

sift_size = 8

for data in tqdm(train_data):
    data = np.asarray(data, dtype=np.uint8)
    gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

    kp = [cv2.KeyPoint(x, y, sift_size)
        for y in range(0, data.shape[0], sift_size)
            for x in range(0, data.shape[1], sift_size)]
    _,des = sift.compute(gray, kp)  


    #rootSIFT 적용부분
    eps=1e-7
    des /= (np.sum(des, axis=1, keepdims=True)+eps)
    des = np.sqrt(des)

    train_features.append(des)


train_features = np.reshape(np.array(train_features),(-1,128))

#PCA 적용부분
pca = PCA(n_components=64, svd_solver='randomized', whiten=True)
train_features = pca.fit_transform(train_features)
print(pca.explained_variance_ratio_)

print(train_features.shape)
#(3133440, k)

np.save('./bow_data/numpy_file/dense_train_features', train_features)


 

# PCA 를 위해 test feature도 여기서 뽑아버리자
test_data = np.load('./bow_data/numpy_file/test_data.npy')

sift = cv2.xfeatures2d.SIFT_create()

test_histogram = np.empty((0))

sift_size = 8

test_features = []
for data in tqdm(test_data):
    # data_norm = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    data = np.asarray(data, dtype=np.uint8)
    gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)


    kp = [cv2.KeyPoint(x, y, sift_size)
        for y in range(0, data.shape[0], sift_size)
            for x in range(0, data.shape[1], sift_size)]
    _,des = sift.compute(gray, kp)

    #rootSIFT 적용부분
    eps=1e-7
    des /= (np.sum(des, axis=1, keepdims=True)+eps)
    des = np.sqrt(des)

    test_features.append(des.tolist())

#PCA 적용부분
test_features = np.reshape(np.array(test_features), (-1,128))
test_features = pca.transform(test_features)


print(test_features.shape)
np.save('./bow_data/numpy_file/test_features', test_features)