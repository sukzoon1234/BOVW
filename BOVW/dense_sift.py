import numpy as np
import pandas as pd
import cv2 as cv2
from PIL import Image
from tqdm import tqdm
# train_data = datasets.train_data # 3060*256*256*3 의 shape
# test_data = datasets.test_data
# label = datasets.label


train_data = np.load('./bow_data/numpy_file/train_data.npy')

#label = np.load('./bow_data/numpy_file/label.npy')

sift = cv2.xfeatures2d.SIFT_create()

train_features = []

sift_size = 8

i=0

for data in tqdm(train_data):
    data = np.asarray(data, dtype=np.uint8)
    gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

    kp = [cv2.KeyPoint(x, y, sift_size)
        for y in range(0, data.shape[0], sift_size)
            for x in range(0, data.shape[1], sift_size)]
    _,des = sift.compute(gray, kp)
            #kp, des = sift.detectAndCompute(data_8bit, None)
                #des 는 (?,128) 의 numpy array vector
    train_features.append(des)

    if i==0:
        cv2.imwrite("./img.jpg", data)
        img = cv2.imread("./img.jpg")
        res = cv2.drawKeypoints(img, kp, None)
        cv2.imwrite("./dense.jpg", res)
        i+=1




train_features = np.reshape(np.array(train_features),(-1,128))
#(3133440, 128)

np.save('./bow_data/numpy_file/dense_train_features', train_features)
#train_features 의 shape는 (?,128) 
