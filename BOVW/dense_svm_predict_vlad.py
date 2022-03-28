import numpy as np
import pandas as pd
import cv2 as cv2
from scipy.cluster.vq import vq
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV, KFold
from tqdm import tqdm
from matplotlib import pyplot as plt 

VLAD = np.load('./bow_data/numpy_file/VLAD.npy')
#(3060, 200,128?)


#Intra-normalization
VLAD = np.reshape(VLAD, (3060,200,128))
for i, each_VLAD in enumerate(VLAD): #(3060, 200,128?)
    intra_norm = np.reshape(np.linalg.norm(each_VLAD, axis=1, ord=2), (-1,1))
    intra_norm = np.where(intra_norm==0, 1, intra_norm) #0으로 나누면 nan 뜨니까 1로 바꾸기
    VLAD[i] = VLAD[i] / intra_norm
VLAD = np.reshape(VLAD, (3060,-1))

# # SSR normalization
# VLAD = np.reshape(VLAD, (3060,200,128))
# for i, each_VLAD in enumerate(VLAD):#(3060, 200,128?)
#     VLAD[i] = np.sign(each_VLAD[:][:])*np.sqrt(abs(each_VLAD[:][:]))



#L2_normalization
VLAD = np.reshape(VLAD, (3060,200,128))
for i, each_VLAD in enumerate(VLAD):
    l2_norm = np.linalg.norm(each_VLAD, ord=2) #전체에 대해 계산
    VLAD[i] = VLAD[i] / l2_norm
print(l2_norm)
VLAD = np.reshape(VLAD, (3060,-1))




#히스토그램 뽑아보기
VLAD = np.reshape(VLAD, (3060, 25600))
x = np.arange(0,25600)
y = np.std(VLAD, axis=0)
plt.bar(x,y)
plt.title('Intra + L2')
plt.xlabel('VLAD Dimension')
plt.ylabel('std')
plt.savefig('./L2_norm_std.png')

codebook = np.load('./bow_data/numpy_file/dense_codebook.npy')

label = pd.read_csv('./bow_data/Label2Names.csv').to_numpy()

y_train = np.repeat(np.arange(102), 30)

# #best parameter 찾기
# svm=SVC()
# param_grid = {
#     'kernel' : ['linear', 'rbf'],
#     'C' : [5],
#     'gamma' :[0.1],
# }
# grid_search = GridSearchCV(estimator=svm, param_grid=param_grid,refit=True,verbose=2)
# grid_search.fit(VLAD,y_train)
# print(grid_search.best_estimator_)
# import pdb; pdb.set_trace



#svm = SVC(kernel='linear', C=4.0, random_state=0, gamma=0.10)
svm = LinearSVC(C=1)
svm.fit(VLAD, y_train)


# ---------------------------------
#이제 predict 단계

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

            #kp, des = sift.detectAndCompute(data_8bit, None)
                #des 는 (?,128) 의 numpy array vectord
    test_features.append(des.tolist())


    # code, dist = vq(test_features, codebook) #x,128  200,128   
    # # ===> x 길이의 nparray  // codeword의 index를 담고있음(0~199?)
    
    # histogram, bin_edge = np.histogram(code, np.arange(201), (0,201))
    # #각 히스토그램 ndarray의 길이는 200
    # test_histogram = np.append(test_histogram, histogram, axis=0)

# test_histogram = np.reshape(test_histogram, (-1,200))


test_features = np.reshape(np.array(test_features), (-1, 1024,128))
print(test_features.shape)


VLAD = []
for feature in test_features:
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
            vector_sum = np.zeros((128))
        VLAD.append(vector_sum.tolist())


VLAD = np.reshape(np.array(VLAD), (-1,200,128))


#Intra-normalization
VLAD = np.reshape(VLAD, (-1,200,128))
for i, each_VLAD in enumerate(VLAD):
    intra_norm = np.reshape(np.linalg.norm(each_VLAD, axis=1, ord=2), (-1,1))
    intra_norm = np.where(intra_norm==0, 1, intra_norm) #0으로 나누면 nan 뜨니까 1로 바꾸기
    VLAD[i] = VLAD[i] / intra_norm
VLAD = np.reshape(VLAD, (-1,25600))

# #SSR normalization
# VLAD = np.reshape(VLAD, (-1,200,128))
# for i, each_VLAD in enumerate(VLAD):#(17XX, 200,128?)
#     VLAD[i] = np.sign(each_VLAD[:][:])*np.sqrt(abs(each_VLAD[:][:]))
    

#L2_normalization
VLAD = np.reshape(VLAD, (-1,200,128))
for i, each_VLAD in enumerate(VLAD):
    l2_norm = np.linalg.norm(each_VLAD, ord=2) #전체에 대해 계산
    VLAD[i] = VLAD[i] / l2_norm

VLAD = np.reshape(VLAD, (-1, 25600))

prediction = svm.predict(VLAD)

np.save('./bow_data/numpy_file/dense_prediction', prediction)

