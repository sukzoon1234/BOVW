import numpy as np
import pandas as pd
import cv2 as cv2
from scipy.cluster.vq import vq
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV, KFold
from tqdm import tqdm
from matplotlib import pyplot as plt 
from sklearn.decomposition import PCA


VLAD = np.load('./bow_data/numpy_file/VLAD.npy')
#(3060, 200,128?)
print(VLAD.shape)
k = VLAD.shape[2]


# #Intra-normalization
intra_vlad = []
for i, each_VLAD in enumerate(VLAD): #(3060, 200,128?)
    intra_norm = np.linalg.norm(each_VLAD, axis=1, ord=2).reshape(-1,1)
    intra_norm = np.where(intra_norm==0, 1, intra_norm) #0으로 나누면 nan 뜨니까 1로 바꾸기
    ##############jmshin version##################
    # zero_idx = np.where(intra_norm==0)[0]
    # intra_norm[zero_idx] = np.finfo(float).tiny
    ##############################################

    out = each_VLAD / intra_norm
    intra_vlad.append(out)

VLAD = np.reshape(intra_vlad, (3060,-1))

# SSR normalization
SSR_vlad = []
VLAD = np.reshape(VLAD, (3060,-1))
for i, each_VLAD in enumerate(VLAD):#(3060, 200*k)
    out = np.sign(each_VLAD)*np.sqrt(abs(each_VLAD))
    l2_norm = np.linalg.norm(out, ord=2) #전체에 대해  L2 norm계산
    out = out / l2_norm #200*64
    SSR_vlad.append(out)

#L2_normalization
# for i, each_VLAD in enumerate(VLAD):
#     l2_norm = np.linalg.norm(each_VLAD, ord=2) #전체에 대해 계산
#     import pdb; pdb.set_trace()
#     VLAD[i] = VLAD[i] / l2_norm #200*64
# VLAD = np.reshape(np.array(SSR_vlad), (3060,-1))

VLAD = np.array(SSR_vlad)


##히스토그램 뽑아보기
# VLAD = np.reshape(VLAD, (3060, 200*k))
# x = np.arange(0,200*k)
# y = np.std(VLAD, axis=0)
# plt.bar(x,y)
# plt.title('Intra + L2')
# plt.xlabel('VLAD Dimension')
# plt.ylabel('std')
# plt.savefig('./L2_norm_std.png')

codebook = np.load('./bow_data/numpy_file/dense_codebook.npy')

label = pd.read_csv('./bow_data/Label2Names.csv').to_numpy()

y_train = np.repeat(np.arange(102), 30)


# #best parameter 찾기
# #svm=SVC()
# svm = LinearSVC()
# param_grid = {
# #    'kernel' : ['linear'],
#     'C' : [1,5,10],
#  #   'gamma' :[0.1,0.5],
# }
# grid_search = GridSearchCV(estimator=svm, param_grid=param_grid,refit=True,verbose=2)
# grid_search.fit(VLAD,y_train)
# print(grid_search.best_estimator_)
# import pdb; pdb.set_trace



#svm = SVC(kernel='linear', C=1.0, random_state=0, gamma=0.10)
svm = LinearSVC(C=100)
svm.fit(VLAD, y_train)





# ---------------------------------
#이제 predict 단계

test_features = np.load('./bow_data/numpy_file/test_features.npy')

test_features = np.reshape(np.array(test_features), (-1, 1024,64))
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
            vector_sum = np.zeros((k))
        VLAD.append(vector_sum.tolist())


VLAD = np.reshape(np.array(VLAD), (-1,200,k))



#Intra-normalization
intra_vlad = []
for i, each_VLAD in enumerate(VLAD): #(1712, 200,128?)
    intra_norm = np.linalg.norm(each_VLAD, axis=1, ord=2).reshape(-1,1)
    intra_norm = np.where(intra_norm==0, 1, intra_norm) #0으로 나누면 nan 뜨니까 1로 바꾸기
    ##############jmshin version##################
    # zero_idx = np.where(intra_norm==0)[0]
    # intra_norm[zero_idx] = np.finfo(float).tiny
    ##############################################

    out = each_VLAD / intra_norm
    intra_vlad.append(out)

VLAD = np.reshape(intra_vlad, (1712,-1))

# SSR normalization
SSR_vlad = []
VLAD = np.reshape(VLAD, (1712,-1))
for i, each_VLAD in enumerate(VLAD):#(3060, 200*k)
    out = np.sign(each_VLAD)*np.sqrt(abs(each_VLAD))
    l2_norm = np.linalg.norm(out, ord=2) #전체에 대해 계산
    out = out / l2_norm #200*64
    SSR_vlad.append(out)

#L2_normalization
# for i, each_VLAD in enumerate(VLAD):
#     l2_norm = np.linalg.norm(each_VLAD, ord=2) #전체에 대해 계산
#     import pdb; pdb.set_trace()
#     VLAD[i] = VLAD[i] / l2_norm #200*64
# VLAD = np.reshape(np.array(SSR_vlad), (3060,-1))

VLAD = np.array(SSR_vlad)


prediction = svm.predict(VLAD)

np.save('./bow_data/numpy_file/dense_prediction', prediction)

