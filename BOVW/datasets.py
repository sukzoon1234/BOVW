import pandas as pd
import os
import glob
import numpy as np
from tqdm import tqdm

train_dir = sorted(glob.glob(os.path.join('./bow_data/train_csv_v2', '*', '*')))
test_dir = sorted(glob.glob(os.path.join('./bow_data/test_csv_v2', '*')))

train_data = []
for data in tqdm(train_dir):
    train_data.append(np.array(pd.read_csv(data)))
train_data = np.reshape(train_data, (-1,256,256,3))
np.save('./bow_data/numpy_file/train_data', train_data)

test_data = []
for data in tqdm(test_dir):
    test_data.append(np.array(pd.read_csv(data)))
test_data = np.reshape(test_data, (-1,256,256,3))
np.save('./bow_data/numpy_file/test_data', test_data)


# train_dir = './bow_data/train_csv_v2'
# train_data = []
# for label in sorted(os.listdir(train_dir)):
#     for data in sorted(os.listdir(os.path.join(train_dir,label))):
#         data_dir = os.path.join(train_dir, label, data)
#         train_data.append(pd.read_csv(data_dir).values.tolist())
        
# train_data = np.reshape(np.array(train_data), (-1,256,256,3))
# np.save('./bow_data/numpy_file/train_data', train_data)


# test_dir = './bow_data/test_csv_v2'
# test_data = []
# for data in sorted(os.listdir(test_dir)):
#     data_dir = os.path.join(test_dir, data)
#     test_data.append(pd.read_csv(data_dir).values.tolist())

# test_data = np.reshape(np.array(test_data), (-1,256,256,3))
# np.save('./bow_data/numpy_file/test_data', test_data)





#256*256*3= 199608*1.  이미지 한장당 shape.
#train 이미지는 이미지는 3060장
#결과적인 shape을  3060*256*256*3 으로 맞춰야함