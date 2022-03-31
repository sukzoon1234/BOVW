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