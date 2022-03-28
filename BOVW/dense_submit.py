import numpy as np
import pandas as pd
import os


prediction = np.load('./bow_data/numpy_file/dense_prediction.npy')
test_id = sorted(os.listdir('./bow_data/test_csv_v2'))

predict=prediction.reshape(-1,1)
test_id=np.array(test_id)
test_id=test_id.reshape(-1,1)

total_result=np.hstack([test_id,predict])
df = pd.DataFrame(total_result,columns=["Id","Category"])
df.to_csv("./bow_data/dense_submit.csv",index=False,header=True)