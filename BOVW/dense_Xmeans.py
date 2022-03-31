import faiss
import numpy as np
from tqdm import tqdm

def x_means(X, voc_size):
    feature=np.array(X).reshape(-1,X.shape[1]).astype('float32')
    d=feature.shape[1]  #128
    k=voc_size
    clus = faiss.Clustering(d, k)
    clus.niter = 300
    clus.seed =10
    clus.max_points_per_centroid = 10000000
    ngpu=1
    res = [faiss.StandardGpuResources() for i in range(ngpu)]
    flat_config = []
    for i in tqdm(range(ngpu)):
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = i
        flat_config.append(cfg)
    if ngpu == 1:
        index = faiss.GpuIndexFlatL2(res[0], d, flat_config[0])
    clus.train(feature, index)
    centroids = faiss.vector_float_to_array(clus.centroids)
    centroids=centroids.reshape(k, d)
    
    np.save('./bow_data/numpy_file/dense_codebook', centroids)
    #return centroids

if __name__ == '__main__':
    features = np.load('./bow_data/numpy_file/dense_train_features.npy')
    print(features.shape) # (3133440, 128)
    x_means(features, 200)