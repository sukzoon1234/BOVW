# BOVW
### Bag Of Visual Words

### Datasets
- https://www.kaggle.com/competitions/2020mltermprojectbovw/overview
- 위의 kaggle 링크에 제공된 csv data를 사용

### datasets.py
- dataloader 부분
- csv 파일 읽어오기

### dense_sift.py
- train data에서 dense_SIFT로 descriptor 추출
- descriptor(128dim) 를 rootSIFT & PCA 를 통해 차원축소
- test data에서도 dense_SIFT로 descriptor 추출
- descriptor(128dim) 를 rootSIFT & PCA 를 통해 차원축소

### dense_Xmeans.py
- train data에서 추출한 descriptor를 이용해서 X means clustering을 통해 codebook 생성

### dense_histogram_vlad.py
- 앞에서 PCA로 차원을 축소시킨 train data의 descriptor를 codebook을 이용해서 VLAD 형태로 변형

### dense_svm_predict_vlad.py
- VLAD형태로 변형한 descriptor에다가 normalization 적용 (Intra, SSR, L2,,)
- SVM 모델 훈련
- 앞서 추출한 test data의 descriptor 에 대해서도 VLAD 형태로 변형
- 훈련된 SVM을 통해 정답 predict

### dense_submit.py
- csv 파일로 변환 후 submit
