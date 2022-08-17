import os
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from data_manipulation import DataManipulation
from ml_algorithms import MLAlgorithms
from utils import extract_sift

batch_size = 64
epochs = 40

y = []
label = 0
img_descs = []
project_path = Path("main.py").parent.absolute().parent
dataset_path = project_path.joinpath("dataset")

# Dataset link => https://drive.google.com/drive/folders/1SY67sDO2ROoOoBhTBIIDn17gStS0AvCB
# creating desc for each file with label
for train_test in ["train", "test"]:
    print(f"[INFO] loading {train_test} data")
    for (dirpath, dirnames, filenames) in os.walk(dataset_path.joinpath(train_test)):
        for dirname in dirnames:
            for(direcpath, direcnames, files) in os.walk(dataset_path.joinpath(train_test, dirname)):
                for file in files:
                    actual_path = dataset_path.joinpath(
                        train_test, dirname, file)
                    des = extract_sift(actual_path)
                    img_descs.append(des)
                    y.append(label)
            label = label+1

# finding indexes of test train and validate
y = np.array(y)
data = DataManipulation(0.3, 0, project_path)
data.train_test_val_split_idxs(len(img_descs))

# creating histogram using kmeans minibatch cluster model
model = MiniBatchKMeans(batch_size=1024, n_clusters=150)
X = data.cluster_features(img_descs, model)

# splitting data into test, train, validate using the indexes
X_train, X_test, X_val, y_train, y_test, y_val = data.perform_data_split(
    X, y)

algorithm = MLAlgorithms(project_path, batch_size=batch_size, epochs=epochs)
# using classification methods
algorithm.predict_knn(X_train, X_test, y_train, y_test)
algorithm.predict_mlp(X_train, X_test, y_train, y_test)
algorithm.predict_svm(X_train, X_test, y_train, y_test)
algorithm.predict_lr(X_train, X_test, y_train, y_test)
algorithm.predict_nb(X_train, X_test, y_train, y_test)
algorithm.train_and_predict_cnn(project_path)
