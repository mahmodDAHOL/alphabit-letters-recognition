
import random
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder

from cnn import CNN


class DataManipulation:
    def __init__(self, percent_test, percent_val, project_path: Path):
        self.project_path = project_path
        self.percent_test = percent_test
        self.percent_val = percent_val
        self.training_idxs = []
        self.test_idxs = []
        self.val_idxs = []

    # utility functions
    def perform_data_split(self, X, y):
        """
        Split X and y into train/test/val sets
        Parameters:
        -----------
        X : eg, use img_bow_hist
        y : corresponding labels for X
        training_idxs : list/array of integers used as indicies for training rows
        test_idxs : same
        val_idxs : same
        Returns:
        --------
        X_train, X_test, X_val, y_train, y_test, y_val
        """
        X_train = X[self.training_idxs]
        X_test = X[self.test_idxs]
        X_val = X[self.val_idxs]

        y_train = y[self.training_idxs]
        y_test = y[self.test_idxs]
        y_val = y[self.val_idxs]

        return X_train, X_test, X_val, y_train, y_test, y_val

    def train_test_val_split_idxs(self, total_rows):
        """
        Get indexes for training, test, and validation rows, given a total number of rows.
        Assumes indexes are sequential integers starting at 0: eg [0,1,2,3,...N]
        Returns:
        --------
        training_idxs, test_idxs, val_idxs
            Both lists of integers
        """
        if self.percent_test + self.percent_val >= 1.0:
            raise ValueError(
                "percent_test and percent_val must sum to less than 1.0")

        row_range = range(total_rows)

        no_test_rows = int(total_rows*(self.percent_test))
        self.test_idxs = np.random.choice(
            row_range, size=no_test_rows, replace=False)
        # remove test indexes
        row_range = [idx for idx in row_range if idx not in self.test_idxs]

        no_val_rows = int(total_rows*(self.percent_val))
        self.val_idxs = np.random.choice(
            row_range, size=no_val_rows, replace=False)
        # remove validation indexes
        self.training_idxs = [
            idx for idx in row_range if idx not in self.val_idxs]

    def cluster_features(self, img_descs, cluster_model):
        """
        Cluster the training features using the cluster_model
        and convert each set of descriptors in img_descs
        to a Visual Bag of Words histogram.
        Parameters:
        -----------
        X : list of lists of SIFT descriptors (img_descs)
        training_idxs : array/list of integers
            Indicies for the training rows in img_descs
        cluster_model : clustering model (eg KMeans from scikit-learn)
            The model used to cluster the SIFT features
        Returns:
        --------
        X, cluster_model :
            X has K feature columns, each column corresponding to a visual word
            cluster_model has been fit to the training set
        """
        n_clusters = cluster_model.n_clusters

        # Concatenate all descriptors in the training set together
        training_descs = [img_descs[i] for i in self.training_idxs]

        all_train_descriptors = [
            desc for desc_list in training_descs for desc in desc_list]

        all_train_descriptors = np.array(all_train_descriptors)

        print(f"[INFO] descriptors shape {all_train_descriptors.shape}")

        # Cluster descriptors to get codebook
        print(f"[INFO] Using clustering model { repr(cluster_model)}...")
        print(
            f"[INFO] Clustering on training set to get codebook of {n_clusters} words")

        # train kmeans or other cluster model on those descriptors selected above
        cluster_model.fit(all_train_descriptors)
        print("[INFO] done clustering. Using clustering model to generate BoW histograms for each image.")

        # compute set of cluster-reduced words for each image
        img_clustered_words = [cluster_model.predict(
            raw_words) for raw_words in img_descs]

        # finally make a histogram of clustered word counts for each image. These are the final features.
        img_bow_hist = np.array(
            [np.bincount(clustered_words, minlength=n_clusters) for clustered_words in img_clustered_words])

        X = img_bow_hist
        print("[INFO] done generating BoW histograms.")

        return X

    def cnn_preprocess(project_path: Path):
        cnn = CNN(project_path)
        # Loading the train images and their corresponding labels
        print("[INFO] loading train data")
        train_data = cnn.load_images(cnn.train_path)

        # Loading the test images and their corresponding labels

        print("[INFO] loading test data")
        test_data = cnn.load_images(cnn.test_path)
        print("[INFO] train and test are loaded")

        # Shuffling the data
        random.shuffle(train_data)

        # Seperating features and labels
        train_images = []
        train_labels = []
        test_images = []
        test_labels = []

        for feature, label in train_data:
            train_images.append(feature)
            train_labels.append(label)

        for feature, label in test_data:
            test_images.append(feature)
            test_labels.append(label)

        # Converting images list to numpy array
        train_images = np.array(train_images)
        test_images = np.array(test_images)
        train_images = train_images.reshape((-1, 224, 224, 1))
        test_images = test_images.reshape((-1, 224, 224, 1))

        # Changing the datatype and Normalizing the data
        train_images = train_images.astype('float32')
        test_images = test_images.astype('float32')

        train_images = train_images/255.0
        test_images = test_images/255.0

        # Encoding the label values
        le = LabelEncoder()
        le.fit_transform(train_labels)
        le = LabelEncoder()
        le.fit_transform(test_labels)

        train_labels_label_encoded = le.transform(train_labels)
        test_labels_label_encoded = le.transform(test_labels)

        return train_images, train_labels_label_encoded, test_images, test_labels_label_encoded
