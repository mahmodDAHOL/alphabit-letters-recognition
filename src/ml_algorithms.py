import pickle
from pathlib import Path

import sklearn.metrics as sm
from sklearn.linear_model import LogisticRegression as lr
from sklearn.naive_bayes import GaussianNB as nb
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.svm import SVC
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

from cnn import CNN
from data_manipulation import DataManipulation
from utils import save_training_results


class MLAlgorithms:

    def __init__(self, project_path: Path, batch_size=None, epochs=None):
        self.project_path = project_path

        self.batch_size = batch_size
        self.epochs = epochs
        self.path_to_save = project_path.joinpath("results")

    def predict_svm(self, X_train, X_test, y_train, y_test):
        svc = SVC(kernel='linear')
        print("[INFO] svm started")
        svc.fit(X_train, y_train)
        y_pred = svc.predict(X_test)
        self.calc_accuracy("SVM", y_test, y_pred, self.path_to_save)

        # save the model to file
        file_name = self.project_path.joinpath("models/SVM")
        outfile = open(file_name, 'wb')
        pickle.dump(svc, outfile)
        outfile.close()

    def predict_lr(self, X_train, X_test, y_train, y_test):
        clf = lr()
        print("[INFO] lr started")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        self.calc_accuracy("Logistic regression", y_test,
                           y_pred, self.path_to_save)

        # save the model to file
        file_name = self.project_path.joinpath("models/LOGISTIC_REGRESSION")
        outfile = open(file_name, 'wb')
        pickle.dump(clf, outfile)
        outfile.close()

    def predict_nb(self, X_train, X_test, y_train, y_test):
        clf = nb()
        print("[INFO] nb started")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        self.calc_accuracy("Naive Bayes", y_test, y_pred, self.path_to_save)

        # save the model to file
        file_name = self.project_path.joinpath("models/NAIVE_BAYES")
        outfile = open(file_name, 'wb')
        pickle.dump(clf, outfile)
        outfile.close()

    def predict_knn(self, X_train, X_test, y_train, y_test):
        clf = knn(n_neighbors=3)
        print("[INFO] knn started")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        self.calc_accuracy("K nearest neighbours", y_test,
                           y_pred, self.path_to_save)

        # save the model to file
        file_name = self.project_path.joinpath("models/KNN")
        outfile = open(file_name, 'wb')
        pickle.dump(clf, outfile)
        outfile.close()

    def predict_mlp(self, X_train, X_test, y_train, y_test):
        clf = mlp()
        print("[INFO] mlp started")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        self.calc_accuracy("MLP classifier", y_test, y_pred, self.path_to_save)

        # save the model to file
        file_name = self.project_path.joinpath("models/ML_PERSEPTRON")
        outfile = open(file_name, 'wb')
        pickle.dump(clf, outfile)
        outfile.close()

    def train_and_predict_cnn(self, project_path):
        cnn = CNN(self.project_path)
        train_images, train_labels_label_encoded, test_images, test_labels_label_encoded = \
            DataManipulation.cnn_preprocess(self.project_path)
        model = cnn.create_model()
        sgd = SGD(lr=1e-2)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd, metrics=['accuracy'])

        model.summary()

        # One hot encoding
        train_labels_one_hot = to_categorical(train_labels_label_encoded)
        test_labels_one_hot = to_categorical(test_labels_label_encoded)
        # training the model
        history = model.fit(train_images, train_labels_one_hot, batch_size=self.batch_size,
                            epochs=self.epochs, verbose=1, validation_data=(
                                test_images, test_labels_one_hot))

        file_name = self.project_path.joinpath("models/keras_model.h5")
        model.save(file_name)

        y_pred, y_test = cnn.predict_cnn(
            test_images, test_labels_label_encoded)
        cnn.plot_accuracy_and_loss(history, cnn)
        self.calc_accuracy("CNN", y_test, y_pred, self.path_to_save)

    def calc_accuracy(self, method, label_test, pred, path_to_save: Path):
        print("[INFO] accuracy score for ", method,
              sm.accuracy_score(label_test, pred))
        print("[INFO] precision_score for ", method, sm.precision_score(
            label_test, pred, average='micro'))
        print("[INFO] f1 score for ", method, sm.f1_score(
            label_test, pred, average='micro'))
        print("[INFO] recall score for ", method, sm.recall_score(
            label_test, pred, average='micro'))

        results = \
            f"""
accuracy score for {method} {sm.accuracy_score(label_test, pred)}
precision_score for {method} {sm.precision_score(label_test, pred, average='micro')}
f1 score for {method} {sm.f1_score(label_test, pred, average='micro')}
recall score for {method} {sm.recall_score(label_test, pred, average='micro')}
                """

        save_training_results(results, path_to_save)
