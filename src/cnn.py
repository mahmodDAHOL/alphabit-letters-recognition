import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.models import Sequential, load_model


class CNN:
    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.test_path = self.project_path.joinpath("dataset/test")
        self.train_path = self.project_path.joinpath("dataset/train")
        self.input_shape = (224, 224, 1)
        self.n_classes = 24

    # Loading images
    # Converting images to an size of (224,224)
    def load_images(self, folder):
        data = []
        for label in os.listdir(folder):
            path = folder.joinpath(label)
            for img in os.listdir(path):
                img = cv2.imread(str(path.joinpath(img)), cv2.IMREAD_GRAYSCALE)
                new_img = cv2.resize(img, (224, 224))
                if new_img is not None:
                    data.append([new_img, label])
        return data

    def create_model(self):
        self.model = Sequential()
        # The first two layers with 32 filters of window size 3x3
        self.model.add(Conv2D(16, (2, 2),
                              activation='relu', input_shape=self.input_shape))
        self.model.add(MaxPooling2D(pool_size=(2, 2),
                       strides=(2, 2), padding='same'))

        self.model.add(Conv2D(32, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(3, 3),
                       strides=(3, 3), padding='same'))

        self.model.add(Conv2D(64, (5, 5), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(5, 5),
                       strides=(5, 5), padding='same'))

        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(self.n_classes, activation='softmax'))
        return self.model

    def predict_cnn(self, test_images, test_labels):

        # load the model from disk
        file_path = self.project_path.joinpath("models/keras_model.h5")
        cnn_model = load_model(file_path)
        y_pred = cnn_model.predict(test_images)
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred, test_labels

    def plot_accuracy_and_loss(self, history, cnn):

        # Visualizing loss
        plt.figure(figsize=[8, 6])
        plt.plot(history.history['loss'], 'r', linewidth=2.0)
        plt.plot(history.history['val_loss'], 'b', linewidth=2.0)
        plt.legend(['Training loss', 'Validation Loss'], fontsize=15)
        plt.xlabel('Epochs ', fontsize=16)
        plt.ylabel('Loss', fontsize=16)
        plt.title('Loss Curves', fontsize=16)
        plt.savefig(cnn.project_path.joinpath(
            "results/cnn_training_result/loss.png"))

        # Visualizing accuracy
        plt.figure(figsize=[8, 6])
        plt.plot(history.history['accuracy'], 'r', linewidth=2.0)
        plt.plot(history.history['val_accuracy'], 'b', linewidth=2.0)
        plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=15)
        plt.xlabel('Epochs ', fontsize=16)
        plt.ylabel('Accuracy', fontsize=16)
        plt.title('Accuracy Curves', fontsize=16)
        plt.savefig(
            cnn.project_path.joinpath("results/cnn_training_result/acc.png"))
