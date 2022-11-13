import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras.layers import Dropout, Flatten, Conv2D, Input, MaxPooling2D
from keras.models import Sequential
from keras.layers.core import Dense

X_train_p = np.load("MNIST_X_train.npy")
y_train = np.load("MNIST_y_train.npy")
X_test_p = np.load("MNIST_X_test.npy")
y_test = np.load("MNIST_y_test.npy")

X_train = X_train_p.reshape(-1,28,28,1)
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes = 10)


def make_model():
        model = tf.keras.Sequential()
        model.add(Input(shape=(28,28,1)))
        model.add(Conv2D(32, kernel_size=(2,2), activation='relu')),
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(64, kernel_size=(2,2), activation='relu')),
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(128, kernel_size=(2,2), activation='relu')),
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Flatten())
        model.add(Dropout(0.3))
        model.add(Dense(10, activation='softmax'))
        return model

def train_model(X, y, model):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X, y, epochs=10, batch_size=10, validation_split = 0.20)
    return model

def predict_nn_nonorm(model, X, y):
    probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
    predictions = probability_model.predict(X)
    return predictions

#model = make_model()
#model = train_model(X_train, y_train_cat, model)
#model.save("mnist_cnn_model.h5")