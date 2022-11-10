import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras, layers
from keras.layers import Dropout, Flatten, Conv2D, Input, MaxPooling2D
from keras.models import Sequential
from keras.layers.core import Dense

mnist = tf.keras.datasets.mnist
(train_nums, train_labels), (test_nums, test_labels) = mnist.load_data()

def make_model():
        model = tf.keras.Sequential()
        model.add(Input(shape=(-1,28,28,1)))
        model.add(Conv2D(32,kernal_size=(3,3), activation='relu')),
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Flatten())
        model.add(Dropout())
        model.add()
        return model

def train_model(X, y, model):
    model.compile(loss='categorical_crossentropy', optmizer='adam', metrics=['accuracy'])
    history = model.fit(X, y, epochs=10, batch_size=10, validation_split = 0.20)
    return model

