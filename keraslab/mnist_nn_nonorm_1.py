import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers.core import Dense

X_train = np.load('MNIST_X_train.npy')
y_train = np.load('MNIST_y_train.npy')
X_test = np.load('MNIST_X_test.npy')
y_test = np.load('MNIST_y_test.npy')




epochs    = 20
batch_size = 1000
num_neurons  = 20


def make_model(layer1_nodes, activation_function='sigmoid'):
    """ relu activation, mse as loss and adam optimizer"""

    model = Sequential()
    model.add(Dense(layer1_nodes, input_shape=(1,), activation=activation_function))
    model.add(Dense(1))
    print(model.summary())
    model.compile(loss='mse', optimizer='adam')
    return model

## example code
mymodel = make_model(2)

def train_model(X, y, model, epochs, batch_size):
    ''' train the model for specified number of epochs, batch_size'''
    h = model.fit(X, y, validation_split=0.2,
               epochs=epochs,
               batch_size=batch_size,
               verbose=1)
    plt.figure(figsize=(15,2.5))
    plt.plot(h.history['loss'])
    return model

def predict_model(X, y):    
    pred = model.predict(X)
    mid_range = 20
    X2 = np.random.random((n_samples,1))*mid_range-(mid_range/2)
    pred2 = model.predict(X2)
