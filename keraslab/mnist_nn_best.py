import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers.core import Dense

mnist = tf.keras.datasets.mnist
(train_nums, train_labels), (test_nums, test_labels) = mnist.load_data()
train_nums = train_nums / 255
test_nums = test_nums / 255

def make_model_nn_with_norm_relu():
    model = tf.keras.Sequential([
        tf.keras.layers.Normalization(axis=None),
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation = 'softmax')
    ])
    return model

def train_model_nn_with_norm_relu(X, y, model):
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(X, y, epochs=10, batch_size=10, validation_split = 0.20)
    return model

def predict_nn_with_norm_relu(X, y):
    probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
    predictions = probability_model.predict(X)
    return predictions


model = make_model_nn_with_norm_relu()
model = train_model_nn_with_norm_relu(train_nums, train_labels, model)

test_loss, test_acc = model.evaluate(test_nums,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

model.save("mnist_dense_model.h5")



