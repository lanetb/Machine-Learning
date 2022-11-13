from keras import utils as kutils
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras.layers import Dropout, Flatten, Conv2D, Input, MaxPooling2D
from keras.models import Sequential
from keras.layers.core import Dense

def preprocess_data(X, y):
    """ Manipulate X and y from their given form in the MNIST dataset to
       whatever form your model is expecting."""
    X_train = X.reshape(-1,28,28,1)
    y_train_cat = tf.keras.utils.to_categorical(y, num_classes = 10)
    # This function does nothing but return what it is sent    
    return X_train, y_train_cat