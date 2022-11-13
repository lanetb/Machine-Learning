import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, Dropout, Flatten, Input, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Dense
from keras.models import Sequential
from tensorflow import keras

X_train_p = np.load("MNIST_X_train.npy")
y_train = np.load("MNIST_y_train.npy")
X_test_p = np.load("MNIST_X_test.npy")
y_test = np.load("MNIST_y_test.npy")

X_train = X_train_p.reshape(-1,28,28,1)
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes = 10)

dataGen = ImageDataGenerator(rotation_range=10,width_shift_range=0.2,height_shift_range=0.2,
                                shear_range=0.15,zoom_range=[0.5,2],validation_split=0.2)
dataGen.fit(X_train)
train_generator = dataGen.flow(X_train, y_train_cat, batch_size=64, shuffle=True, 
                                                seed=2, save_to_dir=None, subset='training')
validation_generator = dataGen.flow(X_train, y_train_cat, batch_size=64, shuffle=True, 
                                                seed=2, save_to_dir=None, subset='validation')


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
    history = model.fit(X, steps_per_epoch = 600, epochs=10, validation_data = y, validation_steps = 150)
    return model

def predict_nn_nonorm(model, X, y):
    probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
    predictions = probability_model.predict(X)
    return predictions

#model = make_model()
#model = train_model(train_generator, validation_generator, model)