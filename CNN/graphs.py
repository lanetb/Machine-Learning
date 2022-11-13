import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras.layers import Dropout, Flatten, Conv2D, Input, MaxPooling2D
from keras.models import Sequential
from keras.layers.core import Dense
from keras.preprocessing.image import ImageDataGenerator

import mnist_nn_cnn_1 as m1
import mnist_nn_cnn_2 as m2
import mnist_nn_cnn_3 as m3

X_train_p = np.load("MNIST_X_train.npy")
y_train = np.load("MNIST_y_train.npy")
X_test_p = np.load("MNIST_X_test.npy")
y_test = np.load("MNIST_y_test.npy")

X_train = X_train_p.reshape(-1,28,28,1)
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes = 10)
X_test = X_test_p.reshape(-1,28,28,1)
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes = 10)

dataGen = ImageDataGenerator(rotation_range=10,width_shift_range=0.2,height_shift_range=0.2,
                                shear_range=0.15,zoom_range=[0.5,2],validation_split=0.2)
dataGen.fit(X_train)
train_generator = dataGen.flow(X_train, y_train_cat, batch_size=64, shuffle=True, 
                                                seed=2, save_to_dir=None, subset='training')
validation_generator = dataGen.flow(X_train, y_train_cat, batch_size=64, shuffle=True, 
                                                seed=2, save_to_dir=None, subset='validation')

print("model 1:")
model1 = m1.make_model()
model1 = m1.train_model(X_train, y_train_cat, model1)


print("model 2:")
model2 = m2.make_model()
model2 = m2.train_model(X_train, y_train_cat, model2)



print("model 3:")
model3 = m3.make_model()
model3 = m3.train_model(train_generator, validation_generator, model3)



epoch = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
plt.figure(figsize=(15, 2.5))
plt.plot(epoch, model1.history.history['loss'], label = "mnist_nn_cnn_1")
plt.plot(epoch, model2.history.history['loss'], label = "mnist_nn_cnn_2")
plt.plot(epoch, model3.history.history['loss'], label = "mnist_nn_cnn_3")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("Loss Function")
plt.legend()
plt.savefig("figure1.pdf", format="pdf")
plt.figure(figsize=(15, 2.5))
plt.plot(epoch, model1.history.history['val_accuracy'], label = "mnist_nn_cnn_1")
plt.plot(epoch, model2.history.history['val_accuracy'], label = "mnist_nn_cnn_2")
plt.plot(epoch, model3.history.history['val_accuracy'], label = "mnist_nn_cnn_3")
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.title("Validation Accuracy")
plt.legend()
plt.savefig("figure2.pdf", format="pdf")

test_model_1_loss, test_model_1_acc = model1.evaluate(X_test, y_test_cat, verbose=2)
test_model_2_loss, test_model_2_acc = model2.evaluate(X_test, y_test_cat, verbose=2)
test_model_3_loss, test_model_3_acc = model3.evaluate(X_test, y_test_cat, verbose=2)
plt.figure(figsize=(15, 2.5))
plt.scatter(model1.count_params(), test_model_1_acc, c="red", s=50)
plt.scatter(model2.count_params(), test_model_2_acc, c="blue", s=50)
plt.scatter(model3.count_params(), test_model_3_acc, c="green", s=50)

plt.xlabel("Memory Utilization")
plt.ylabel("Accuracy")
plt.savefig("figure3.pdf", format="pdf")

plt.show()
