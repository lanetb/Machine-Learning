import mnist_nn_nonorm_1 as nonorm
import mnist_nn_with_norm_2 as sigmoidnorm
import mnist_nn_with_norm_relu_3 as relu

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers.core import Dense

mnist = tf.keras.datasets.mnist
(train_nums, train_labels), (test_nums, test_labels) = mnist.load_data()

model_1 = nonorm.make_model_nn_nonorm()
model_1, history1 = nonorm.train_model_nn_nonorm(train_nums, train_labels, model_1)

test_loss1, test_acc1 = model_1.evaluate(test_nums,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc1)

train_nums = train_nums / 255
test_nums = test_nums / 255

model_2 = sigmoidnorm.make_model_nn_with_norm()
model_2, history2= sigmoidnorm.train_model_nn_with_norm(train_nums, train_labels, model_2)

test_loss2, test_acc2 = model_2.evaluate(test_nums,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc2)

model_3 = relu.make_model_nn_with_norm_relu()
model_3, history3 = relu.train_model_nn_with_norm_relu(train_nums, train_labels, model_3)

test_loss3, test_acc3 = model_3.evaluate(test_nums,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc3)

epoch = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
plt.figure(figsize=(15, 2.5))
plt.plot(epoch, history1.history['loss'], label = "mnist_nn_nonorm_1")
plt.plot(epoch, history2.history['loss'], label = "mnist_nn_with_norm_2")
plt.plot(epoch, history3.history['loss'], label = "mnist_nn_with_norm_relu_3")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("Loss Function")
plt.legend()
plt.savefig("figure1.pdf", format="pdf")
plt.figure(figsize=(15, 2.5))
plt.plot(epoch, history1.history['val_accuracy'], label = "mnist_nn_nonorm_1")
plt.plot(epoch, history2.history['val_accuracy'], label = "mnist_nn_with_norm_2")
plt.plot(epoch, history3.history['val_accuracy'], label = "mnist_nn_with_norm_relu_3")
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.title("Validation Accuracy")
plt.legend()
plt.savefig("figure2.pdf", format="pdf")
plt.show()

plt.savefig("figures.pdf")



sevens = np.matrix([[]])
for i in range(len(train_nums)):
    if train_labels[i] == 7:
        sevens = np.append(sevens, train_nums[i])

print(sevens.shape)

#sevens = sevens / 255
#
#plt.figure(figsize=(10,10))
#for i in range(25):
#    plt.subplot(5,5,i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
#    plt.imshow(sevens[i], cmap=plt.cm.binary)
#    plt.xlabel("7")
#plt.show()
#sevens = sevens.flatten()
#
#
#predictions, p_model = relu.predict_nn_with_norm_relu(model_3, sevens, test_labels)
#averages = np.mean(predictions)
#
#classnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
#
#fig, ax = plt.subplots()
#im = ax.imshow(predictions)
#
## Show all ticks and label them with the respective list entries
#ax.set_xticks(np.arange(len(classnames)), labels="predicted")
#ax.set_yticks(np.arange(len(classnames)), labels="actual")
#
## Rotate the tick labels and set their alignment.
#plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#         rotation_mode="anchor")
#
## Loop over data dimensions and create text annotations.
#for i in range(len(classnames)):
#    for j in range(len(classnames)):
#        text = ax.text(j, i, averages[i, j],
#                       ha="center", va="center", color="w")
#
#ax.set_title("Harvest of local farmers (in tons/year)")
#fig.tight_layout()
#plt.show()
#