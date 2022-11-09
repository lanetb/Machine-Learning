"""
   MNist Dense NN Leaderboard for Gradescope
   Excepts 2 files:
     -- model file from keras
     -- a preprocessing file for manipulating the input data
"""
import unittest
import numpy as np

from gradescope_utils.autograder_utils.decorators import weight, number, leaderboard
import keras

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.models import load_model
import importlib
import inspect


class TestMNistNN(unittest.TestCase):
    

    @classmethod
    def setUpClass(cls):
        cls.pre = None

    def helper_load_preprocessor(self):
    
        self.preprocessor_filename = "cs445_mnist_densenn_pre"

        print('reading in module:', self.preprocessor_filename)
        self.pre = importlib.import_module(self.preprocessor_filename)
        if ('preprocess_data') not in [n for n,o in inspect.getmembers(self.pre)]:
            print("ERROR! Can not find preprocess_data in your cs445_mnist_densenn_pre.py file")

    @number("1.0")
    @leaderboard("accuracy","asc")
    def test_get_accuracy(self, set_leaderboard_value=None):
        """Check accuracy of your model"""

        self.helper_load_preprocessor()

        model = load_model("mnist_dense_model.h5")

        model.summary()
        
        X_test = np.load("data/MNIST_X_test.npy")


        y_test = np.load("data/MNIST_y_test.npy")


        (X_test, y_test) = self.pre.preprocess_data(X_test, y_test)
        
        score = model.evaluate(X_test, y_test,verbose=1)
        set_leaderboard_value(score[1])


if __name__ == '__main__':
    try:
        unittest.main(exit=False)
    except:
        print('in except from main')

