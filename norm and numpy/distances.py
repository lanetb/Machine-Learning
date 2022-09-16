"""
Utilities for calculating distances and finding nearest neighbors.
(These are intended to be completed as numpy exercises.)

Author:
Date:
"""

from cmath import sqrt
import numpy as np


def distance_loop(x1, x2):
    """ Returns the Euclidean distance between the 1-d numpy arrays x1 and x2"""
    sum = 0
    for i in range(len(x1)):
            sum += (x1[i] - x2[i]) ** 2

    return np.sqrt(sum)


def nearest_loop(X, target):
    """ Return the index of the nearest point in X to target.
         Arguments:
          X      - An m x d numpy array storing m points of dimension d
         target - a length-d numpy array
    """
    min_dist = float('inf')
    index = 0
    for i in range(len(X)):
        dist = distance_loop(X[i], target)
        if dist < min_dist:
            min_dist = dist
            index = i


    return index


def distance(x1, x2):
    """ Returns the Euclidean distance between the 1-d numpy arrays x1 and x2"""
    return np.linalg.norm(x1 - x2)


def nearest(X, target):
    """ Return the index of the nearest point in X to target.
         Arguments:
          X      - An m x d numpy array storing m points of dimension d
         target - a length-d numpy array
    """

    return np.argmin(np.sqrt(np.sum(X - target, axis =1) ** 2))


def digit_demo():
    """Create a data set containing only 1's and 7's"""
    from sklearn.datasets import load_digits
    import matplotlib.pyplot as plt
    digits, labels = load_digits(return_X_y=True)

    digits_subset = None  # Suppress warnings...
    labels_subset = None

    # YOUR CODE HERE!
 

    print(digits_subset.shape)  # should be (361, 64)
    print(labels_subset.shape)  # should be (361,)
    plt.gray()
    plt.matshow(digits_subset[0, :].reshape(8, 8))  # should be a "1"
    plt.figure()
    plt.matshow(digits_subset[180, :].reshape(8, 8))  # should be "7"
    plt.show()


if __name__ == "__main__":
    digit_demo()
