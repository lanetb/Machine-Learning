"""Pure Python Decision Tree Classifier.

Simple multi-class binary decision tree classifier.
Splits are based on entropy impurity.

Initial Author: Nathan Sprague
Modified by:
 molloykp -- added comments and switch impurity to entropy

"""
import numpy as np
from collections import namedtuple
import argparse

# Named tuple is a quick way to create a simple wrapper class...
Split_ = namedtuple('Split',
                    ['dim', 'pos', 'X_left', 'y_left', 'X_right', 'y_right'])


# This class does not require any student modification
# treat it as an immutable object without any methods
class Split(Split_):
    """
    Represents a possible split point during the decision tree creation process.

    Attributes:

        dim (int): the dimension along which to split
        pos (float): the position of the split
        X_left (ndarray): all X entries that are <= to the split position
        y_left (ndarray): labels corresponding to X_left
        X_right (ndarray):  all X entries that are > the split position
        y_right (ndarray): labels corresponding to X_right
    """
    pass


def split_generator(X, y):
    """
    Utility method for generating all possible splits of a data set
    for the decision tree construction algorithm.

    :param X: Numpy array with shape (num_samples, num_features)
    :param y: Numpy integer array with length num_samples
    :return: A generator for Split objects that will yield all
            possible splits of the data
    """

    # Loop over all of the dimensions.
    for dim in range(X.shape[1]):
        # Get the indices in sorted order so we can sort both  data and labels
        ind = np.argsort(X[:, dim])

        # Copy the data and the labels in sorted order
        X_sort = X[ind, :]
        y_sort = y[ind]

        # Loop through the midpoints between each point in the current dimension
        for index in range(1, X_sort.shape[0]):

            # don't try to split between equal points.
            if X_sort[index - 1, dim] != X_sort[index, dim]:
                pos = (X_sort[index - 1, dim] + X_sort[index, dim]) / 2.0

                # Yield a possible split.  Note that the slicing here does
                # not make a copy, so this should be relatively fast.
                yield Split(dim, pos,
                            X_sort[0:index, :], y_sort[0:index],
                            X_sort[index::, :], y_sort[index::])


def impurity(classes, y):
    """
    Return the impurity/entropy of the data in y

    :param y: Numpy array with class labels having shape (num_samples_in_node,)

    :return: A scalar with the entropy of the class labels
    """

    unique, count = np.unique(y, return_counts=True)
    return np.sum(-np.multiply((count/len(y)), np.log2(count/len(y))))


def weighted_impurity(classes, y_left, y_right):
    """
    Weighted entropy impurity for a possible split.
    :param classes: list of the classes
             y_left: class labels for the left node in the split
             y_right: class labels for the right node in the spit

    :return: A scalar with the weighted entropy
    """
    left = impurity(classes, y_left)
    right = impurity(classes, y_right)
    left_plus_right = len(y_left) + len(y_right)

    return np.add((np.multiply(len(y_left)/left_plus_right, left)),
                  (np.multiply(len(y_right)/left_plus_right, right)))


class DecisionTree:
    """
    A binary decision tree classifier for use with real-valued attributes.

    Attributes:
        classes (set): The set of integer classes this tree can classify.
    """

    def __init__(self, max_depth=np.inf):
        """
        Decision tree constructor.

        :param max_depth: limit on the tree depth.
                          A depth 0 tree will have no splits.
        """
        self.max_depth = max_depth
        self.depth = 0
        self.root = Node()

    def find_best_split(self, classes,X, y):
        splits = split_generator(X, y)
        best_info_gain = 0
        best_split = None

        for i in splits:
            entropy = impurity(classes, y)
            w_entropy = weighted_impurity(classes, i.y_left, i.y_right)

            info_gain = entropy - w_entropy

            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_split = i
        
        return best_split

    def tree_build(self, X, y, classes):
        if self.get_depth() == self.max_depth or len(set(y)) == 1:
            idx, counts = 0, 0
            leaf = Node()
            leaf.left = None
            leaf.right = None
            leaf.split = None
            idx, counts = np.unique(y, return_counts = True)
            leaf.predicted_class = idx[counts.argmax()]
            return leaf
        else:
            root = Node()
            root.split = self.find_best_split(classes, X, y)
            self.depth = self.get_depth() + 1
            root.left = self.tree_build(root.split.X_left, root.split.y_left, classes)
            root.right = self.tree_build(root.split.X_right, root.split.y_right, classes)
            return root

        

    def fit(self, X, y):
        """
        Construct the decision tree using the provided data and labels.

        :param X: Numpy array with shape (num_samples, num_features)
        :param y: Numpy integer array with length num_samples
        """

        # create a python set with all possible class labels

        self.classes = set(y)
        self.root = self.tree_build(X, y, self.classes)
        
        
    def single_predict(self, x):
        node = self.root
        if self.max_depth == 0:
            return node.predicted_class

        while node.left:
            if x[node.split.dim] <= node.split.pos:
                node = node.left
            else:
                node = node.right
    
        return node.predicted_class


    def predict(self, X):
        """
        Predict labels for a data set by finding the appropriate leaf node for
        each input and using the majority label as the prediction.

        :param X:  Numpy array with shape (num_samples, num_features)
        :return: A length num_samples numpy array containing predicted labels.
        """
        return np.array([self.single_predict(inputs) for inputs in X])
        

    def get_depth(self):
        """
        :return: The depth of the decision tree.
        """
        return self.depth


class Node:
    """
    It will probably be useful to have a Node class.  In order to use the
    visualization code in draw_trees, the node class must have three
    attributes:

    Attributes:
        left:  A Node object or Null for leaves.
        right - A Node object or Null for leaves.
        split - A Split object representing the split at this node,
                or Null for leaves
    """

    def __init__(self):
        self.left = None
        self.right = None
        self.split = None
        self.predicted_class = None


def tree_demo():
    import draw_tree
    X = np.array([[0.88, 0.39],
                  [0.49, 0.52],
                  [0.68, 0.26],
                  [0.57, 0.51],
                  [0.61, 0.73]])
    y = np.array([1, 0, 0, 0, 1])
    tree = DecisionTree()
    tree.fit(X, y)
    draw_tree.draw_tree(X, y, tree)


def parse_args():
    parser = argparse.ArgumentParser(description='Decision Tree modeling')

    parser.add_argument('--inputFile', action='store',
                        dest='input_filename', default="", required=False,
                        help='csv data file.  Last column is the class label')

    parser.add_argument('--depthLimit', action='store', type=int,
                        dest='depth_limit', default=-1, required=False,
                        help='max depth of the decision tree')

    parser.add_argument('--testDataFile', action='store',
                        dest='test_data_filename', default="", required=False,
                        help='data file with test data')

    parser.add_argument('--treeModelFile', action='store',
                        dest='tree_model_file', default="", required=False,
                        help='output of the learned model/tree')

    parser.add_argument('-demoFlag', action='store_true',
                        dest='demo_flag',
                        help='run demo data hardcoded into program')

    return parser.parse_args()


def main():
    parms = parse_args()

    if parms.demo_flag:
        tree_demo()
    else:
        # read in training and test data
        # compute model on training data
        # run test data
        # optionally print out tree model
        pass


if __name__ == "__main__":
    main()
