from keras import utils as kutils

def preprocess_data(X, y):
    """ Manipulate X and y from their given form in the MNIST dataset to
       whatever form your model is expecting."""
    
    # This function does nothing but return what it is sent  
    X = X / 255
    return X, y