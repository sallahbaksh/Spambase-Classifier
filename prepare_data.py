import numpy as np
import math
from sklearn.model_selection import train_test_split


def split_data():
    # read in the data
    X = np.array(np.genfromtxt('spambase.data', delimiter=','))

    # seed random number generator with zero prior and randomize data (shuffle by row)
    np.random.seed(0)
    np.random.shuffle(X)

    # create x matrix - all rows/column excluding last one
    x_mat = X[:, :-1]

    # create y matrix (last column) and convert from 1D to 2D matrix
    y_mat = np.reshape(X[:, -1], (-1, 1))

    # split x and y into training and testing data
    train_x, test_x = train_test_split(x_mat, test_size=0.333, random_state=1, shuffle=False)
    train_y, test_y = train_test_split(y_mat, test_size=0.333, random_state=1, shuffle=False)

    return train_x, test_x, train_y, test_y


def standardize_data(train_x, test_x, add_bias=False):
    mean = np.mean(train_x, axis=0)
    std = np.std(train_x, axis=0, ddof=1)

    # standardize training and testing data
    s_train_x = (train_x - mean) / std
    s_test_x = (test_x - mean) / std

    # add bias feature to standardized data
    if add_bias:
        train_bias = np.ones((s_train_x.shape[0], 1))
        s_train_x = np.concatenate((train_bias, s_train_x), 1)
        test_bias = np.ones((s_test_x.shape[0], 1))
        s_test_x = np.concatenate((test_bias, s_test_x), 1)

    return s_train_x, s_test_x
