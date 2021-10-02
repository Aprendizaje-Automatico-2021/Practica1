import numpy as np
from numpy.lib import diff
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_data():
    """
    Reads the data of the file and return the result as a float
    """
    valores = read_csv("./ex1data2.csv", header=None).to_numpy()
    return valores.astype(float)

def function_J(m, X, Y, Theta):
    """
    Calculates the cost J with vectors
    """
    X_Theta = np.dot(X, Theta)
    diff = X_Theta - Y

    return (1 / (2 * m)) * np.transpose(diff) * diff

def new_Theta(m, n, alpha, Theta, X, Y):
    """
    Calculates the new values of the Theta matrix
    """
    # The matrix for the hypotesis results of every theta
    H = np.zeros(m)
    #print(H)
    t_Theta = np.transpose(Theta)
    #print(t_Theta)

    for i in range(n):
        print(X[:, i])
        h = t_Theta * X[:, i]
        H[i] = np.sum(h)

    diff = H - Y 
    sum = (1 / m) * np.transpose(X) * diff
    return Theta - alpha * sum

def normalize_X(X):
    """
    Normalizes each value of the X array: xi = [xi - average(x)] / desviation_i
    """
    avg = np.mean(X)
    desv = np.std(X)
    # When all values are the same, the desv = 0
    if desv == 0:
        return X

    return [((x - avg) / desv).astype(float) for x in X]

def add_colum_ones():
    return 1

def gradient():
    valores = read_data()
    # Add all the rows and the col(len - 1)
    X = valores[:, :-1]
    # The -1 value add the col(len - 1)
    Y = valores[:, -1]
    # Row X
    m = np.shape(X)[0]
    # Cols X
    n = np.shape(X)[1]   
    # Add a column of 1's to X
    X = np.hstack([np.ones([m, 1]), X])

    for i in range(n):
        aux = X[:, i]
        #print("X before: {}".format(aux))
        X[:, i] = normalize_X(aux)
        #print("X after: {}".format(aux))

    Theta = np.zeros(m)
    alpha = 0.1

    fig = plt.figure()
    for i in range(2):
        # New Values of Theta
        Theta = new_Theta(m, n, alpha, Theta, X, Y)

        J = function_J(m, X, Y, Theta)
        plt.plot(i, np.sum(J), "x", c='red')

    plt.show()

gradient()