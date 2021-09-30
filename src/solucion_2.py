import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_data():
    """
    Read dthe data of the file and return the result as a float
    """
    valores = read_csv("./ex1data2.csv", header=None).to_numpy()
    return valores.astype(float)

def function_J(m, X, Y, Theta):
    np.dot(X, T)
    return 

def normalize_X(X):
    avg_1 = np.mean(X)

def gradient():
    valores = read_data()
    X = valores[:, 0:2]
    Y = valores[:, 2]
    m = len(Y)
    for x in X:
        x = normalize_X(x)

    Theta = np.zeros(m)
    J = function_J(m, X, Y, Theta)

gradient()