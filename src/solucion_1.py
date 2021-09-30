import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def function_J(m, X, Y, theta0, theta1):
    """
    Calculate the function J(theta)
    """
    sum = 0
    for i in range(m):
        sum = sum + (hipotesis(X[i], theta0, theta1) - Y[i]) ** 2

    result = (1 / (2 * m)) * sum
    return  result.astype(float)

def hipotesis(x, theta0, theta1):
    """
    Calculate the hypothesis function h(x) = theta0 + theta1 * x
    """
    result = theta0 + theta1 * x
    return result.astype(float)

def diff(x, y, theta0, theta1):
    """
        return h(xi) - yi 
    """
    result = hipotesis(x, theta0, theta1) - y
    return result.astype(float)

def new_theta_0(m, X, Y, theta0, theta1, alpha):
    """
    Calculate the new value of theta0
    """
    sum = 0
    for i in range(m):
        sum = sum + diff(X[i], Y[i], theta0, theta1)

    result = theta0 - (alpha / m) * sum
    return  result.astype(float)

def new_theta_1(m, X, Y, theta0, theta1, alpha):
    """
    Calculate the new value of theta1
    """
    sum = 0
    for i in range(m):
        sum = sum + diff(X[i], Y[i], theta0, theta1) * X[i]

    result = theta1 - (alpha / m) * sum 
    return  result.astype(float)

def read_data():
    """
    Read dthe data of the file and return the result as a float
    """
    valores = read_csv("./ex1data1.csv", header=None).to_numpy()
    return valores.astype(float)

def gradient():
    """
    Main function to calculate the descent of the gradient
    """
    # Valores de muestra
    valores = read_data()
    X = valores[:, 0]
    Y = valores[:, 1]
    m  = len(X)
    
    theta0 = 0.0
    theta1 = 0.0
    alpha = 0.01
    # Current value of the function J(theta)
    curr_J = function_J(m, X, Y, theta0, theta1)
    min_J = curr_J
    min_t0 = 0.0
    min_t1 = 0.0

    i = 0
    for i in range(1500):
        # Calculate the new values to theta_0 and theta_1
        temp0  = new_theta_0(m, X, Y, theta0, theta1, alpha)
        temp1  = new_theta_1(m, X, Y, theta0, theta1, alpha)
        theta0 = temp0
        theta1 = temp1

        # Calculate the new cost of J function
        curr_J = function_J(m, X, Y, theta0, theta1)
        if curr_J < min_J:
            min_J = curr_J
            min_t0 = theta0
            min_t1 = theta1

        #print("Para theta0 = {} // theta1 = {}: J = {}".format(theta0, theta1, curr_J))

    # Graph drawing
    makeData = make_data(m, [-10, 10], [-1, 4], X, Y)
    #print(makeData)

    fig = plt.figure()
    #ax = Axes3D(fig)
    # np.logspace(-2, 3, 20)
    #ax.plot_surface(makeData[0], makeData[1], makeData[2], cmap='jet')
    plt.contour(makeData[0], makeData[1], makeData[2],np.logspace(-2, 3, 20), colors='blue')
    plt.plot(min_t0, min_t1, "x")
    plt.show()

    #plt.plot(X, Y, "x", c='red')
    #C = min_t0 + min_t1 * X
    #plt.plot(X, C, color="blue", linewidth=1.0, linestyle="solid")
    #plt.show()

def make_data(m, t0_range, t1_range, X, Y):
    """
    Calculate the matrix of Theta0 and Theta1. 
    """

    step = 0.1
    Theta0 = np.arange(t0_range[0], t0_range[1], step)
    Theta1 = np.arange(t1_range[0], t1_range[1], step)
    Theta0, Theta1 = np.meshgrid(Theta0, Theta1)
    Coste = np.empty_like(Theta0)
    for ix, iy in np.ndindex(Theta0.shape):
        Coste[ix][iy] = function_J(m, X, Y, Theta0[ix][iy], Theta1[ix][iy])

    return [Theta0, Theta1, Coste]

# main
gradient()