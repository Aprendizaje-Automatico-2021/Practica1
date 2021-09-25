import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt

def function_J(m, X, Y, theta0, theta1):
    """
    Calculate the function J(theta)
    """
    sum = 0
    for i in range(m):
        sum = sum + (hipotesis(X[i], theta0, theta1) - Y[i]) ** 2

    result = (1 / 2 * m) * sum
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
    curr_J = 0

    i = 0
    # Calculate the new cost of J function
    curr_J = function_J(m, X, Y, theta0, theta1)
    print("Para theta0 = {} // theta1 = {}: J = {}".format(theta0, theta1, curr_J))

    for i in range(1500):
        # Calculate the new values to theta_0 and theta_1
        temp0  = new_theta_0(m, X, Y, theta0, theta1, alpha)
        temp1  = new_theta_1(m, X, Y, theta0, theta1, alpha)
        theta0 = temp0
        theta1 = temp1

        # Calculate the new cost of J function
        curr_J = function_J(m, X, Y, theta0, theta1)
        print("Para theta0 = {} // theta1 = {}: J = {}".format(theta0, theta1, curr_J))

# main
gradient()