import numpy as np
from pandas.io.parsers import read_csv

def function_J(m, valores, theta_0, theta_1):
    sum = 0
    for i in range(m):
        sum = sum + (valores[i][1] ** 2)

    return  (1 / 2 * m) * sum

def hipotesis(x, theta_0, theta_1):
    """
    Función de la hipotésis
    """
    return theta_0 + theta_1 * x

def diff(x, y, theta_0, theta_1):
    """
        return h(xi) - yi 
    """
    return hipotesis(x, theta_0, theta_1) - y

def new_theta_0(m, valores, theta_0, theta_1, alpha):
    sum = 0
    for i in range(m):
        sum = sum + diff(valores[i][0], valores[i][1], theta_0, theta_1)

    return  theta_0 - (alpha / m) * sum

def new_theta_1(m, valores, theta_0, theta_1, alpha):
    sum = 0
    for i in range(m):
        sum = sum + diff(valores[i][0], valores[i][1], theta_0, theta_1) * valores[i][0]

    return  theta_1 - (alpha / m) * sum

def gradient():
    valores = read_csv("./p1/ex1data1.csv" , header=None).to_numpy()
    m  = len(valores)
    
    theta_0 = 0.0
    theta_1 = 0.0
    alpha = 0.1
    curr_J = 0
    min = 1000000

    i = 0
    for i in range(1500):
        # Calculate the new cost of J function
        curr_J = function_J(m, valores, theta_0, theta_1)
        if curr_J < min: 
            min = curr_J

        # Calculate the new values to theta_0 and theta_1
        temp_0  = new_theta_0(m, valores, theta_0, theta_1, alpha)
        temp_1  = new_theta_1(m, valores, theta_0, theta_1, alpha)
        theta_0 = temp_0
        theta_1 = temp_1

    print(curr_J)

    # Graph drawing

gradient()