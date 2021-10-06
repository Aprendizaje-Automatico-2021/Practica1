import numpy as np
from numpy.lib import diff
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt

def read_data():
    """
    Reads the data of the file and return the result as a float
    """
    valores = read_csv("./src/ex1data2.csv", header=None).to_numpy()
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
    # The new value of theta
    NewTheta = Theta

    # Contains the hypotesis function of every row
    H = np.matmul(X, NewTheta)
    
    # diff
    Diff = H - Y
    
    # Calculate every new Theta of the matrix Theta
    for i in range(n):
        Prod = Diff * X[:, i]
        NewTheta[i] -= (alpha / m) * Prod.sum()

    return NewTheta

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
    # Add a column of 1's to X
    X = np.hstack([np.ones([m, 1]), X])
    n = np.shape(X)[1]   

    for i in range(n):
        aux = X[:, i]
        #print("X before: {}".format(aux))
        X[:, i] = normalize_X(aux)
        #print("X after: {}".format(aux))

    # Theta need to have the same values as the columns of X
    Theta = np.zeros(n)
    alpha = 0.0001

    # No. expermients
    exp = 1500
    # The X values for the graph
    axisX = np.arange(0, exp)
    # The Y values for the graph
    axisY = np.zeros(exp)

    for i in range(exp):
        # New Values of Theta
        Theta = new_Theta(m, n, alpha, Theta, X, Y)
        # Min J
        J = function_J(m, X, Y, Theta)
        axisY[i] = J.sum()

    #fig = plt.figure()
    #plt.title(r'$\alpha$: ' + str(alpha))
    #plt.xlabel('Number Iterations', c = 'green', size='15')
    #plt.ylabel(r'MIN J($\theta$)', c = 'red', size = '15')
    #plt.plot(axisX, axisY, "-", c='blue', label = r'J($\theta$)')
    #plt.legend(loc='upper right')
    #plt.show()
    return Theta

def new_normal_Theta(X, Y):
    #Theta = (X(trans) * X)^-1 * X(trans) * Y
    X_t = np.transpose(X)
    Prod = np.dot(X_t, X)
    Inv = np.linalg.pinv(Prod)
    b = np.dot(Inv, X_t)
    newTheta = np.matmul(b, Y)

    return newTheta

def normal_equation():
    valores = read_data()
    # Add all the rows and the col(len - 1)
    X = valores[:, :-1]
    # The -1 value add the col(len - 1)
    Y = valores[:, -1]
    # Row X
    m = np.shape(X)[0]
    # Cols X
    # Add a column of 1's to X
    X = np.hstack([np.ones([m, 1]), X])
    n = np.shape(X)[1]   

    for i in range(n):
        aux = X[:, i]
        #print("X before: {}".format(aux))
        X[:, i] = normalize_X(aux)
        #print("X after: {}".format(aux))

    Theta = np.zeros(n)

    # No. expermients
    exp = 1500
    # The X values for the graph
    axisX = np.arange(0, exp)
    # The Y values for the graph
    axisY = np.zeros(exp)

    Theta = new_normal_Theta(X, Y)

    return Theta

def hypotesis(Theta, X):
    H = np.matmul(X, Theta)

    return H.sum()

Theta = gradient()
NormalTheta = normal_equation()
X = [1, 1650, 3]
print("Hipotesis: ", hypotesis(Theta, X))
print("Normal hipotesis: ", hypotesis(NormalTheta, X))
print("Theta: ", Theta)
print("Theta shape: ", np.shape(Theta))
print("Normal Theta: ", NormalTheta)
print("Normal Theta shape: ", np.shape(NormalTheta))