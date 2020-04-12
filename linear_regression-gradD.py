import numpy as np
import random
import matplotlib.pyplot as plt
from utils import get_color, get_XY, get_plot, get_train_test_split

np.random.seed(seed=150)
a = np.random.randint(0,5)
b = np.random.randint(0,5)

# Hyper params
epochs = 2000
lr = 0.00002
mode = 'batch'

# Set up X data points
N = 300
X = np.linspace(0,20,N)

# Equation of Y
noise = np.random.randn(N) * 6
Y = a * X + b + noise

# Set up X matrix
ones = np.ones(N)
X = np.vstack((ones, X)).T

# Random init of weights
w = np.random.randn(2)

# Split data set
X_train, y_train, X_test, y_test = get_train_test_split(X, Y)

weights = []
train_error = []
test_error =[]
for i in range(epochs):
    
    X_gd, Y_gd = get_XY(mode, X_train, y_train, i)

    y_pred = np.dot(X_gd, w)

    # Update weights
    w = w - lr * np.dot((y_pred-Y_gd).T, X_gd)
    
    # Save weights
    weights.append(w)
      
    # Calculate train error
    train_difference = np.dot(X_train, w.T) - y_train
    train_sq_err = np.dot(train_difference.T, train_difference)/y_train.shape[0]

    # Calculate test error
    test_difference = np.dot(X_test, w.T) - y_test
    test_sq_err = np.dot(test_difference.T, test_difference)/y_test.shape[0]

    # Calculate train error
    train_error.append(train_sq_err)
    test_error.append(test_sq_err) 

get_plot(X, Y, X_test, y_test, train_error, test_error, weights, mode)        
print(weights[-1], '\tActual: ', b,', ', a)