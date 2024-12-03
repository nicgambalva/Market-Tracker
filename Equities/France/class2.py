# THis cose is for the class 2 of Introduction to Machine Learning
# Copyright Nicolas Gamboa Alvarez / EDHEC Business School

# Libraries ------------------------------------------------

# Basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# Functions -----------------------------------------------

def gradient_descent(alpha, x, y, ep=0.0001, max_iter=10000):
    converged = False
    num_iter = 0
    m = x.shape[0] # number of samples
    theta0 = 1.0
    theta1 = 1.0 # initialise thetas
    J = sum([(theta0 + theta1*x[i] - y[i])**2 for i in range(m)]) # error
    while not converged:
        grad0 = 1.0/m * sum([(theta0 + theta1*x[i] - y[i]) for i in range(m)])
        grad1 = 1.0/m * sum([(theta0 + theta1*x[i] - y[i])*x[i] for i in range(m)])
        temp0 = theta0 - alpha * grad0
        temp1 = theta1 - alpha * grad1
        theta0 = temp0
        theta1 = temp1
        e = sum([(theta0 + theta1*x[i] - y[i])**2 for i in range(m)])
        if abs(J-e) <= ep:
            converged = True
        J = e
        num_iter += 1 # update iter
        if num_iter == max_iter:
            converged = True
    return theta0, theta1