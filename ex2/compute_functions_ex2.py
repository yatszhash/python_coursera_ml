import numpy as np
import pandas as pd


def sigmoid_function(z):
    g = 1 / (1 + np.exp(-z))
    return g

def hypothesis_function(X, theta):
    linear_dot = np.dot(X, theta)
    h = sigmoid_function(linear_dot)
    return h

def cost_function(X, y, theta):
    m = len(y)
    h = hypothesis_function(X, theta)
    j = (1 / m) * (-np.dot(y.T, np.log(h)) - np.dot((1 - y).T, np.log(1 - h)))

    grad = (1 / m) * np.dot((h - y).T, X).T

    return j, grad

def cost_function_reg(X, y, theta, lambda_):
    j, grad = cost_function(X, y, theta)

    m = X.shape[0]
    j_reg_term = (lambda_ / (2 * m)) * np.dot(theta.T, theta)
    #remove reguralization by theta[0]
    j_reg_term -= lambda_ / (2 * m) * theta[0] ** 2
    j_regularized = j + j_reg_term

    grad_reg_term = (lambda_ / (m)) * theta
    grad_reg_term[0, 0] = 0

    grad_regulatized = grad + grad_reg_term

    return j_regularized, grad_regulatized

#only for two fetures
def map_feature(X, degree):
    assert X.shape[1], 2
    degreed_get_poly_term = lambda x: get_poly_terms(x, degree)
    poly_X = map(degreed_get_poly_term, X)
    return np.array(list(poly_X))

def get_poly_terms(x, degree):
    poly_terms = np.array([])
    for partial_degree in range(degree + 1):
        for i in range(partial_degree + 1):
            poly_terms = np.append(poly_terms,
                                   ((x[0] ** (partial_degree - i)) * (x[1]  ** i)))
    return poly_terms