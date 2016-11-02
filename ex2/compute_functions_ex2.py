import itertools
import numpy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def sigmoid_function(z):
    g = 1 / (1 + np.exp(-z))
    return g

def hypothesis_function(X, theta):
    linear_dot = np.dot(X, theta)
    h = sigmoid_function(linear_dot)
    return h.reshape(len(h), 1)

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

    grad_reg_term = (lambda_ / m) * theta
    grad_reg_term.reshape(len(grad_reg_term), 1)
    grad_reg_term[0] = 0

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


def plot_data(X, y, theta=None):
    plt.interactive(False)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    if theta is not None and theta.shape[0] <= 3:
        dec_x_lim = np.array([np.min(X[:, 1]) - 2,  np.max(X[:, 1]) + 2])

        dec_y_lim = (-1 / theta[2]) * (theta[1] * dec_x_lim + theta[0])

        ax.plot(dec_x_lim, dec_y_lim, "b-", visible=True, lw=2)

    elif theta is not None:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros((u.shape[0], v.shape[0]))

        for i, j in itertools.product(range(len(u)), range(len(v))):
            temp_x = map_feature(np.array([[u[i], v[j]]]), 6)
            #temp_x = np.c_[np.one, temp_x]
            z[i, j] = np.dot(temp_x, theta[1:])

        ax.contour(u, v, z, levels=[0], linewidths=2)

    ax.scatter(X[(y==0).nonzero()[0], 0],
               X[(y==0).nonzero()[0], 1],
               marker="o", c="yellow", s=50, label="not admitted", zorder=2)
    ax.scatter(X[(y == 1).nonzero()[0], 0],
               X[(y==1).nonzero()[0], 1],
               marker="+", c="black",  s=50, label="admitted", zorder=2)

    ax.set_title("Figure 1: Scatter plot of training data",)
    ax.set_xlabel("exam1 score")
    ax.set_ylabel("exam2 score")
    ax.legend()

    plt.show()
    plt.savefig("test.png")