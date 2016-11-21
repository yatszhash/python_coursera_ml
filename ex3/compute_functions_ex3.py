import numpy as np
import matplotlib.pyplot as plt

from ex2.compute_functions_ex2 import sigmoid_function, cost_function_reg, gradient_descent, optimize_with_solver


def display_data(X, example_width):
    m, n = X.shape
    example_hight = n / example_width
    imgs = X.map(lambda x: x.reshpe(example_hight, example_width))

    display_rows = np.floor(np.sqrt(m))
    display_cols = np.ceil(m / display_rows)
    fig = plt.subplot(m )
    plt.gray()

def opt_one_vs_all(X, Y, initial_theta, num_labels):

    X = np.c_[np.ones((X.shape[0], 1)), X]

    labels = np.arange(1, num_labels + 1)
    all_theta = np.zeros((num_labels, initial_theta.shape[0]))
    for k in labels:
        each_Y = np.equal(Y, k).astype(int)
        each_Y = each_Y.reshape(each_Y.shape[0], 1)

        lambda_ = 0.1
        each_theta = optimize_with_solver(X, each_Y,
                                                            initial_theta, lambda_)
        all_theta[k - 1, :] = each_theta.T

    return all_theta

def predict_one_vs_all(X, all_theta):
    X = np.c_[np.ones((X.shape[0], 1)), X]

    result = sigmoid_function(np.dot(X, all_theta.T))

    predicted = np.argmax(result, axis=1)

    predicted += 1

    return predicted.reshape(len(predicted), 1)

def predict(Theta1, Theta2, X):
    a1 = np.c_[np.ones((X.shape[0], 1)), X]

    z2 = np.dot(a1, Theta1.T)

    a2 = sigmoid_function(z2)

    a2 = np.c_[np.ones((z2.shape[0], 1)), a2]

    z3 = np.dot(a2, Theta2.T)

    h = sigmoid_function(z3)
    p = np.argmax(h, axis=1)
    p = p.reshape(len(p), 1)

    p += 1
    return p