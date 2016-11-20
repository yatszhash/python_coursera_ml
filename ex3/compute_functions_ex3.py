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
        #each_theta = gradient_descent(X, each_Y, initial_theta,
        #                              0.5, 0.1, num_iters=10000)[0]

        all_theta[k - 1, :] = each_theta.T

    #vfunc = np.vectorize(each_gradient_descent)
    #all_theta = np.apply_along_axis(each_gradient_descent, 0, labels)

    return all_theta

def predict_one_vs_all(X, all_theta):
    X = np.c_[np.ones((X.shape[0], 1)), X]

    result = sigmoid_function(np.dot(X, all_theta.T))

    predicted = np.argmax(result, axis=1)

    predicted += 1

    return predicted.reshape(len(predicted), 1)

def predict(Theta1, Theta2):
    pass