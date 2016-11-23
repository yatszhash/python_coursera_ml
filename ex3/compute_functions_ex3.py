import numpy as np
import matplotlib.pyplot as plt

from ex2.compute_functions_ex2 import sigmoid_function, cost_function_reg, gradient_descent, optimize_with_solver

def display_sampling(X, sample_size=100):
    # randomly select 100 data
    np.random.seed(10)
    random_indices = np.random.randint(0, X.shape[0], sample_size)
    selected_X = X[random_indices, :]

    display_data(selected_X)

def display_data(X):
    m, n = X.shape

    n_col, n_row = 10, 10
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.title("plot images", size=16)
    #fig = plt.subplot(m )
    image_shape = (20, 20)

    for i, comp in enumerate(X):
        plt.subplot(n_row, n_col, i+1)
        vmax = comp.max() - comp.min()
        plt.imshow(comp.reshape(image_shape).T,
                   cmap=plt.get_cmap('gray'), interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks()
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)
    #plt.savefig("../../ex3_image")
    plt.show()

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