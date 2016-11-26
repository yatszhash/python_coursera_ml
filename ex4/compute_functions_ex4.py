import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import label_binarize

from ex2.compute_functions_ex2 import sigmoid_function
from ex3.compute_functions_ex3 import nn_h


def nn_costfunction(nn_params, input_layer_size,
                                    hidden_layer_size,
                                    num_labels,
                                    X, raw_Y, lambda_):
    m = X.shape[0]

    Theta1, Theta2 = to_Thetas(nn_params, input_layer_size,
                               hidden_layer_size, num_labels)

    Y = to_binary_matrix(raw_Y, num_labels)

    forwards = nn_h(Theta1, Theta2, X)
    h = forwards["a3"]


    cost_matrix = (1.0 / m) * (- np.multiply(Y, np.log(h))
                                            -np.multiply(- Y + 1,  np.log(- h + 1)))

    cost = np.sum(cost_matrix)

    coef = (1.0 * lambda_) / (2.0 * m)
    reg_term = coef * (np.sum(np.power(Theta1, 2))
                       + np.sum(np.power(Theta2, 2)))
    #remove regulation for bias term
    reg_term -= coef * (np.sum(Theta1[:, 0] ** 2) + np.sum(Theta2[:, 0] ** 2))

    Theta1_grad, Theta2_grad = back_propagation(Theta1, Theta2,
                                                forwards, Y)

    unbias_Theta1 = np.copy(Theta1)
    unbias_Theta2 = np.copy(Theta2)

    unbias_Theta1[:, 0] = 0
    unbias_Theta2[:, 0] = 0

    Theta1_grad = Theta1_grad + (lambda_ / m) *unbias_Theta1
    Theta2_grad = Theta2_grad + (lambda_ / m) * unbias_Theta2

    grad = unroll_thetas(Theta1_grad, Theta2_grad)

    return cost + reg_term, grad

def back_propagation(Theta1, Theta2, forwards, Y):
    m = Y.shape[0]
    diff_3 = forwards["a3"] - Y

    diff_2 = np.dot(diff_3, Theta2[:, 1:]) * sigmoid_gradient(forwards["z2"])

    delta_2 = np.dot(diff_3.T, forwards["a2"])
    delta_1 = np.dot(diff_2.T, forwards["a1"])

    Theta1_grad = (1 / m) * delta_1
    Theta2_grad = (1 / m) * delta_2

    return Theta1_grad, Theta2_grad


def compute_numerical_gradient(J_function, theta):
    num_grad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)

    epsilon = 1e-4

    for i in np.arange(num_grad.shape[1]):
        perturb[0, i] = epsilon

        loss1 = J_function(theta - perturb)
        loss2 = J_function(theta + perturb)

        num_grad[0, i] = (loss2 - loss1) / (2 * epsilon)
        perturb[0, i] = 0

    return num_grad

def nn_optimize_with_solver(initial_nn_params, input_layer_size,
                            hidden_layer_size, num_labels,
                             X, y, lambda_):

    target_func = lambda nn_params: nn_costfunction(initial_nn_params,
                                                     input_layer_size,  hidden_layer_size,
                                                     num_labels, X, y, lambda_)[0].flatten()
    options = {"disp" : True, "maxiter" : 400}
    res = minimize(target_func, initial_nn_params,
                   method="L-BFGS-B", jac=False,  options=options)

    new_nn_params = res.x
    print(res.fun)
    return new_nn_params

def debug_initialize_wight(input_num, output_num):
    W = np.zeros((output_num, 1 + input_num))

    W = np.sin(np.arange(1, W.size + 1))\
            .reshape((W.shape[1], W.shape[0])).T / 10

    return W

def rand_initialize_weights(input_unit_num, output_unit_num):
    epsilon_init = 0.12
    return np.random.uniform(size=(output_unit_num, 1 + input_unit_num)) \
           * 2 *  epsilon_init - epsilon_init

def sigmoid_gradient(z):
    return sigmoid_function(z) * (1 - sigmoid_function(z))

def get_epsilon(input_unit_num, oupput_unit_num):
    return np.sqrt(6) / (np.sqrt(input_unit_num) + np.sqrt(oupput_unit_num))

def unroll_thetas(Theta1, Theta2):
    nn_params = np.c_[Theta1.flatten(1).reshape(1, Theta1.size),
                      Theta2.flatten(1).reshape(1, Theta2.size)]
    return nn_params


def to_Thetas(nn_params, input_layer_size, hidden_layer_size, num_labels):
    Theta1 = nn_params[0, : hidden_layer_size * (input_layer_size + 1)]\
                        .reshape(input_layer_size + 1, hidden_layer_size).T

    Theta2 = nn_params[0, hidden_layer_size * (input_layer_size + 1):]\
                            .reshape(hidden_layer_size + 1, num_labels).T

    return Theta1, Theta2

def to_binary_matrix(Y, num_labels):
    return label_binarize(Y, np.arange(1, num_labels + 1))
