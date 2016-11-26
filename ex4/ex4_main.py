import numpy as np
import pandas as pd

from ex3.compute_functions_ex3 import predict, display_sampling
from ex4.compute_functions_ex4 import nn_costfunction, rand_initialize_weights, unroll_thetas, nn_optimize_with_solver, \
    to_Thetas


def main():
    #already trained params
    nn_params = pd.read_csv("data/ex4nn_params.csv", header=None)\
                            .as_matrix()
    nn_params = nn_params.reshape((1, nn_params.size))

    X = pd.read_csv("data/ex4data1_X.csv").as_matrix()
    y = pd.read_csv("data/ex4data1_Y.csv").as_matrix()

    m = y.shape[0]

    input_layer_size = 400
    hidden_layer_size = 25
    num_labels = 10

    # =====part 1: visualize data======
    #display_sampling(X)

    # =====part 2: predict =========
    print("FeedForward Using Neural Network ...")
    lambda_ = 0

    actual_J = nn_costfunction(nn_params, input_layer_size, hidden_layer_size,
                        num_labels, X, y, lambda_)[0]
    expected_J = 0.287629

    np.testing.assert_almost_equal(actual_J, expected_J, decimal=3)

    print("cost function works well")

    #=====part3 : cost with reg =======
    lambda_ = 1
    actual_J = nn_costfunction(nn_params, input_layer_size, hidden_layer_size,
                               num_labels, X, y, lambda_)[0]
    expected_J = 0.383770

    np.testing.assert_almost_equal(actual_J, expected_J, decimal=3)

    print("cost function works well with regularization")

    #=== part4 : initializing parameters =====

    print("initializing neural network parameters ....\n")

    initial_Theta1 = rand_initialize_weights(input_layer_size,
                                             hidden_layer_size)

    initial_Theta2 = rand_initialize_weights(hidden_layer_size,
                                             num_labels)

    initial_nn_params = unroll_thetas(initial_Theta1, initial_Theta2)

    #=======part 5 : Implements Regularization ========

    print("checking backpropagation (w / Regularization) ... \n")

    lambda_ = 3
    debug_J = nn_costfunction(nn_params, input_layer_size,
                              hidden_layer_size, num_labels, X, y, lambda_)[0]

    expected_debug_J = 0.576051

    np.testing.assert_almost_equal(debug_J, expected_debug_J, decimal=3)

    print("regularization works well")

    #=======part 6 : training nn=======

    print("\nTraining Neural Network....")

    lambda_ = 0.01

    #TODO this doesn't work. Each element of Theta is almost same
    nn_params = nn_optimize_with_solver(initial_nn_params, input_layer_size,
                                                                hidden_layer_size, num_labels,
                                                                X, y, lambda_)

    nn_params = nn_params.reshape(1, nn_params.size)

    Theta1, Theta2 = to_Thetas(nn_params, input_layer_size,
                               hidden_layer_size, num_labels)

    resub_y = predict(Theta1, Theta2, X)

    resub_error_rate = np.sum(np.equal(resub_y, y).astype(int)) * 100 / m
    expected_error_rate = 95.3
    print("actual error rate : {} \n expected error rate : {}".format(
         resub_error_rate, expected_error_rate
    ))

    if resub_error_rate >= expected_error_rate:
         print("This classifier works well")
if __name__ == '__main__':
    main()