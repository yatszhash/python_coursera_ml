from unittest import TestCase

import unittest

import numpy as np

from ex1.compute_functions_ex1 import magic
from ex2.compute_functions_ex2 import sigmoid_function
from ex4.compute_functions_ex4 import nn_costfunction, sigmoid_gradient, get_epsilon, compute_numerical_gradient, \
    debug_initialize_wight, unroll_thetas, rand_initialize_weights


class TestEx4(TestCase):
    def test_nn_costfunction_wo_reg(self):
        il = 2 #input layer
        hl = 2  # hidden layer
        nl = 4 # number of labels
        nn = np.arange(1, 19) / 10 # nn_params
        nn = nn.reshape((1, nn.size))

        X = np.cos(np.array([[1, 2],
                                    [3, 4],
                                    [5, 6]]))
        y = np.array([[4, 2, 3]]).T
        lambda_ = 0
        cost = nn_costfunction(nn, il, hl, nl, X, y, lambda_)[0]

        expected = 7.4070
        np.testing.assert_almost_equal(cost, expected, decimal=3)


    def test_nn_costfunction_reg(self):
        il = 2  # input layer
        hl = 2  # hidden layer
        nl = 4  # number of labels
        nn = np.arange(1, 19) / 10  # nn_params
        nn = nn.reshape((1, nn.size))

        X = np.cos(np.array([[1, 2],
                             [3, 4],
                             [5, 6]]))
        y = np.array([[4, 2, 3]]).T
        lambda_ = 4
        cost = nn_costfunction(nn, il, hl, nl, X, y, lambda_)

        expected = 19.474
        np.testing.assert_almost_equal(cost, expected, decimal=3)

    def test_sigmoid_gradient(self):
        actual = sigmoid_gradient(np.vstack((np.array([[-1., - 2., - 3.]]), magic(3))))
        expected =np.array([[1.9661e-001, 1.0499e-001, 4.5177e-002],
                                        [3.3524e-004, 1.9661e-001, 2.4665e-003],
                                        [4.5177e-002, 6.6481e-003, 9.1022e-004],
                                        [1.7663e-002, 1.2338e-004, 1.0499e-001]])

        np.testing.assert_almost_equal(actual, expected, decimal=3)

    def test_get_epsilon(self):
        input_unit_num = 401
        output_unit_num = 10

        actual = get_epsilon(input_unit_num, output_unit_num)

        expected = 0.12

        self.assertLessEqual(actual, expected)

    def test_compute_numerical_gradient(self):
        theta = np.array([[1], [2]])

        J_fuction = lambda X : np.linalg.norm(X) ** 2
        expected = theta * 2

        actual = compute_numerical_gradient(J_fuction, theta)

        np.testing.assert_almost_equal(expected, actual)

    def test_back_propagation(self):
        self.check_nn_gradients()

    def check_nn_gradients(self, lambda_=0):
        input_layer_size = 3
        hidden_layer_size = 5
        num_labels = 3
        m = 5

        Theta1 = debug_initialize_wight(input_layer_size, hidden_layer_size)
        Theta2 = debug_initialize_wight(hidden_layer_size, num_labels)

        X = debug_initialize_wight(input_layer_size - 1, m)
        y = 1 + np.mod(np.arange(1, m + 1), num_labels)
        y = y.reshape(len(y), 1).astype(np.float64)

        nn_params = unroll_thetas(Theta1, Theta2)

        cost_func = lambda theta : nn_costfunction(theta, input_layer_size,
                                            hidden_layer_size, num_labels, X, y,
                                            lambda_)

        cost, grad = cost_func(nn_params)

        only_cost_func = lambda theta : cost_func(theta)[0]
        num_grad = compute_numerical_gradient(only_cost_func, nn_params)

        np.testing.assert_almost_equal(grad, num_grad, decimal=3)

        diff = np.linalg.norm(num_grad - grad) \
               / np.linalg.norm(num_grad + grad)

        self.assertLessEqual(diff, 1e-9)

    def test_rand_initialize_weights(self):
        input_unit_num = 2
        output_unit_num = 5

        actual_theta = rand_initialize_weights(input_unit_num,
                                               output_unit_num)
        expected_shape = (5, 3)

        self.assertEquals(actual_theta.shape, expected_shape)


if __name__ == '__main__':
    unittest.main()


