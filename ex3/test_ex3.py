from unittest import TestCase

import unittest

import ex2.compute_functions_ex2 as ex2
import numpy as np

import ex3.compute_functions_ex3 as ex3

from ex1.compute_functions_ex1 import magic


class TestEx3(TestCase):
    def test_cost_function_reg(self):
        X = np.arange(1, 16).reshape(3, 5).T / 10.0
        X = np.c_[np.ones((5, 1)), X]
        y = np.array([[1.0, 0.0, 1.0, 0.0, 1.0]] ) >= 0.5
        y = y.T
        theta = np.array([[-2.0, -1.0, 1.0, 2.0]]).T
        lambda_ = 3.0

        j, grad = ex2.cost_function_reg(X, y, theta, lambda_)

        answer_j = np.array([[2.5348]])
        answer_grad = np.array([[0.14656,
                                                -0.54856,
                                                0.72472,
                                                1.39800]]).T

        #np.testing.assert_almost_equal(j, answer_j, decimal=3)
        np.testing.assert_array_almost_equal(grad, answer_grad, decimal=3)

    def test_predict_one_vs_all(self):
        all_theta = np.array([[1, -6, 3],
                                        [-2, 4, -3]])

        X = np.array([[1, 7],
                                [4, 5],
                                [7, 8],
                                [1, 4]])

        actual = ex3.predict_one_vs_all(X, all_theta)

        expected = np.array([[1],
                                            [2],
                                            [2],
                                            [1]])

        np.testing.assert_array_equal(actual, expected)

    def test_opt_one_vs_all(self):
        X = np.r_[magic(3),
                    np.sin(np.arange(1, 4)).reshape(1, 3),
                    np.cos(np.arange(1, 4)).reshape(1, 3)]
        y = np.array([[1, 2, 2, 1, 3]])
        y = y.T

        num_labels = 3

        theta = np.ones((X.shape[1] + 1, 1)) * 0.1
        actual = ex3.opt_one_vs_all(X, y, theta, num_labels)

        expected = np.array([
                                        [-0.559478, 0.619220, -0.550361, -0.093502],
                                        [-5.472920, -0.471565, 1.261046, 0.634767],
                                        [0.068368, -0.375582, -1.652262, -1.410138]
                                        ])
        np.testing.assert_array_almost_equal(actual, expected, decimal=3)

    def test_predict(self):
        Theta1 = np.sin(np.arange(0, 5.9, 0.5)).reshape(3, 4).T
        Theta2 = np.sin(np.arange(0, 5.9, 0.3)).reshape(5, 4).T
        X = np.sin(np.arange(1, 17)).reshape(2, 8).T
        actual_p = ex3.predict(Theta1, Theta2, X)

        expected_p = np.array([[4],
                                            [1],
                                            [1],
                                            [4],
                                            [4],
                                            [4],
                                            [4],
                                            [2]])

        np.testing.assert_array_equal(actual_p, expected_p)

if __name__ == "__main__":
    unittest.main()