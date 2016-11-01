import numpy as np
import unittest

from ex2.compute_functions_ex2 import *
from ex1.compute_functions_ex1 import magic

class TestEx2(unittest.TestCase):

    def test_sigmoid_function1(self):
        X_1 = np.array([[0]])
        X_2 = np.zeros((2, 3))

        answer_1 = np.array([[0.5]])
        answer_2 = np.ones((2, 3)) * 0.5

        np.testing.assert_array_equal(sigmoid_function(X_1),
                                                        answer_1)
        np.testing.assert_array_equal(sigmoid_function(X_2),
                                                        answer_2)

    def test_sigmoid_function2(self):
        X_1 = np.array([[1200000]])
        X_2 = np.array([[-25000]])
        X_3 = np.array([[4, 5, 6]])
        X_4 = magic(3)
        X_5 = np.eye(2)

        answer_1 = np.array([[1]])
        answer_2 = np.zeros((1, 1))
        answer_3 = np.array([[ 0.9820, 0.9933, 0.9975]])
        answer_4 = np.array([[0.9997, 0.7311, 0.9975],
                                            [0.9526, 0.9933, 0.9991],
                                            [0.9820, 0.9999, 0.8808]])
        answer_5 = np.array([[0.7311, 0.5000],
                                        [0.5000, 0.7311]])
        
        np.testing.assert_array_equal(sigmoid_function(X_1),
                                                        answer_1)
        np.testing.assert_array_equal(sigmoid_function(X_2),
                                                        answer_2)
        np.testing.assert_array_almost_equal(sigmoid_function(X_3),
                                                        answer_3, decimal=3)
        np.testing.assert_array_almost_equal(sigmoid_function(X_4),
                                                        answer_4, decimal=3)
        np.testing.assert_array_almost_equal(sigmoid_function(X_5),
                                                        answer_5, decimal=3)

    def test_cost_function1(self):
        X = np.c_[np.ones((3, 1)), magic(3)]
        y = np.array([[1, 0, 1]]).T
        theta = np.array([[-2, -1, 1, 2]]).T
        j, grad = cost_function(X, y, theta)

        answer_j = 4.6832
        answer_grad =   np.array([[0.31722,
                                                0.87232,
                                                1.64812,
                                                2.23787]]).T

        np.testing.assert_almost_equal(j,  answer_j, decimal=3)
        np.testing.assert_array_almost_equal(grad, answer_grad, decimal=3)

    def test_cost_function_reg1(self):
        X = np.c_[np.ones((3, 1)), magic(3)]
        y = np.array([[1, 0, 1]]).T
        theta = np.array([[-2, -1, 1, 2]]).T
        lambda_ = 3

        j, grad = cost_function_reg(X, y, theta, lambda_)

        answer_j = 7.6832
        answer_grad = np.array([[0.31722,
                                                -0.12768,
                                                2.64812,
                                                4.23787]]).T

        np.testing.assert_almost_equal(j, answer_j, decimal=3)
        np.testing.assert_array_almost_equal(grad, answer_grad, decimal=3)


    def test_map_feature(self):
        X = np.array([[1, 2], [1, 2]])
        degree = 3
        expected_row = np.array([1,
                            1, 2,
                            (1 ** 2) , (1 ** 1) * (2 ** 1),  (2 ** 2) ,
                            (1 ** 3 ), (1 ** 2) * (2 ** 1), (1 ** 1) * (2 ** 2), (2 ** 3) ])
        actual = map_feature(X, degree)

        np.testing.assert_array_equal(actual[0], expected_row)
        np.testing.assert_array_equal(actual[1], expected_row)

if __name__ == "__main__":
    unittest.main()