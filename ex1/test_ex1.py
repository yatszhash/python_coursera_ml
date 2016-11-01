import unittest
from ex1 import *
from ex1.compute_functions_ex1 import *
import numpy as np

class TestEx1(unittest.TestCase):

    def test_cost_function(self):
        X = np.c_[np.ones((4, 1)), np.array([2, 3, 4, 5])]
        y = np.array([7, 6, 5, 4])
        y = y.T.reshape(len(y), 1)
        theta = np.array([0.1, 0.2])
        theta = theta.T.reshape(len(theta), 1)
        answer = 11.9450
        np.testing.assert_almost_equal(compute_cost(X, y, theta), answer, decimal=2)

    def test_cost_function2(self):
        X = np.array([[2, 1, 3],
                            [7, 1, 9],
                            [1, 8, 1],
                            [3, 7, 4]])
        #X = np.c_[np.ones((4, 1)), X]
        y = np.array([[2], [5], [5], [6]])
        theta = np.array([0.4, 0.6, 0.8])
        theta = theta.T.reshape(len(theta), 1)
        answer = 5.2950
        np.testing.assert_almost_equal(compute_cost(X, y, theta), answer, decimal=3)

    def test_gradient_decent1(self):
        X = np.array([[1, 5], [1, 2], [1, 4], [1, 5]])
        y = np.array([1, 6, 4, 2]).T
        theta = np.array([0, 0]).T
        alpha = 0.01
        num_iters = 1000

        answer_theta = np.array([ 5.21475495, -0.57334591])
        answer_first_J = 5.9794
        answer_final_J = 0.85426

        final_theta, J_hist= gradient_descent(X=X, y=y, theta=theta, alpha=alpha,
                         num_iters=1000)

        np.testing.assert_array_almost_equal(final_theta, answer_theta, decimal=3)
        np.testing.assert_almost_equal(J_hist[0], answer_first_J, decimal=3)
        np.testing.assert_almost_equal((J_hist[-1]), answer_final_J, decimal=3)

    def test_gradient_decescent2(self):
        X = np.array([[1, 5], [1, 2]])
        y = np.array([1, 6]).T
        theta = np.array([0.5, 0.5]).T
        alpha = 0.1
        num_iters = 10

        answer_theta = np.array([1.70986, 0.19229])
        answer_first_J = 5.8853
        answer_final_J = 4.5117

        final_theta, J_hist = gradient_descent(X=X, y=y, theta=theta, alpha=alpha,
                                               num_iters=num_iters)

        np.testing.assert_array_almost_equal(final_theta, answer_theta, decimal=3)
        np.testing.assert_almost_equal(J_hist[0], answer_first_J, decimal=3)
        np.testing.assert_almost_equal((J_hist[-1]), answer_final_J, decimal=3)

    def test_gradient_descentMulti(self):
        X = np.array([[2, 1, 3],
                      [7, 1, 9],
                      [1, 8, 1],
                      [3, 7, 4]])
        # X = np.c_[np.ones((4, 1)), X]
        y = np.array([[2], [5], [5], [6]])

        theta = np.zeros((3, 1))
        alpha = 0.01
        num_iters = 100

        answer_theta = np.array([[0.23680],
                                                [0.56524],
                                                [0.31248]])
        answer_first_J = np.array([2.8229])
        answer_final_J = np.array([0.0017196])

        final_theta, J_hist = gradient_descent(X=X, y=y, theta=theta, alpha=alpha,
                                               num_iters=num_iters)

        np.testing.assert_array_almost_equal(final_theta, answer_theta, decimal=3)
        np.testing.assert_array_almost_equal(J_hist[0], answer_first_J, decimal=2)
        np.testing.assert_array_almost_equal((J_hist[-1]), answer_final_J, decimal=2)

    def test_feature_normalize1(self):
        X = np.array([[1], [2], [3]])
        ans_X_norm = [[-1], [0], [1]]
        ans_mu = 2
        ans_sigma = 1

        X_norm, mu, sigma = feature_normalize(X)

        np.testing.assert_array_almost_equal(X_norm, ans_X_norm, decimal=3)
        np.testing.assert_almost_equal(mu, ans_mu, decimal=3)
        np.testing.assert_almost_equal(sigma, ans_sigma, decimal=3)


    def test_feature_normalize2(self):
        X = magic(3)
        ans_X_norm = np.array([[ 1.13389,  -1.00000,   0.37796],
                                [-0.75593,   0.00000,   0.75593],
                                [-0.37796,   1.00000,  -1.13389]])

        ans_mu = np.array([5, 5, 5])
        ans_sigma = np.array([2.6458, 4., 2.6458])

        X_norm, mu, sigma = feature_normalize(X)

        np.testing.assert_array_almost_equal(X_norm, ans_X_norm, decimal=3)
        np.testing.assert_array_almost_equal(mu, ans_mu, decimal=3)
        np.testing.assert_array_almost_equal(sigma, ans_sigma, decimal=3)

if __name__ == '__main__':
    unittest.main()
