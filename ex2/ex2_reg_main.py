from ex2.compute_functions_ex2 import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from functools import partial

def main():
    #TODO change data1 -> data2
    ex2_data2 = pd.read_csv("data/ex2data2.txt", header=None)

    X = np.array(ex2_data2.iloc[:, 0:2])
    y = np.array(ex2_data2.iloc[:, 2]).T
    y = y.reshape((len(y), 1))

    m, n = X.shape

   #plotting
    '''
    fprintf(['Plotting data with + indicating (y = 1) examples and o '...
             'indicating (y = 0) examples.\n']);
    '''
    #plot_data(X, y)
    #check non reguralized cost function

    mapped_X = map_feature(X, 6)
    #mapped_X = np.c_[np.ones((m, 1)), mapped_X]

    initial_theta = np.zeros((mapped_X.shape[1], 1))

    num_iters = 400
    expected_j = 0.693
    lambda_ = 1

    alpha = 1

    j, _= cost_function_reg(mapped_X, y, initial_theta, lambda_)
    np.testing.assert_almost_equal(j, expected_j, decimal=3)
    theta, _ = gradient_descent(mapped_X, y, initial_theta, alpha,
                                lambda_, num_iters)

    plot_data(X, y, theta)

    #TODO fix optimization (because inaprropriate of result)
    #optimizing using (fminunc)
    target_func= lambda theta: cost_function_reg(X=mapped_X, y=y,
                                                 theta=theta, lambda_=lambda_)

    res = optimize.minimize(target_func, initial_theta,
                                    method="Nelder-Mead",
                                    jac=True,
                                    options={"maxiter": 400}
                                    )
    theta2 = res.x
    cost = res.fun
    print(theta2)
    #np.testing.assert_almost_equal(cost, 0.203, decimal=2)

    plot_data(X, y, theta2)

    #TODO remove


if __name__ == "__main__":
    main()
