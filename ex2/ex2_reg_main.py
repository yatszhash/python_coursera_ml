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
    mapped_X = np.c_[np.ones((m, 1)), mapped_X]

    initial_theta = np.zeros((mapped_X.shape[1], 1))

    expected_j = 0.693
    lambda_ = 500
    j, _ = cost_function_reg(mapped_X, y, initial_theta, lambda_)
    np.testing.assert_almost_equal(j, expected_j, decimal=3)
    #TODO theta with decent gradient and compare with optimizer's theta

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

    #Estimate
    testX_1 = np.array([[45, 85]])
    anticipated_prob = 0.776
    testX_1 = np.c_[np.ones((testX_1.shape[0], 1)), testX_1]
    result_1 = sigmoid_function(np.dot(testX_1, theta2))
    np.testing.assert_almost_equal(result_1, anticipated_prob, decimal=3)

    pass


if __name__ == "__main__":
    main()