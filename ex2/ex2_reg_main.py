from ex2.compute_functions_ex2 import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from functools import partial

def main():
    ex2_data2 = pd.read_csv("data/ex2data2.txt", header=None)

    X = np.array(ex2_data2.iloc[:, 0:2])
    y = np.array(ex2_data2.iloc[:, 2]).T
    y = y.reshape((len(y), 1))

    m, n = X.shape

    #check non reguralized cost function

    mapped_X = map_feature(X, 6)

    initial_theta = np.zeros((mapped_X.shape[1], 1))

    num_iters = 1000
    expected_j = 0.693
    lambda_ = 1

    alpha = 1

    j, _= cost_function_reg(mapped_X, y, initial_theta, lambda_)
    np.testing.assert_almost_equal(j, expected_j, decimal=3)
    theta, _ = gradient_descent(mapped_X, y, initial_theta, alpha,
                                lambda_, num_iters)

    plot_data(X, y, theta)

    #optimizing using (fminunc)
    target_func= lambda theta: cost_function_reg(X=mapped_X, y=y,
                                                 theta=theta, lambda_=lambda_)
    cost_func_for_opt = lambda theta : target_func(theta)[0][0, 0]
    grad_func_for_opt = lambda theta : target_func(theta)[1]

    lambda_ = 0.1
    res = optimize.minimize(cost_func_for_opt, initial_theta,
                                    method="L-BFGS-B",
                                    #jac=grad_func_for_opt,
                                    #options={'disp': True}
                                    )
    theta2 = res.x
    cost = res.fun
    theta2 = theta2.reshape(len(theta2), 1)

    plot_data(X, y, theta2)

    predicted1 = predict(mapped_X, theta)
    score1 = (predicted1 == y).astype(int)
    print("gradient descent train accuracy : {}  %".format(
        np.mean(score1) * 100
    ))

    predicted2 = predict(mapped_X, theta2)
    score2 = (predicted2 == y).astype(int)
    print("analytic optimization train accuracy : {} %".format(
        np.mean(score2) * 100)
    )


if __name__ == "__main__":
    main()
