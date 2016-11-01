from ex2.compute_functions_ex2 import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from functools import partial

def main():
    ex2_data1 = pd.read_csv("data/ex2data1.txt", header=None)

    X = np.array(ex2_data1.iloc[:, 0:2])
    y = np.array(ex2_data1.iloc[:, 2]).T
    y = y.reshape((len(y), 1))

    m, n = X.shape

   #plotting
    '''
    fprintf(['Plotting data with + indicating (y = 1) examples and o '...
             'indicating (y = 0) examples.\n']);
    '''
    plot_data(X, y);
    #check non reguralized cost function
    X = np.c_[np.ones((m, 1)), X]
    initial_theta = np.zeros((n + 1, 1))

    anticipated_j = 0.693
    j, _ = cost_function(X, y, initial_theta)
    np.testing.assert_almost_equal(j, anticipated_j, decimal=3)

    #optimizing using (fminunc)
    target_func= lambda theta: cost_function(X=X, y=y, theta=theta)

    res = optimize.minimize(target_func, initial_theta,
                                    method="Nelder-Mead",
                                    jac=True,
                                    options={"maxiter": 400}
                                    )
    theta2 = res.x
    cost = res.fun
    print(theta2)
    np.testing.assert_almost_equal(cost, 0.203, decimal=2)

    #Estimate
    testX_1 = np.array([[45, 85]])
    anticipated_prob = 0.776
    testX_1 = np.c_[np.ones((testX_1.shape[0], 1)), testX_1]
    result_1 = sigmoid_function(np.dot(testX_1, theta2))
    np.testing.assert_almost_equal(result_1, anticipated_prob, decimal=3)

    pass

def plot_data(X, y, theta=None):
    fig, ax = plt.subplot(1, 1)

    ax.scatter(X[[y ==0], 0], X[[y==0], 1], "r+")
    ax.scatter(X[[y == 1], 0], X[[y==1], 0], "yo")

    plt.show()

if __name__ == "__main__":
    main()