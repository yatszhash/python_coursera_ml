import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ex1.compute_functions_ex1 import *

def main():
    #load datas
    ex1_data1= pd.read_csv("data/ex1data1.txt", header=None)

    X = np.array(ex1_data1.iloc[ :, 0]).T

    y = ex1_data1.iloc[ :, 1]
    y = np.array(y).T
    m = len(y)
    y = y.reshape(m, 1)

    # plot
    plot_scatter(X, y)

    #compute cost
    X = np.c_[np.ones((len(X), 1)), X]

    initial_theta = np.zeros((X.shape[1], 1))

    cost = compute_cost(X, y, initial_theta)

    np.testing.assert_almost_equal(cost, 32.07, decimal=2)

    #compute gradient decent
    num_iters = 1500
    alpha=0.01
    theta, J_hist = gradient_descent(X, y, initial_theta, alpha, num_iters)

    fitting = np.dot(X, theta)
    plot_scatter(X[:, 1], y, fitting)

    test_X1 = np.array([[1, 3.5]])
    predict_result1 = np.dot(test_X1, theta)
    print("predict {} : result {}", test_X1, predict_result1 * 10000)

    test_X2 = np.array([[1, 7]])
    predict_result2 = np.dot(test_X2, theta)
    print("predict {} : result {}", test_X2, predict_result2 * 10000)



def plot_scatter(X, y, predictions=None):
    fig, ax = plt.subplots(1, 1)

    # In[12]:

    ax.scatter(X, y, color="r", marker="x")

    # In[13]:

    ax.set_xlabel("Protfit in $10, 000s")

    # In[14]:

    ax.set_ylabel("Population of City in 10, 000s")

    # In[15]:

    ax.set_title("Scatter plot of training data")

    if predictions is not None:
        ax.plot(X, predictions)
    plt.show()
    return

if __name__ == "__main__":
    main()