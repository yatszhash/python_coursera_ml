import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ex1.compute_functions_ex1 import *

def main():
    #load datas
    ex1_data2= pd.read_csv("data/ex1data2.txt", header=None)

    raw_X = np.array(ex1_data2.iloc[:, 0:2])

    y = ex1_data2.iloc[:, 2]
    y = np.array(y).T
    m = len(y)
    y = y.reshape(m, 1)

    print("First 10 example from the dataset :")
    np.apply_along_axis(func1d=lambda row : print(
                                    "x = {} , y= {}".format(row[0:2], row[2])) or True,
                                    axis=0,  arr=np.c_[raw_X, y][:10, :])

    print("Normalizing Features ...\n")
    X, mu, sigma = feature_normalize(raw_X)

    #compute cost
    X = np.c_[np.ones((len(X), 1)), X]

    initial_theta = np.zeros((X.shape[1], 1))

    #compute gradient decent
    num_iters = 400
    alpha=0.01

    print("Running gradient descent ...\n")
    theta, J_hist = gradient_descent(X, y, initial_theta, alpha, num_iters)

    plt.plot(range(len(J_hist)), J_hist, "-b")
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost J')
    plt.show()

    print("Theta computed from gradient descent: \n {}".format(theta))

    #Estimate the price of a 1650 sq-ft, 3 br house
    test_X = np.array([[1650, 3]])
    norm_test_X = (test_X - mu) / sigma
    np.testing.assert_array_equal(norm_test_X.shape, test_X.shape)

    norm_test_X = np.c_[np.ones((test_X.shape[0], 1)), norm_test_X]

    estimated_y = np.dot(norm_test_X, theta)
    price  = estimated_y

    print("predicted price of a 1650 sq-ft, 3 br house "
          "...(using gradient desent\n {}".format(price))

    #Normal equation
    unnormalized_X = np.c_[np.ones((raw_X.shape[0], 1)), raw_X]
    theta2 = normal_equation(unnormalized_X, y)

    print("Theta comupted from the normal equations: \n {}".format(
        theta2
    ))

    estimated_y2 = np.dot(np.c_[np.ones((test_X.shape[0], 1)), test_X],
                                        theta2
                          )

    print("predicted price of a 1650 sq-ft, 3 br house "
          "...(using normal equation)\n {}".format(estimated_y))

if __name__ == "__main__":
    main()