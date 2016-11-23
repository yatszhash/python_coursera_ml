import numpy as np
import pandas as pd

from ex3.compute_functions_ex3 import predict, display_sampling


def main():
    #already trained params
    theta1 = pd.read_csv("data/ex3_theta1.csv", header=None).as_matrix()
    theta2 = pd.read_csv("data/ex3_theta2.csv", header=None).as_matrix()

    X = pd.read_csv("data/ex3data1_X.csv").as_matrix()
    y = pd.read_csv("data/ex3data1_y.csv").as_matrix()

    m = y.shape[0]

    # =====part 1: visualize data======
    display_sampling(X)

    resub_y = predict(theta1, theta2, X)

    resub_error_rate = np.sum(np.equal(resub_y, y).astype(int)) * 100 / m
    expected_error_rate = 97.5

    print("actual error rate : {} \n expected error rate : {}".format(
        resub_error_rate, expected_error_rate
    ))

    if resub_error_rate >= expected_error_rate:
        print("This classifier works well")
if __name__ == '__main__':
    main()