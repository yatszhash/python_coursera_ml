import numpy as np
import pandas as pd

from ex3.compute_functions_ex3 import predict_one_vs_all, opt_one_vs_all


def main():
    X = pd.read_csv("data/ex3data1_X.csv").as_matrix()
    y = pd.read_csv("data/ex3data1_y.csv").as_matrix()

    m = y.shape[0]

    #randomly select 100 data
    SAMPLING_SIZE = 100
    np.random.seed(10)
    #random_indices = np.random.rand(0, m, SAMPLING_SIZE)
    #selected_X  = X[random_indices, :]

    #todo visualize data

    #=====part 2: vectorize logistic regression

    print("Training One-vs-All Logistic Regression...\n")

    lambda_ = 0.1

    initial_theta = np.zeros((X.shape[1] + 1, 1))
    num_labels = np.max(y)
    all_theta = opt_one_vs_all(X, y, initial_theta, num_labels)
    resub_y = predict_one_vs_all(X, all_theta)

    resub_error_rate = np.sum(np.equal(resub_y, y).astype(int)) * 100 / m
    expected_error_rate = 94.9

    print("actual error rate : {} \n expected error rate : {}".format(
        resub_error_rate, expected_error_rate
    ))

    if resub_error_rate >= expected_error_rate:
        print("This classifier works well")

if __name__ == "__main__":
    main()