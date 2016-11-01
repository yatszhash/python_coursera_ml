import numpy as np
import pandas as pd

def main():
    X = pd.read_csv("data/ex3data1_X.csv").as_matrix()
    y = pd.read_csv("data/ex3data1_y.csv").as_matrix()

    m = y.shape[0]

    #randomly select 100 data
    SAMPLING_SIZE = 100
    np.random.seed(10)
    random_indices = np.random.rand(0, m, SAMPLING_SIZE)
    selected_X  = X[random_indices, :]


if __name__ == "__main__":
    main()