import numpy as np
import matplotlib.pyplot as plt

def display_data(X, example_width):
    m, n = X.shape
    example_hight = n / example_width
    imgs = X.map(lambda x: x.reshpe(example_hight, example_width))

    display_rows = np.floor(np.sqrt(m))
    display_cols = np.ceil(m / display_rows)
    fig = plt.subplot(m )
    plt.gray()

def predict(Theta1, Theta2):
    pass