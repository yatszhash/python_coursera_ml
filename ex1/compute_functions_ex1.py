
import numpy as np

def hypothesis_function(X, theta):
    assert theta.shape[0], X.shape[1]
    h = np.dot(X, theta)
    return h

def  compute_cost(X, y, theta):

    j = 0
    h = hypothesis_function(X, theta)
    m = y.shape[0]

    assert  h.shape,  y.shape

    difference= h - y
    j = (1 / (2 * m)) * np.dot(difference.T, difference)
    assert type(j), float
    return  j

def  update_theta(X, y, theta, alpha):

    correction= alpha * (1.0 / len(y)) * np.dot(
                                    (hypothesis_function(X, theta) - y).T, X).T

    np.testing.assert_allclose(correction.shape, theta.shape)
    new_theta =  theta - correction

    return new_theta


def gradient_descent(X, y, theta, alpha, num_iters):
    J_history = np.zeros((num_iters, 1))
    temp_theta = theta

    for iter in range(num_iters):
        temp_theta = update_theta(X, y, temp_theta, alpha)
        J_history[iter, ] = compute_cost(X, y, temp_theta)
    return temp_theta, J_history

def feature_normalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0, ddof=1)

    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

def normal_equation(X, y):
    theta = np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), y)
    return theta

def magic(N):
    magic_square = np.zeros((N,N), dtype=int)

    n = 1
    i, j = 0, N//2

    while n <= N**2:
        magic_square[i, j] = n
        n += 1
        newi, newj = (i-1) % N, (j+1)% N
        if magic_square[newi, newj]:
            i += 1
        else:
            i, j = newi, newj

    return magic_square
