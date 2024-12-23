import numpy as np


class LinearRegression:
    def __init__(self):
        self.weights = None

    def _gradient_descent(self, x: np.array, y: np.array):

    def fit(self, x: np.array, y: np.array) -> None:
        n_rows, n_cols = x.shape
        if (n_rows < 1000) and (n_cols < 1000):
            mtx = x.T @ x
            mtx = np.linalg.inv(mtx)
        elif (n_rows < 10_000) and (n_cols < 10_000):
            chol = np.linalg.cholesky(x)
            mtx = chol.T @ chol
        else:


        self.weights = mtx @ (x.T @ y)



size = (100, 5)

X = np.random.rand(*size)
Y = np.random.randint(2, 4, size=(size[0],)) * X[:, 0] + np.random.randint(5, 7, size=(size[0],)) * X[:, 1]

