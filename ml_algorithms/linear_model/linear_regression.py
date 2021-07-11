import numpy as np

from ml_algorithms import utils


class LinearRegression:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.n_features = 0
        self.weights = []
        self.bias = 0

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n, k = X.shape

        self.n_features = k

        if self.fit_intercept:
            ones = np.ones((n, 1))
            X = np.concatenate((ones, X), 1)

        # closed form solution
        beta = np.dot(np.linalg.inv(X.T @ X) @ X.T, y)

        if self.fit_intercept:
            self.bias = beta[0]
            self.weights = beta[1:]
        else:
            self.bias = 0
            self.weights = beta

        y_pred = np.dot(X, beta)

        # mean squared error loss
        loss = utils.mse_score(y, y_pred)
        # r^2 score
        r2 = utils.r2_score(y, y_pred)

        return loss, r2

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def evaluate(self, X, y):
        y = np.array(y)
        y_pred = self.predict(X)

        # mean squared error loss
        loss = utils.mse_score(y, y_pred)
        # r^2 score
        r2 = utils.r2_score(y, y_pred)

        return loss, r2
