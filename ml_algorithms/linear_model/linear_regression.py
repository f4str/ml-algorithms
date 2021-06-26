import numpy as np


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
        sse = np.sum(np.square(y - y_pred))
        s_yy = np.sum(np.square(y - np.mean(y)))

        # mean squared error loss
        loss = sse / n
        # r^2 score
        r2 = 1 - sse / s_yy

        return loss, r2

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def evaluate(self, X, y):
        y = np.array(y)
        y_pred = self.predict(X)

        sse = np.sum(np.square(y - y_pred))
        s_yy = np.sum(np.square(y - np.mean(y)))

        # mean squared error loss
        loss = sse / len(y)
        # r^2 score
        r2 = 1 - sse / s_yy

        return loss, r2
