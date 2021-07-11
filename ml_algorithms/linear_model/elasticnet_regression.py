import numpy as np

from ml_algorithms import utils


class ElasticNetRegression:
    def __init__(self, alpha=1.0, l1_ratio=0, fit_intercept=True):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.n_features = 0
        self.n_classes = 0
        self.weights = []
        self.bias = 0

    def fit(self, X, y, epochs=100, lr=1e-3):
        X = np.array(X)
        y = np.array(y)
        n, k = X.shape

        self.n_features = k
        self.n_classes = 2

        if self.fit_intercept:
            ones = np.ones((n, 1))
            X = np.concatenate((ones, X), 1)

        training_loss = []
        training_r2 = []

        # random weight initialization
        self.weights = np.random.randn(k) / np.sqrt(k)
        # zero bias initialization
        self.bias = 0

        # gradient descent
        for _ in range(epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            d_weights = -2 * np.mean(np.dot(X, y - y_pred))
            d_bias = -2 * np.mean(y - y_pred)
            d_penalty = utils.elasticnet_penalty_gradient(self.alpha, self.l1_ratio, self.weights)

            self.weights -= lr * (d_weights + d_penalty)
            self.bias -= lr * d_bias

            # mean squared error + elasticnet penalty loss
            loss = utils.mse_score(y, y_pred) + utils.elasticnet_penalty(
                self.alpha, self.l1_ratio, self.weights
            )
            # r^2 score
            r2 = utils.r2_score(y, y_pred)

            training_loss.append(loss)
            training_r2.append(r2)

        return training_loss, training_r2

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def evaluate(self, X, y):
        y = np.array(y)
        y_pred = self.predict(X)

        # mean squared error + elasticnet penalty loss
        loss = utils.mse_score(y, y_pred) + utils.elasticnet_penalty(
            self.alpha, self.l1_ratio, self.weights
        )
        # r^2 score
        r2 = utils.r2_score(y, y_pred)

        return loss, r2
