import numpy as np

from ml_algorithms import utils


class LogisticRegression:
    def __init__(self, penalty='l2', C=1, l1_ratio=0, fit_intercept=True):
        self.penalty = penalty.lower()
        self.C = C
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.n_features = 0
        self.n_classes = 0
        self.weights = []
        self.bias = 0

    def _penalty_cost(self):
        if self.penalty == 'l1':
            return utils.l1_penalty(self.C, self.weights)
        elif self.penalty == 'l2':
            return utils.l2_penalty(self.C, self.weights)
        elif self.penalty == 'elasticnet':
            return utils.elasticnet_penalty(self.C, self.l1_ratio, self.weights)
        else:
            return utils.no_penalty(self.C, self.weights)

    def _penalty_gradient(self):
        if self.penalty == 'l1':
            return utils.l1_penalty_gradient(self.C, self.weights)
        elif self.penalty == 'l2':
            return utils.l2_penalty_gradient(self.C, self.weights)
        elif self.penalty == 'elasticnet':
            return utils.elasticnet_penalty_gradient(self.C, self.l1_ratio, self.weights)
        else:
            return utils.no_penalty_gradient(self.C, self.weights)

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
        training_acc = []

        # random weight initialization
        beta = np.random.randn(k) / np.sqrt(k)

        # gradient descent
        for _ in range(epochs):
            z = np.dot(X, beta)
            a = utils.sigmoid(z)
            gradient = np.dot(X.T, a - y) / n
            penalty = self._penalty_gradient()
            beta -= lr * (gradient + penalty)

            if self.fit_intercept:
                self.bias = beta[0]
                self.weights = beta[1:]
            else:
                self.bias = 0
                self.weights = beta

            # cross entropy + penalty loss
            loss = utils.binary_cross_entropy(a, y) + self._penalty_cost()
            # binary accuracy
            acc = np.mean(np.around(a) == y)

            training_loss.append(loss)
            training_acc.append(acc)

        return training_loss, training_acc

    def predict_proba(self, X):
        return utils.sigmoid(np.dot(X, self.weights) + self.bias)

    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

    def predict(self, X):
        return np.around(self.predict_proba(X))

    def evaluate(self, X, y):
        y = np.array(y)
        y_pred_prob = self.predict_proba(X)
        y_pred = np.argmax(y_pred_prob, axis=1)

        # binary cross entropy + penalty loss
        loss = utils.binary_cross_entropy(y_pred_prob, y) + self._penalty_cost()
        # binary accuracy
        acc = np.mean(y_pred == y)

        return loss, acc
