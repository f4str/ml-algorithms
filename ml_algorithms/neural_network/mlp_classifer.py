import numpy as np

from ml_algorithms import utils


class MLPClassifier:
    def __init__(self, hidden_sizes=(100,), activation='relu'):
        self.hidden_sizes = hidden_sizes
        self.activation = activation.lower()
        self.n_layers = len(hidden_sizes) + 2
        self.n_features = 0
        self.n_classes = 0
        self.weights = []
        self.biases = []

        if self.activation == 'sigmoid':
            self.activation_fn = utils.sigmoid
            self.derivative_fn = utils.sigmoid_derivative
        elif self.activation == 'tanh':
            self.activation_fn = utils.tanh
            self.derivative_fn = utils.tanh_derivative
        elif self.activation == 'relu':
            self.activation_fn = utils.relu
            self.derivative_fn = utils.relu_derivative
        else:
            self.activation_fn = utils.identify
            self.derivative_fn = utils.identify_derivative

    def fit(self, X, y, epochs=100, lr=1e-3, batch_size=32):
        X = np.array(X)
        y = np.array(y)
        n, k = X.shape

        self.n_features = k
        self.n_classes = np.max(y) + 1

        sizes = np.concatenate(([k], self.hidden_sizes, [self.n_classes]))

        # random weight initialization
        self.weights = [
            np.random.randn(row, col) / np.sqrt(row) for row, col in zip(sizes[:-1], sizes[1:])
        ]
        # zero bias initialization
        self.biases = [np.zeros(row) for row in sizes[1:]]

        training_loss = []
        training_acc = []

        for _ in range(epochs):
            # shuffle all data
            p = np.random.permutations(n)
            X = X[p]
            y = y[p]

            # split into batches
            batches = [
                (X[i : i + batch_size], y[i : i + batch_size]) for i in range(0, n, batch_size)
            ]

            # batch stochastic gradient descent
            for X_batch, y_batch in batches:
                m = len(X_batch)

                partial_W = [np.zeros(W.shape) for W in self.weights]
                partial_b = [np.zeros(b.shape) for b in self.biases]

                # forward pass
                z_batch = 0
                z_batch_layers = []
                a_batch = X_batch
                a_batch_layers = [X_batch]

                for W, b in zip(self.weights, self.biases):
                    z_batch = np.dot(a_batch, W) + b
                    z_batch_layers.append(z_batch)
                    a_batch = self.activation_fn(z_batch)
                    a_batch_layers.append(a_batch)

                a_batch = utils.softmax(z_batch, axis=1)
                a_batch_layers[-1] = a_batch

                # backward pass
                delta = a_batch - y_batch
                partial_W[-1] = np.dot(a_batch[-2].T, delta) / m
                partial_b[-1] = np.mean(delta, axis=0)

                for layer in range(2, self.n_layers):
                    delta = np.dot(delta, self.weights[-layer + 1].T) * self.derivative_fn(
                        z_batch_layers[-layer]
                    )
                    partial_W[-layer] = np.dot(a_batch_layers[-layer - 1].T, delta) / m
                    partial_b[-layer] = np.mean(delta, axis=0)

                self.weights = [W - lr * pW for W, pW in zip(self.weights, partial_W)]
                self.biases = [b - lr * pb for b, pb in zip(self.biases, partial_b)]

            loss, acc = self.evaluate(X, y)
            training_loss.append(loss)
            training_acc.append(acc)

        return training_loss, training_acc

    def predict_proba(self, X):
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            X = self.activation_fn(np.dot(X, W) + b)

        X = np.dot(X, self.weights[-1]) + self.biases[-1]
        return utils.softmax(X, axis=1)

    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def evaluate(self, X, y):
        one_hot_y = np.eye(self.n_classes)[y]
        y_pred_prob = self.predict_proba(X)
        y_pred = np.argmax(y_pred_prob, axis=1)

        # cross entropy loss
        loss = utils.cross_entropy(y_pred_prob, one_hot_y, axis=1)
        # categorical accuracy
        acc = np.mean(y_pred == y)

        return loss, acc
