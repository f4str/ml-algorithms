import numpy as np


def sigmoid(z):
	return 1 / (1 + np.exp(-z))


def cross_entropy(a, y):
	return -y * np.log(a) - (1 - y) * np.log(1 - a)


class LogisticRegression:
	def __init__(self, fit_intercept=True):
		self._coef = []
		self._intercept = 0
		self.fit_intercept = fit_intercept
	
	@property
	def coef(self):
		return self._coef
	
	@property
	def intercept(self):
		return self._intercept
	
	def fit(self, X, y, epochs=10000, lr=1e-3):
		X = np.array(X)
		if len(X.shape) == 1:
			X = X.reshape(-1, 1)
		if self.fit_intercept:
			ones = np.ones((X.shape[0], 1))
			X = np.concatenate((ones, X), 1)
		
		beta = np.zeros(X.shape[1])
		for _ in range(epochs):
			z = np.dot(X, beta)
			a = sigmoid(z)
			gradient = np.dot(X.T, a - y) / y.size
			beta -= lr * gradient
		
		if self.fit_intercept:
			self._intercept = beta[0]
			self._coef = beta[1:]
		else:
			self._coef = beta
	
	def predict_prob(self, X):
		return sigmoid(np.dot(X, self._coef) + self._intercept)
	
	def predict(self, X):
		return np.around(self.predict_prob(X))
	
	def score(self, X, y):
		y_pred = self.predict(X)
		return np.sum(y == y_pred) / len(y)
