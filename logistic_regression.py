import numpy as np


def sigmoid(z):
	return 1 / (1 + np.exp(-z))


def cross_entropy(a, y):
	return -np.mean(np.nan_to_num(y * np.log(a) + (1 - y) * np.log(1 - a)))


class LogisticRegression:
	def __init__(self, fit_intercept=True):
		self.weights = []
		self.bias = 0
		self.fit_intercept = fit_intercept
	
	def fit(self, X, y, epochs=1000, lr=1e-3):
		X = np.array(X)
		y = np.array(y)
		
		if len(X.shape) == 1:
			X = X.reshape(-1, 1)
		
		n, k = X.shape
		
		if self.fit_intercept:
			ones = np.ones((n, 1))
			X = np.concatenate((ones, X), 1)
		
		training_loss = []
		training_acc = []
		
		# xavier weight initialization
		beta = np.random.randn(k) / np.sqrt(k)
		
		# gradient descent
		for _ in range(epochs):
			z = np.dot(X, beta)
			a = sigmoid(z)
			gradient = np.dot(X.T, a - y) / n
			beta -= lr * gradient
			
			# cross entropy loss
			loss = cross_entropy(a, y)
			# binary accuracy
			acc = np.mean(np.around(a) == y)
			
			training_loss.append(loss)
			training_acc.append(acc)
		
		if self.fit_intercept:
			self.bias = beta[0]
			self.weights = beta[1:]
		else:
			self.weights = beta
		
		return training_loss, training_acc
	
	def predict_proba(self, X):
		return sigmoid(np.dot(X, self.weights) + self.bias)
	
	def predict(self, X):
		return np.around(self.predict_proba(X))
	
	def score(self, X, y):
		y_pred = self.predict(X)
		return np.mean(y_pred == y)
