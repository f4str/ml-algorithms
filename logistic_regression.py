import numpy as np


def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def cross_entropy(a, y):
	return -np.mean(np.nan_to_num(y * np.log(a) + (1 - y) * np.log(1 - a)))


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
	
	def _penalty(self):
		# check if weights initialized
		if not self.weights:
			return 0
		
		if self.penalty == 'l1':
			return self.C * np.mean(np.abs(self.weights))
		elif self.penalty == 'l2':
			return self.C * np.mean(np.square(self.weights))
		elif self.penalty == 'elasticnet':
			l1 = self.l1_ratio + self.C * np.mean(np.abs(self.weights))
			l2 = (1 - self.l1_ratio) * self.C * np.mean(np.square(self.weights))
			return l1 + l2
		else:
			return 0
	
	def _penalty_gradient(self):
		# check if weights initialized 
		if not self.weights:
			return 0
		
		if self.penalty == 'l1':
			return self.C * np.mean(np.sign(self.weights))
		elif self.penalty == 'l2':
			return 2 * self.C * np.mean(self.weights)
		elif self.penalty == 'elasticnet':
			l1 = self.C * np.mean(np.sign(self.weights))
			l2 = 2 * self.C * np.mean(self.weights)
			return self.l1_ratio * l1 + (1 - self.l1_ratio) * l2
		else:
			return 0
			
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
			a = sigmoid(z)
			gradient = np.dot(X.T, a - y) / n
			beta -= lr * (gradient + self._penalty_gradient())
			
			if self.fit_intercept:
				self.bias = beta[0]
				self.weights = beta[1:]
			else:
				self.bias = 0
				self.weights = beta
			
			# cross entropy + penalty loss
			loss = cross_entropy(a, y) + self._penalty()
			# binary accuracy
			acc = np.mean(np.around(a) == y)
			
			training_loss.append(loss)
			training_acc.append(acc)
		
		return training_loss, training_acc
	
	def predict_proba(self, X):
		return sigmoid(np.dot(X, self.weights) + self.bias)
	
	def predict_log_proba(self, X):
		return np.log(self.predict_proba(X))
	
	def predict(self, X):
		return np.around(self.predict_proba(X))
	
	def evaluate(self, X, y):
		y = np.array(y)
		y_pred_prob = self.predict_proba(X)
		y_pred = np.argmax(y_pred_prob, axis=1)
		
		# cross entropy + penalty loss
		loss = cross_entropy(y_pred_prob, y) + self._penalty()
		# binary accuracy
		acc = np.mean(y_pred == y)
		
		return loss, acc
