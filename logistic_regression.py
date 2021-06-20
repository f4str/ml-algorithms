import numpy as np


def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def cross_entropy(a, y):
	return -np.mean(np.nan_to_num(y * np.log(a) + (1 - y) * np.log(1 - a)))

def no_penalty(C, w):
	return 0

def no_penalty_gradient(C, w):
	return 0

def l1_penalty(C, w):
	return C * np.mean(np.abs(w))

def l1_penalty_gradient(C, w):
	return C * np.mean(np.sign(w))

def l2_penalty(C, w):
	return C * np.mean(np.square(w))

def l2_penalty_gradient(C, w):
	return 2 * C * np.mean(w)


class LogisticRegression:
	def __init__(self, fit_intercept=True, penalty='l2', C=1):
		self.fit_intercept = fit_intercept
		self.penalty = penalty.lower()
		self.C = C
		self.n_features = 0
		self.n_classes = 0
		self.weights = []
		self.bias = 0
		
		if self.penalty == 'l1':
			self.penalty = l1_penalty
			self.penalty_gradient = l2_penalty_gradient
		elif self.penalty == 'l2':
			self.penalty = l2_penalty
			self.penalty_gradient = l2_penalty_gradient
		else:
			self.penalty = no_penalty
			self.penalty_gradient = no_penalty_gradient
			
	def fit(self, X, y, epochs=1, lr=1e-3):
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
		
		# xavier weight initialization
		beta = np.random.randn(k) / np.sqrt(k)
		
		# gradient descent
		for _ in range(epochs):
			z = np.dot(X, beta)
			a = sigmoid(z)
			gradient = np.dot(X.T, a - y) / n
			beta -= lr * gradient + self.penalty_gradient(self.C, beta)
			
			# cross entropy loss
			loss = cross_entropy(a, y) + self.penalty(self.C, beta)
			# binary accuracy
			acc = np.mean(np.around(a) == y)
			
			training_loss.append(loss)
			training_acc.append(acc)
		
		if self.fit_intercept:
			self.bias = beta[0]
			self.weights = beta[1:]
		else:
			self.bias = 0
			self.weights = beta
		
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
		
		# cross entropy loss
		loss = cross_entropy(y_pred_prob, y) + self.penalty(self.C, self.weights)
		# binary accuracy
		acc = np.mean(y_pred == y)
		
		return loss, acc
