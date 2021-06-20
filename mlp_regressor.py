import numpy as np


def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
	return sigmoid(z) * (1 - sigmoid(z))

def tanh(z):
	return np.tanh(z)

def tanh_derivative(z):
	return 1 - np.square(np.tanh(z))

def relu(z):
	return z * (z > 0)

def relu_derivative(z):
	return 1.0 * (z > 0)

def identify(z):
	return z

def identify_derivative(z):
	return 1


class MLPRegressor:
	def __init__(self, hidden_sizes=(10,), activation='relu'):
		self.hidden_sizes = hidden_sizes
		self.n_layers = len(hidden_sizes) + 2
		self.n_features = 0
		self.n_outputs = 1
		self.weights = []
		self.biases = []
		
		if activation == 'sigmoid':
			self.activation = sigmoid
			self.derivative = sigmoid_derivative
		elif activation == 'tanh':
			self.activation = tanh
			self.derivative = tanh_derivative
		elif activation == 'relu':
			self.activation = relu
			self.derivative = relu_derivative
		else:
			self.activation = identify
			self.derivative = identify_derivative
	
	def fit(self, X, y, epochs=100, lr=1e-3, batch_size=32):
		X = np.array(X)
		y = np.array(y)
		n, k = X.shape
		
		self.n_features = k
		self.n_outputs = 1
		
		sizes = np.concatenate(([k], self.hidden_sizes, [self.n_outputs]))
		
		# random weight initialization
		self.weights = [np.random.randn(row, col) / np.sqrt(row) for row, col in zip(sizes[:-1], sizes[1:])]
		# zero bias initialization
		self.biases = [np.zeros(row) for row in sizes[1:]]
		
		training_loss = []
		training_r2 = []
		
		for _ in range(epochs):
			# shuffle all data
			p = np.random.permutations(n)
			X = X[p]
			y = y[p]
			
			# split into batches
			batches = [(X[i:i + batch_size], y[i:i + batch_size]) for i in range(0, n, batch_size)]
			
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
					a_batch = self.activation(z_batch)
					a_batch_layers.append(a_batch)
				
				# backward pass
				delta = z_batch - y_batch
				partial_W[-1] = np.dot(a_batch[-2].T, delta) / m
				partial_b[-1] = np.mean(delta, axis=0)
				
				for l in range(2, self.n_layers):
					delta = np.dot(delta, self.weights[-l + 1].T) * self.derivative(z_batch_layers[-l])
					partial_W[-l] = np.dot(a_batch_layers[-l - 1].T, delta) / m
					partial_b[-l] = np.mean(delta, axis=0)
				
				self.weights = [W - lr * pW for W, pW in zip(self.weights, partial_W)]
				self.biases = [b - lr * pb for b, pb in zip(self.biases, partial_b)]
			
			loss, r2 = self.evaluate(X, y)
			training_loss.append(loss)
			training_r2.append(r2)
		
		return training_loss, training_r2
	
	def predict(self, X):
		for W, b in zip(self.weights[:-1], self.biases[:-1]):
			X = self.activation(np.dot(X, W) + b)
		
		return np.dot(X, self.weights[-1]) + self.biases[-1]
	
	def evaluate(self, X, y):
		y = np.array(y)
		y_pred = self.predict(X)
		
		sse = np.sum(np.square(y - y_pred))
		s_yy = np.sum(np.square(y - y.mean()))
		
		# mean squared error loss
		loss = sse / len(y)
		# r^2 score
		r2 = 1 - sse / s_yy
		
		return loss, r2
