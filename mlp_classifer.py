import numpy as np


def sigmoid(z):
	return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
	return sigmoid(z) * (1 - sigmoid(z))


def softmax(z, axis=0):
	t = np.exp(z)
	return t / np.sum(t, axis=axis)


class MLPClassifier:
	def __init__(self, n_inputs, n_classes, hidden_sizes=(100,), activation='relu'):
		self.n_inputs = n_inputs
		self.n_classes = n_classes
		self.hidden_sizes = hidden_sizes
		self.n_layers = len(hidden_sizes) + 2
		
		sizes = np.concatenate(([n_inputs], hidden_sizes,[n_classes]))
		
		# xavier weight initialization
		self.weights = [np.random.randn(row, col) / np.sqrt(col) for row, col in zip(sizes[1:], sizes[:-1])]
		# zero bias initialization
		self.biases = [np.zeros(row) for row in sizes[1:]]
		
		if activation in {'sigmoid', 'relu', 'tanh'}:
			self.activation = activation
		else:
			self.activation = 'none' 
	
	def feedforward(self, a):
		for w, b in zip(self.weights, self.biases):
			a = sigmoid(np.dot(w, a) + b)
		return a
	
	def predict(self, a):
		return np.argmax(self.feedforward(a))
	
	def fit(self, training_data, epochs, test_data=None, lr=1e-1, batch_size=32):
		training_cost = []
		test_cost = []
		training_accuracy = []
		test_accuracy = []
		
		n = len(training_data)
		
		for e in range(epochs):
			np.random.shuffle(training_data)
			mini_batches = [training_data[i:i + batch_size] for i in range(0, n, batch_size)]
			for mini_batch in mini_batches:
				self.update_batch(mini_batch, lr)
			
			if test_data:
				training_cost.append(self.total_cost(training_data))
				test_cost.append(self.total_cost(test_data))
				training_accuracy.append(self.accuracy(training_data))
				test_accuracy.append(self.accuracy(test_data))
			
			print(f'Epoch {e + 1}: complete')
		
		return (training_cost, test_cost, training_accuracy, test_accuracy)
	
	def update_batch(self, batch, lr):
		partial_w = [np.zeros(w.shape) for w in self.weights]
		partial_b = [np.zeros(b.shape) for b in self.biases]
		
		for x, y in batch:
			delta_partial_w, delta_partial_b = self.backpropagation(x, y)
			partial_w = [pw + dpw for pw, dpw in zip(partial_w, delta_partial_w)]
			partial_b = [pb + dpb for pb, dpb in zip(partial_b, delta_partial_b)]
		
		self.weights = [w - (lr / len(batch)) * pw for w, pw in zip(self.weights, partial_w)]
		self.biases = [b - (lr / len(batch)) * pb for b, pb in zip(self.biases, partial_b)]
	
	def backpropagation(self, x, y):
		partial_w = [np.zeros(w.shape) for w in self.weights]
		partial_b = [np.zeros(b.shape) for b in self.biases]
		
		# feedforward
		activation = x
		activations = [x]
		zs = []
		
		for w, b in zip(self.weights, self.biases):
			z = np.dot(w, activation) + b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)
		
		# backward pass
		delta = (activations[-1] - y)
		partial_w[-1] = np.outer(delta, activations[-2])
		partial_b[-1] = delta
		
		for l in range(2, self.n_layers):
			delta = np.dot(delta, self.weights[-l + 1]) * sigmoid_derivative(zs[-l])
			partial_w[-l] = np.outer(delta, activations[-l - 1])
			partial_b[-l] = delta
		
		return (partial_w, partial_b)
	
	def total_cost(self, data):
		cost = 0.0
		for x, y in data:
			a = self.feedforward(x)
			
			one_hot_y = np.zeros(10)
			one_hot_y[y] = 1.0
			y = one_hot_y
			
			cost += np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a))) / len(data)
		return cost
	
	def accuracy(self, data):
		results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data]
		
		accuracy = sum(int(x == y) for (x, y) in results)
		return accuracy
