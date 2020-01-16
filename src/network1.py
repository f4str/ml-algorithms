'''
feedforward neural network 
stochastic gradient descent learning with backpropagation
'''

import random
import numpy as np

class NeuralNetwork:
	def __init__(self, sizes):
		self.sizes = sizes
		self.layers = len(sizes)
		self.weights = [np.random.randn(row, col) for row, col in zip(sizes[1:], sizes[:-1])]
		self.biases = [np.random.randn(row) for row in sizes[1:]]
	
	def feedforward(self, a):
		for w, b in zip(self.weights, self.biases):
			a = sigmoid(np.dot(w, a) + b)
		return a
	
	def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, eta, test_data = None):
		n = len(training_data)
		
		if test_data:
			n_test = len(test_data)
		
		for e in range(epochs):
			random.shuffle(training_data)
			mini_batches = [training_data[i:i + mini_batch_size] for i in range(0, n, mini_batch_size)]
			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch, eta)
			if test_data:
				print(f'Epoch {e + 1}: {self.evaluate(test_data)} / {n_test}')
			else:
				print(f'Epoch {e + 1}: complete')
	
	def update_mini_batch(self, mini_batch, eta):
		partial_w = [np.zeros(w.shape) for w in self.weights]
		partial_b = [np.zeros(b.shape) for b in self.biases]
		
		for x, y in mini_batch:
			delta_partial_w, delta_partial_b = self.backpropagation(x, y)
			partial_w = [pw + dpw for pw, dpw in zip(partial_w, delta_partial_b)]
			partial_b = [pb + dpb for pb, dpb in zip(partial_b, delta_partial_b)]
		
		self.weights = [w - (eta / len(mini_batch)) * pw for w, pw in zip(self.weights, partial_w)]
		self.biases = [b - (eta / len(mini_batch)) * pb for b, pb in zip(self.biases, partial_b)]
	
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
		delta = (activations[-1] - y) * sigmoid_derivative(zs[-1])
		partial_w[-1] = np.dot(delta, activations[-2])
		partial_b[-1] = delta
		
		for l in range(2, self.layers):
			z =  zs[-l]
			delta = np.dot(self.weights[-l + 1], delta) * sigmoid_derivative(z)
			partial_w[-l] = np.dot(delta, activations[-l - 1])
			partial_b[-l] = delta
		
		return (partial_w, partial_b)
	
	def evaluate(self, test_data):
		test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
		return sum(int(x == y) for (x, y) in test_results)


def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
	return sigmoid(z) * (1 - sigmoid(z))
