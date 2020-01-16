import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'src'))
sys.path.append(path)

import feedforward
import data_loader

# load data
training_data = data_loader.load_training_data()
test_data = data_loader.load_test_data()
print('data loaded')

# train network
nn = feedforward.NeuralNetwork([784, 30, 10])
nn.stochastic_gradient_descent(training_data, 30, 10, 3.0, test_data = test_data)
print('training complete')
