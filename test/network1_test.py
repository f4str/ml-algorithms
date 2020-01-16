import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'src'))
sys.path.append(path)

import network1  
import data_loader 

training_data, validation_data, test_data = data_loader.load_data()
print('data loaded')
nn = network1.NeuralNetwork([784, 30, 10])
nn.stochastic_gradient_descent(training_data, 30, 10, 3.0, test_data = test_data)
print('training complete')
