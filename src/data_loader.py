import os
import numpy as np

def load_training_data():
	file = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data', 'training-data.npz'))
	training_data = np.load(file)
	return list(zip(training_data['images'], training_data['labels']))

def load_validation_data():
	file = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data', 'validation-data.npz'))
	validation_data = np.load(file)
	return list(zip(validation_data['images'], validation_data['labels']))

def load_test_data():
	file = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data', 'test-data.npz'))
	test_data = np.load(file)
	return list(zip(test_data['images'], test_data['labels'])) 
