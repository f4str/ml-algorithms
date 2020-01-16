import os
import numpy as np
from mnist import MNIST 
	
def vector(i):
	v = np.zeros(10)
	v[i] = 1.0
	return v

def load_data():
	path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data'))
	mndata = MNIST(path = path, gz = True)
	
	images, labels = mndata.load_training()
	test_images, test_labels = mndata.load_testing()
	
	training_data = list(zip(images[:50000], labels[:50000]))
	validation_data = list(zip(images[50000:], labels[50000:]))
	test_data = list(zip(test_images, test_labels))
	
	return (training_data, validation_data, test_data)

load_data()