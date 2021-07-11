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


def softmax(z, axis):
    t = np.exp(z)
    return t / np.sum(t, axis=axis)
