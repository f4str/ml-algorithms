import numpy as np


def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


def r2_score(y_true, y_pred):
    sse = np.sum(np.square(y_true - y_pred))
    s_yy = np.sum(np.square(y_true - np.mean(y_true)))
    return 1 - sse / s_yy


def mse_score(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))


def rmse_score(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_true - y_pred)))


def mae_score(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def poisson_score(y_true, y_pred):
    return np.mean(y_true * np.log(y_true / y_pred) - y_true + y_pred)


def binary_cross_entropy(a, y):
    return -np.mean(np.nan_to_num(y * np.log(a) + (1 - y) * np.log(1 - a)))


def cross_entropy(a, y, axis):
    return -np.mean(np.sum(y * np.nan_to_num(np.log(a)), axis=axis))


def gini_score(p):
    return 1 - np.sum(np.square(p))


def entropy_score(p):
    return -np.sum(p * np.log(p))


def misclassification_score(p):
    return 1 - np.max(p)
