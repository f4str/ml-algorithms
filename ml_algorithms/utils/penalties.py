import numpy as np


def no_penalty(alpha, weights):
    return 0


def no_penalty_gradient(alpha, weights):
    return 0


def l1_penalty(alpha, weights):
    if not weights:
        return 0
    return alpha * np.mean(np.abs(weights))


def l1_penalty_gradient(alpha, weights):
    if not weights:
        return 0
    return alpha * np.mean(np.sign(weights))


def l2_penalty(alpha, weights):
    if not weights:
        return 0
    return alpha * np.mean(np.square(weights))


def l2_penalty_gradient(alpha, weights):
    if not weights:
        return 0
    return 2 * alpha * np.mean(weights)


def elasticnet_penalty(alpha, l1_ratio, weights):
    if not weights:
        return 0
    l1 = alpha * np.mean(np.sign(weights))
    l2 = 2 * alpha * np.mean(weights)
    return l1_ratio * l1 + (1 - l1_ratio) * l2


def elasticnet_penalty_gradient(alpha, l1_ratio, weights):
    if not weights:
        return 0
    d_l1 = alpha * np.mean(np.sign(weights))
    d_l2 = 2 * alpha * np.mean(weights)
    return l1_ratio * d_l1 + (1 - l1_ratio) * d_l2
