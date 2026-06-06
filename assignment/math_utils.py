import numpy as np

def sigmoid(z):
    z = np.clip(z, -60, 60)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative_from_output(output):
    return output * (1.0 - output)


def binary_threshold(matrix, threshold):
    return (matrix >= threshold).astype(int)
