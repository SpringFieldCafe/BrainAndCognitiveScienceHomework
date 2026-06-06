import numpy as np
from math_utils import sigmoid

class DenseSigmoidLayer:
    def __init__(self, input_size, output_size, seed):
        rng = np.random.default_rng(seed)
        limit = np.sqrt(6.0 / (input_size + output_size))
        self.weights = rng.uniform(-limit, limit, size=(input_size, output_size))
        self.biases = np.zeros((1, output_size))

    def forward(self, x):
        return sigmoid(x @ self.weights + self.biases)

    def apply_gradients(self, weight_gradient, bias_gradient, learning_rate):
        self.weights -= learning_rate * weight_gradient
        self.biases -= learning_rate * bias_gradient
