from math_utils import mse_loss
import numpy as np
from math_utils import sigmoid

class AutoEncoder838:
    def __init__(self, input_size, hidden_size, output_size, seed):
        rng = np.random.default_rng(seed)
        w1_limit = np.sqrt(6.0 / (input_size + hidden_size))
        w2_limit = np.sqrt(6.0 / (hidden_size + output_size))
        self.w1 = rng.uniform(-w1_limit, w1_limit, size=(input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = rng.uniform(-w2_limit, w2_limit, size=(hidden_size, output_size))
        self.b2 = np.zeros((1, output_size))

    def forward(self, x):
        z1 = x @ self.w1 + self.b1
        h = sigmoid(z1)
        z2 = h @ self.w2 + self.b2
        o = sigmoid(z2)
        return h, o

    def train_one_epoch(self, x, y, learning_rate):
        sample_count = x.shape[0]
        h, o = self.forward(x)
        loss = mse_loss(o, y)

        delta2 = (o - y) * o * (1.0 - o)
        grad_w2 = h.T @ delta2 / sample_count
        grad_b2 = np.mean(delta2, axis=0, keepdims=True)

        delta1 = (delta2 @ self.w2.T) * h * (1.0 - h)
        grad_w1 = x.T @ delta1 / sample_count
        grad_b1 = np.mean(delta1, axis=0, keepdims=True)

        self.w2 -= learning_rate * grad_w2
        self.b2 -= learning_rate * grad_b2
        self.w1 -= learning_rate * grad_w1
        self.b1 -= learning_rate * grad_b1

        return loss

    def encode(self, x):
        h, _ = self.forward(x)
        return h

    def decode(self, hidden_code):
        return sigmoid(hidden_code @ self.w2 + self.b2)

    def predict(self, x):
        _, o = self.forward(x)
        return o