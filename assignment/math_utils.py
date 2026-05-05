import numpy as np
import pandas as pd

def sigmoid(z):
    z = np.clip(z, -60, 60)
    return 1.0 / (1.0 + np.exp(-z))


def mse_loss(y_pred, y_true):
    return 0.5 * np.mean(np.sum((y_pred - y_true) ** 2, axis=1))