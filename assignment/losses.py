import numpy as np

def mean_squared_error(y_pred, y_true):
    return 0.5 * np.mean(np.sum((y_pred - y_true) ** 2, axis=1))
