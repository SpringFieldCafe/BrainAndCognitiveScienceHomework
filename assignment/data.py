import numpy as np

def identity_dataset(size):
    x = np.eye(size)
    return x, x.copy()
