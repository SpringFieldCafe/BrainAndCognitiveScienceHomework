import numpy as np

def train(model, x, y, epochs, learning_rate):
    losses = np.empty(epochs, dtype=float)
    for index in range(epochs):
        losses[index] = model.train_one_epoch(x, y, learning_rate)
    return losses
