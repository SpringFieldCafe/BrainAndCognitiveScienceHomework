import sys

import numpy as np

from config import ASSIGNMENT_DIR, EPOCHS, HIDDEN_SIZE, INPUT_SIZE, LEARNING_RATE, OUTPUT_SIZE, SEED, THRESHOLD

sys.path.insert(0, str(ASSIGNMENT_DIR))

from autoencoder import AutoEncoder838
from data import identity_dataset
from math_utils import binary_threshold
from trainer import train


def train_snapshot():
    x_train, y_train = identity_dataset(INPUT_SIZE)
    model = AutoEncoder838(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, SEED)
    losses = train(model, x_train, y_train, EPOCHS, LEARNING_RATE)
    hidden, decoded = model.forward(x_train)
    binary_decoded = binary_threshold(decoded, THRESHOLD)
    return {
        "model": model,
        "x_train": x_train,
        "y_train": y_train,
        "losses": np.asarray(losses),
        "hidden": hidden,
        "decoded": decoded,
        "binary_decoded": binary_decoded,
    }
