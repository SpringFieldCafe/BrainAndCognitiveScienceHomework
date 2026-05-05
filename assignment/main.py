import os
import numpy as np
import pathlib
import pandas 
import matplotlib.pyplot as plt
from config import (INPUT_SIZE,HIDDEN_SIZE,OUTPUT_SIZE,EPOCHS,LEARNING_RATE,SEED,LOSS_IMAGE_NAME,RESULT_TEXT_NAME,)
from autoencoder import AutoEncoder838


def main():
    np.set_printoptions(precision=4, suppress=True)

    x_train = np.eye(INPUT_SIZE)
    y_train = x_train.copy()

    model = AutoEncoder838(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, SEED)
    losses = []

    for epoch in range(1, EPOCHS + 1):
        loss = model.train_one_epoch(x_train, y_train, LEARNING_RATE)
        losses.append(loss)

    encoded_matrix = model.encode(x_train)
    decoded_matrix = model.predict(x_train)
    binary_decoded_matrix = (decoded_matrix >= 0.5).astype(int)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    loss_image_path = os.path.join(base_dir, LOSS_IMAGE_NAME)
    result_text_path = os.path.join(base_dir, RESULT_TEXT_NAME)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, EPOCHS + 1), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.xlim(-10,600)
    plt.tight_layout()
    plt.savefig(loss_image_path, dpi=300)
    plt.close()

    result_text = (
        "Final loss:\n"
        + str(losses[-1])
        + "\n\nOriginal 8x8 identity matrix:\n"
        + str(x_train.astype(int))
        + "\n\nEncoded 8x3 hidden-layer matrix:\n"
        + str(np.round(encoded_matrix, 4))
        + "\n\nDecoded 8x8 output matrix:\n"
        + str(np.round(decoded_matrix, 4))
        + "\n\nDecoded 8x8 binary matrix with threshold 0.5:\n"
        + str(binary_decoded_matrix)
        + "\n\nLoss image path:\n"
        + loss_image_path
        + "\n"
    )

    with open(result_text_path, "w", encoding="utf-8") as f:
        f.write(result_text)

    print(result_text)


if __name__ == "__main__":
    main()