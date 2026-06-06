import numpy as np
from pathlib import Path
from autoencoder import AutoEncoder838
from config import (
    DECISION_THRESHOLD,
    EPOCHS,
    HIDDEN_SIZE,
    INPUT_SIZE,
    LEARNING_RATE,
    LOSS_IMAGE_NAME,
    LOSS_PLOT_MAX_EPOCH,
    OUTPUT_SIZE,
    RESULT_TEXT_NAME,
    SEED,
)
from data import identity_dataset
from math_utils import binary_threshold
from output import build_result_text, write_text
from plotting import save_loss_curve
from trainer import train


def main():
    np.set_printoptions(precision=4, suppress=True)

    x_train, y_train = identity_dataset(INPUT_SIZE)
    model = AutoEncoder838(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, SEED)
    losses = train(model, x_train, y_train, EPOCHS, LEARNING_RATE)

    encoded_matrix = model.encode(x_train)
    decoded_matrix = model.predict(x_train)
    binary_decoded_matrix = binary_threshold(decoded_matrix, DECISION_THRESHOLD)

    base_dir = Path(__file__).resolve().parent
    loss_image_path = base_dir / LOSS_IMAGE_NAME
    result_text_path = base_dir / RESULT_TEXT_NAME

    save_loss_curve(losses, loss_image_path, LOSS_PLOT_MAX_EPOCH)
    result_text = build_result_text(
        x_train,
        encoded_matrix,
        decoded_matrix,
        binary_decoded_matrix,
        losses[-1],
        loss_image_path,
        DECISION_THRESHOLD,
    )
    write_text(result_text_path, result_text)

    print(result_text)


if __name__ == "__main__":
    main()
