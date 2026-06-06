import numpy as np
import matplotlib.pyplot as plt

from config import OUTPUT_DIR
from model_snapshot import train_snapshot
from plot_utils import heatmap, matrix_grid, normalized_rows, save_current


def plot_loss_windows(losses):
    windows = [200, 2000, len(losses)]
    for window in windows:
        values = losses[:window]
        plt.figure(figsize=(8, 5))
        plt.plot(np.arange(1, window + 1), values, linewidth=2)
        plt.title(f"Loss Curve First {window} Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(alpha=0.25)
        save_current(f"loss_curve_first_{window}.png")


def plot_weights(model):
    heatmap(model.encoder.weights.T, "Encoder Weights", "Input neuron", "Hidden neuron", "encoder_weights.png", cmap="coolwarm")
    heatmap(model.decoder.weights, "Decoder Weights", "Output neuron", "Hidden neuron", "decoder_weights.png", cmap="coolwarm")

    plt.figure(figsize=(8, 5))
    plt.hist(model.encoder.weights.ravel(), bins=18, alpha=0.75, label="Encoder")
    plt.hist(model.decoder.weights.ravel(), bins=18, alpha=0.75, label="Decoder")
    plt.title("Weight Distribution")
    plt.xlabel("Weight value")
    plt.ylabel("Count")
    plt.legend()
    save_current("weight_distribution.png")


def plot_biases(model):
    labels = [f"H{i}" for i in range(1, model.encoder.biases.shape[1] + 1)]
    plt.figure(figsize=(7, 4))
    plt.bar(labels, model.encoder.biases.ravel())
    plt.title("Encoder Biases")
    plt.xlabel("Hidden neuron")
    plt.ylabel("Bias")
    save_current("encoder_biases.png")

    labels = [f"O{i}" for i in range(1, model.decoder.biases.shape[1] + 1)]
    plt.figure(figsize=(8, 4))
    plt.bar(labels, model.decoder.biases.ravel())
    plt.title("Decoder Biases")
    plt.xlabel("Output neuron")
    plt.ylabel("Bias")
    save_current("decoder_biases.png")


def plot_hidden_codes(hidden):
    heatmap(hidden, "Hidden Layer Codes", "Hidden neuron", "Input sample", "hidden_codes.png", cmap="magma", vmin=0.0, vmax=1.0)
    normalized = normalized_rows(hidden)
    angles = np.linspace(0, 2 * np.pi, hidden.shape[1], endpoint=False)
    angles = np.concatenate([angles, angles[:1]])

    plt.figure(figsize=(7, 7))
    axis = plt.subplot(111, polar=True)
    for index, row in enumerate(normalized):
        values = np.concatenate([row, row[:1]])
        axis.plot(angles, values, linewidth=2, label=f"I{index + 1}")
    axis.set_title("Normalized Hidden Code Profiles")
    axis.set_xticks(angles[:-1])
    axis.set_xticklabels([f"H{i}" for i in range(1, hidden.shape[1] + 1)])
    axis.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12))
    save_current("hidden_code_profiles.png")


def plot_reconstruction(decoded, binary_decoded):
    matrix_grid(decoded, "Decoded Output Probabilities", "decoded_probabilities.png")
    matrix_grid(binary_decoded, "Thresholded Decoded Matrix", "thresholded_decoded_matrix.png", cmap="Greens", vmin=0.0, vmax=1.0)

    error = np.abs(decoded - np.eye(decoded.shape[0]))
    matrix_grid(error, "Absolute Reconstruction Error", "reconstruction_error.png", cmap="Reds", vmin=0.0, vmax=max(float(error.max()), 1e-12))


def main():
    snapshot = train_snapshot()
    plot_loss_windows(snapshot["losses"])
    plot_weights(snapshot["model"])
    plot_biases(snapshot["model"])
    plot_hidden_codes(snapshot["hidden"])
    plot_reconstruction(snapshot["decoded"], snapshot["binary_decoded"])
    print(f"Images will be saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
