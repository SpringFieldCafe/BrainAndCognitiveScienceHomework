import numpy as np
import matplotlib.pyplot as plt

from config import OUTPUT_DIR


def save_current(name):
    path = OUTPUT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def heatmap(matrix, title, xlabel, ylabel, path_name, cmap="viridis", vmin=None, vmax=None):
    plt.figure(figsize=(7, 5))
    image = plt.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    plt.colorbar(image)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(range(matrix.shape[1]), range(1, matrix.shape[1] + 1))
    plt.yticks(range(matrix.shape[0]), range(1, matrix.shape[0] + 1))
    save_current(path_name)


def matrix_grid(matrix, title, path_name, cmap="Blues", vmin=0.0, vmax=1.0):
    plt.figure(figsize=(6, 6))
    image = plt.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.title(title)
    plt.xlabel("Output neuron")
    plt.ylabel("Input sample")
    plt.xticks(range(matrix.shape[1]), range(1, matrix.shape[1] + 1))
    plt.yticks(range(matrix.shape[0]), range(1, matrix.shape[0] + 1))
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            color = "white" if value > 0.5 else "black"
            plt.text(col, row, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8)
    save_current(path_name)


def normalized_rows(matrix):
    row_min = matrix.min(axis=1, keepdims=True)
    row_max = matrix.max(axis=1, keepdims=True)
    return (matrix - row_min) / np.maximum(row_max - row_min, 1e-12)
