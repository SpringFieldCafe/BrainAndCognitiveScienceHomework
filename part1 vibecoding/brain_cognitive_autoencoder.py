# 文件划分说明：本文件可以作为一个完整的 brain_cognitive_autoencoder.py 直接运行；如果需要拆分，共拆成 4 个文件：config.py、math_utils.py、autoencoder.py、main.py。
# 文件划分说明：config.py 从“config.py START”到“config.py END”；math_utils.py 从“math_utils.py START”到“math_utils.py END”；autoencoder.py 从“autoencoder.py START”到“autoencoder.py END”；main.py 从“main.py START”到“main.py END”。
# 文件划分说明：如果拆分，math_utils.py 保留 import numpy as np；autoencoder.py 需要导入 numpy、sigmoid；main.py 需要导入 os、numpy、matplotlib.pyplot、config.py 中的常量和 autoencoder.py 中的 AutoEncoder838。

# ===== config.py START =====
INPUT_SIZE = 8
HIDDEN_SIZE = 3
OUTPUT_SIZE = 8
EPOCHS = 50000
LEARNING_RATE = 10.0
SEED = 42
LOSS_IMAGE_NAME = "loss_curve.png"
RESULT_TEXT_NAME = "decoded_result.txt"
# ===== config.py END =====

# ===== math_utils.py START =====
import numpy as np


def sigmoid(z):
    z = np.clip(z, -60, 60)
    return 1.0 / (1.0 + np.exp(-z))


def mse_loss(y_pred, y_true):
    return 0.5 * np.mean(np.sum((y_pred - y_true) ** 2, axis=1))
# ===== math_utils.py END =====

# ===== autoencoder.py START =====
import numpy as np


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
# ===== autoencoder.py END =====

# ===== main.py START =====
import os
import numpy as np
import matplotlib.pyplot as plt


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
# ===== main.py END =====
