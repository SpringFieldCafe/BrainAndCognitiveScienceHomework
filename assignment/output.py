import numpy as np

def matrix_text(matrix, precision=4):
    return np.array2string(np.round(matrix, precision), separator=" ")


def build_result_text(original, encoded, decoded, binary_decoded, final_loss, loss_image_path, threshold):
    parts = [
        "Final loss:",
        f"{final_loss:.12f}",
        "",
        "Original 8x8 identity matrix:",
        matrix_text(original.astype(int), 0),
        "",
        "Encoded 8x3 hidden-layer matrix:",
        matrix_text(encoded),
        "",
        "Decoded 8x8 output matrix:",
        matrix_text(decoded),
        "",
        f"Decoded 8x8 binary matrix with threshold {threshold}:",
        matrix_text(binary_decoded, 0),
        "",
        "Loss image path:",
        str(loss_image_path),
        "",
    ]
    return "\n".join(parts)


def write_text(path, text):
    path.write_text(text, encoding="utf-8")
