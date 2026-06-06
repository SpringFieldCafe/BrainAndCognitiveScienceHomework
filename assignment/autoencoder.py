from layers import DenseSigmoidLayer
from losses import mean_squared_error
from math_utils import sigmoid_derivative_from_output

class AutoEncoder838:
    def __init__(self, input_size, hidden_size, output_size, seed):
        self.encoder = DenseSigmoidLayer(input_size, hidden_size, seed)
        self.decoder = DenseSigmoidLayer(hidden_size, output_size, seed + 1)

    def forward(self, x):
        hidden = self.encoder.forward(x)
        output = self.decoder.forward(hidden)
        return hidden, output

    def train_one_epoch(self, x, y, learning_rate):
        sample_count = x.shape[0]
        hidden, output = self.forward(x)
        loss = mean_squared_error(output, y)

        output_delta = (output - y) * sigmoid_derivative_from_output(output)
        decoder_weight_gradient = hidden.T @ output_delta / sample_count
        decoder_bias_gradient = output_delta.mean(axis=0, keepdims=True)

        hidden_delta = (output_delta @ self.decoder.weights.T) * sigmoid_derivative_from_output(hidden)
        encoder_weight_gradient = x.T @ hidden_delta / sample_count
        encoder_bias_gradient = hidden_delta.mean(axis=0, keepdims=True)

        self.decoder.apply_gradients(decoder_weight_gradient, decoder_bias_gradient, learning_rate)
        self.encoder.apply_gradients(encoder_weight_gradient, encoder_bias_gradient, learning_rate)

        return loss

    def encode(self, x):
        hidden, _ = self.forward(x)
        return hidden

    def decode(self, hidden_code):
        return self.decoder.forward(hidden_code)

    def predict(self, x):
        _, output = self.forward(x)
        return output
