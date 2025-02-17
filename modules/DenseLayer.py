from modules.BaseLayer import BaseLayer
from modules.Activation import ReLU, Softmax, Sigmoid
import numpy as np

class DenseLayer(BaseLayer):
    def __init__(self, input_size, output_size, activation="relu", initialization="he"):
        activation_classes = {
            "relu": ReLU,
            "sigmoid": Sigmoid,
            "softmax": Softmax
        }

        if activation not in activation_classes:
            raise ValueError(f"Unsupported activation function: {activation}")

        self.activation = activation_classes[activation]()

        if initialization == "xavier":
            self.weights = np.random.randn(output_size, input_size) * np.sqrt(1 / input_size)
        elif initialization == "he":
            self.weights = np.random.randn(output_size, input_size) * np.sqrt(2 / input_size)
        else:
            self.weights = np.random.randn(output_size, input_size) * 0.01

        self.biases = np.zeros((output_size, 1))

    def forward(self, X):
        self.input = X
        self.pre_activation = np.dot(self.weights, X) + self.biases
        self.output = self.activation.forward(self.pre_activation)
        return self.output

    def backward(self, d_output, Z, input_data, learning_rate):
        if d_output.shape[0] != self.weights.shape[0]:
            d_output = d_output.T

        if Z.shape[0] != self.weights.shape[0]:
            Z = Z.T

        d_output *= self.activation.derivative(Z)

        if input_data.shape[0] != self.weights.shape[1]:
            input_data = input_data.T

        if input_data.shape[0] != self.weights.shape[1]:
            raise ValueError(
                f"Layer mismatch: input_data {input_data.shape} does not match expected input size {self.weights.shape[1]}"
            )

        d_weights = np.dot(d_output, input_data.T) / input_data.shape[1]
        d_biases = np.mean(d_output, axis=1, keepdims=True)
        d_input = np.dot(self.weights.T, d_output)
        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases

        return d_input