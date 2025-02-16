from modules.BaseLayer import BaseLayer
from modules.ActivationFunction import ActivationFunction
import numpy as np

class DenseLayer(BaseLayer):
    def __init__(self, input_size, output_size, activation="relu", initialization="he"):
        self.activation_func = ActivationFunction(activation)
        self.activation_derivative = getattr(self, f"derivative_{activation}", None)

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
        self.output = self.activation_func(self.pre_activation)
        return self.output

    def backward(self, d_output, learning_rate):
        if self.activation_derivative:
            d_output *= self.activation_derivative(self.pre_activation)

        d_weights = np.dot(d_output, self.input.T) / self.input.shape[1]
        d_biases = np.mean(d_output, axis=1, keepdims=True)
        d_input = np.dot(self.weights.T, d_output)

        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases
        return d_input
