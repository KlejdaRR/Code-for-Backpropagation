from modules.BaseLayer import BaseLayer
import numpy as np

class DenseLayer(BaseLayer):

    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) * 0.01
        self.biases = np.zeros((output_size, 1))

    def forward(self, X):
        self.input = X
        self.output = 1 / (1 + np.exp(-np.dot(self.weights, X) - self.biases))
        return self.output

    def backward(self, d_output, learning_rate):
        d_sigmoid = self.output * (1 - self.output) * d_output
        d_weights = np.dot(d_sigmoid, self.input.T)
        d_biases = np.sum(d_sigmoid, axis=1, keepdims=True)
        d_input = np.dot(self.weights.T, d_sigmoid)

        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases
        return d_input
