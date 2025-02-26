from modules.BaseLayer import BaseLayer
from modules.Activation import ReLU, Softmax, Sigmoid, Linear
import numpy as np

class DenseLayer(BaseLayer):
    def __init__(self, input_size, output_size, activation, initialization, batch_norm=True):
        activation_classes = {
            "relu": ReLU,
            "sigmoid": Sigmoid,
            "softmax": Softmax,
            "linear": Linear
        }

        if activation not in activation_classes:
            raise ValueError(f"Unsupported activation function: {activation}")

        self.activation = activation_classes[activation]()

        # Initializing weights based on the chosen method
        if initialization == "xavier":
            self.weights = np.random.randn(output_size, input_size) * np.sqrt(1 / input_size)
        elif initialization == "he":
            self.weights = np.random.randn(output_size, input_size) * np.sqrt(2 / input_size)
        else:
            self.weights = np.random.randn(output_size, input_size) * 0.01

        # biases are initialized to zero
        self.biases = np.zeros((output_size, 1))

        self.batch_norm = batch_norm
        if self.batch_norm:
            self.gamma = np.ones((output_size, 1))  # Scale parameter
            self.beta = np.zeros((output_size, 1))  # Shift parameter

    def forward(self, X):
        self.input = X

        # Computing the pre-activation output: Z = weights * X + biases
        self.pre_activation = np.dot(self.weights, X) + self.biases

        # If batch_norm is enabled, we normalize the pre-activation output
        # Batch normalization normalizes the outputs by adjusting them
        # to have a mean of 0 and a standard deviation of 1 across each mini-batch of data
        if self.batch_norm:
            batch_mean = np.mean(self.pre_activation, axis=1, keepdims=True)
            batch_std = np.std(self.pre_activation, axis=1, keepdims=True) + 1e-8
            self.pre_activation = (self.pre_activation - batch_mean) / batch_std

            # After normalization, the outputs are scaled by self.gamma (a learnable scaling parameter)
            # and shifted by self.beta (a learnable shifting parameter)
            self.pre_activation = self.gamma * self.pre_activation + self.beta

        # Application of the chosen the activation function to the pre-activation output
        self.output = self.activation.forward(self.pre_activation)
        return self.output

    def backward(self, d_output, Z, input_data, learning_rate, lambda_reg=0.001):
        """Computes gradients using backpropagation and updates weights with L2 regularization."""

        # Ensuring the gradient of the loss (d_output) and pre-activation output (Z)
        # have the correct shapes for backpropagation
        if d_output.shape[0] != self.weights.shape[0]:
            d_output = d_output.T

        if Z.shape[0] != self.weights.shape[0]:
            Z = Z.T

        # If the activation is not softmax we multiply the gradient by the derivative of the activation function.
        if isinstance(self.activation, Softmax):
            d_output = d_output
        else:
            d_output *= self.activation.derivative(Z)

        # input_data = input to the current layer (output of the previous layer or the original input for the first layer)
        if input_data.shape[0] != self.weights.shape[1]:
            input_data = input_data.T

        if input_data.shape[0] != self.weights.shape[1]:
            raise ValueError(
                f"Layer mismatch: input_data {input_data.shape} does not match expected input size {self.weights.shape[1]}"
            )

        # Computing gradient of the loss with respect to the weights using the chain rule
        # The dot product of d_output and input_data.T gives the contribution of each weight to the loss
        # The result is averaged over the number of samples in the batch (input_data.shape[1])
        d_weights = np.dot(d_output, input_data.T) / input_data.shape[1]

        # Computing gradient of the loss with respect to the biases
        d_biases = np.mean(d_output, axis=1, keepdims=True)

        #L2 Regularization/weight decay
        d_weights += lambda_reg * self.weights

        # Computing gradient of the loss with respect to the input
        # because d_input becomes the gradient of the loss with respect to previous layer's output
        d_input = np.dot(self.weights.T, d_output)

        # Updating weights and biases
        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases

        return d_input