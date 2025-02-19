import numpy as np
import modules.BaseLayer as BaseLayer

class DropoutLayer:
    def __init__(self, rate):
        self.rate = rate

    def forward(self, X, training=True):
        if training:
            self.mask = np.random.binomial(1, 1 - self.rate, size=X.shape) / (1 - self.rate)
            return X * self.mask
        return X  # No dropout during inference

    def backward(self, d_output, Z, input_data, learning_rate):
        return d_output * self.mask