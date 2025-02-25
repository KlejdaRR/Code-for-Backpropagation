import numpy as np
import modules.BaseLayer as BaseLayer

# Neurons are randomly dropped out with probability p
# The outputs of the remaining neurons are scaled up by 1/1-p to maintain the expected value of the output
class DropoutLayer:
    def __init__(self, rate):
        self.rate = rate

    # The mask is a binary matrix that determines which neurons are dropped out during training
    def forward(self, X, training=True):
        if training:
            self.mask = np.random.binomial(1, 1 - self.rate, size=X.shape) / (1 - self.rate) # scaling
            return X * self.mask
        return X  # No dropout during inference/testing

    # During backpropagation, the gradient of the loss with respect to the input (d_output) is multiplied by the mask
    # This way gradients are only propagated through the neurons that were kept during the forward pass
    def backward(self, d_output, Z, input_data, learning_rate):
        return d_output * self.mask