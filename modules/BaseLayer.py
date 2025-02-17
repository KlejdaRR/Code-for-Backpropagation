class BaseLayer:
    """Abstract class for layers in the network."""

    def forward(self, X):
        raise NotImplementedError

    def backward(self, d_output, Z, input_data, learning_rate):
        raise NotImplementedError
