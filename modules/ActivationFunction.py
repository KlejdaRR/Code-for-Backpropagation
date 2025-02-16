import numpy as np

class ActivationFunction:
    def __init__(self, function):
        self.function_name = function
        self.function = getattr(self, function)

    def __call__(self, X):
        return self.function(X)

    def relu(self, X):
        return np.maximum(0, X)

    def softmax(self, X):
        X_max = np.max(X, axis=0, keepdims=True)
        exp_X = np.exp(X - X_max)
        return exp_X / np.sum(exp_X, axis=0, keepdims=True)

    def derivative_relu(self, X):
        return (X > 0).astype(float)
