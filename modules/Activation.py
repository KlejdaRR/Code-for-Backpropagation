import numpy as np

class Activation:
    """Base class for activation functions."""
    def forward(self, X):
        raise NotImplementedError

    def derivative(self, X):
        raise NotImplementedError

class ReLU(Activation):
    def forward(self, X):
        return np.maximum(0, X)

    def derivative(self, X):
        return (X > 0).astype(float)

class Sigmoid(Activation):
    def forward(self, X):
        return 1 / (1 + np.exp(-X))

    def derivative(self, X):
        sig = self.forward(X)
        return sig * (1 - sig)

class Softmax(Activation):
    def forward(self, X):
        X_max = np.max(X, axis=0, keepdims=True)
        exp_X = np.exp(X - X_max)
        return exp_X / np.sum(exp_X, axis=0, keepdims=True)

    def derivative(self, X):
        S = self.forward(X)
        return S * (1 - S)

class Linear(Activation):
    def forward(self, X):
        return X

    def derivative(self, X):
        return np.ones_like(X)