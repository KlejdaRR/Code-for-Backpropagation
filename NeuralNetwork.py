import numpy as np

def sigmoid(x, compute_derivative=False):
    sig = 1 / (1 + np.exp(-x))
    if compute_derivative:
        return sig * (1 - sig)
    return sig

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        A = []
        O = []

        for layer in self.layers:
            X = layer.forward(X)
            A.append(layer.pre_activation)
            O.append(X)

        return X, A, O

    def loss(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-12, 1.0)
        return -np.mean(y_true * np.log(y_pred))

    def backward(self, X, y_true, y_pred, A, O):
        """Computing of  gradients using backpropagation."""
        L = len(self.layers)
        G = [None] * L
        d_loss = y_pred - y_true

        for l in range(L - 1, -1, -1):
            if l == L - 1:
                Delta = sigmoid(A[l], compute_derivative=True) * d_loss
            else:
                Wl_plus_1 = self.layers[l + 1].weights
                Delta = sigmoid(A[l], compute_derivative=True) * np.matmul(Wl_plus_1.T, Delta)

            if l > 0:
                Wl_grad = np.matmul(Delta, O[l - 1].T)
            else:
                Wl_grad = np.matmul(Delta, X.T)

            bl_grad = np.sum(Delta, axis=1, keepdims=True)

            G[l] = (Wl_grad, bl_grad)

        for l in range(L):
            W_grad, b_grad = G[l]
            self.layers[l].weights -= 0.001 * W_grad
            self.layers[l].biases -= 0.001 * b_grad

    def train(self, X_train, y_train, X_val, y_val, epochs=50, learning_rate=0.001, batch_size=64, patience=10):
        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        no_improve_count = 0

        num_samples = X_train.shape[1]
        for epoch in range(epochs):
            permutation = np.random.permutation(num_samples)
            X_train = X_train[:, permutation]
            y_train = y_train[:, permutation]

            for i in range(0, num_samples, batch_size):
                X_batch = X_train[:, i:i + batch_size]
                y_batch = y_train[:, i:i + batch_size]

                y_pred, A, O = self.forward(X_batch)
                self.backward(X_batch, y_batch, y_pred, A, O)

            train_loss = self.loss(y_train, self.forward(X_train)[0])
            val_loss = self.loss(y_val, self.forward(X_val)[0])

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= patience:
                print("Early stopping triggered!")
                break

        return train_losses, val_losses

    def evaluate(self, X_test, y_test):
        y_pred = self.forward(X_test)[0]
        test_loss = self.loss(y_test, y_pred)
        accuracy = np.mean(np.argmax(y_pred, axis=0) == np.argmax(y_test, axis=0)) * 100
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
        return test_loss, accuracy


