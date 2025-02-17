import numpy as np
from modules.Activation import Sigmoid
from utils.StopCriterion import StopCriterion

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.sigmoid = Sigmoid()

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

    def backward(self, X, y_true, y_pred, A, O, learning_rate, lambda_reg=0.01):
        """Computing gradients using backpropagation and updating weights with learning rate."""
        L = len(self.layers)
        G = [None] * L
        d_loss = y_pred - y_true

        for l in range(L - 1, -1, -1):
            if l == L - 1:
                Delta = self.sigmoid.derivative(A[l]) * d_loss
            else:
                Wl_plus_1 = self.layers[l + 1].weights
                Delta = self.sigmoid.derivative(A[l]) * np.matmul(Wl_plus_1.T, Delta)

            if l > 0:
                Wl_grad = np.matmul(Delta, O[l - 1].T)
            else:
                Wl_grad = np.matmul(Delta, X.T)

            bl_grad = np.sum(Delta, axis=1, keepdims=True)

            G[l] = (Wl_grad, bl_grad)

        for l in range(L):
            W_grad, b_grad = G[l]
            self.layers[l].weights -= learning_rate * (
                        W_grad + lambda_reg * self.layers[l].weights)  # L2 Regularization
            self.layers[l].biases -= learning_rate * b_grad

    def train(self, X_train, y_train, X_val, y_val, epochs=100, learning_rate=0.001, batch_size=128, patience=10):
        """
        Train the neural network with early stopping and loss plateau detection.

        Args:
            X_train (numpy array): Training data.
            y_train (numpy array): Training labels.
            X_val (numpy array): Validation data.
            y_val (numpy array): Validation labels.
            epochs (int): Maximum number of epochs.
            learning_rate (float): Learning rate.
            batch_size (int): Mini-batch size.
            patience (int): Number of epochs for early stopping.

        Returns:
            train_losses, val_losses, train_accuracies, val_accuracies
        """
        stop_criterion = StopCriterion(criteria=['early_stopping', 'loss_plateau', 'max_epochs'], patience=patience)
        stop_criterion.set_max_epochs(epochs)

        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []

        num_samples = X_train.shape[1]

        for epoch in range(epochs):
            # Shuffle training data
            permutation = np.random.permutation(num_samples)
            X_train, y_train = X_train[:, permutation], y_train[:, permutation]

            for i in range(0, num_samples, batch_size):
                X_batch = X_train[:, i:i + batch_size]
                y_batch = y_train[:, i:i + batch_size]
                y_pred, A, O = self.forward(X_batch)

                self.backward(X_batch, y_batch, y_pred, A, O, learning_rate)

            # Compute training and validation loss
            y_train_pred = self.forward(X_train)[0]
            y_val_pred = self.forward(X_val)[0]

            train_loss = self.loss(y_train, y_train_pred)
            val_loss = self.loss(y_val, y_val_pred)

            train_accuracy = np.mean(np.argmax(y_train_pred, axis=0) == np.argmax(y_train, axis=0)) * 100
            val_accuracy = np.mean(np.argmax(y_val_pred, axis=0) == np.argmax(y_val, axis=0)) * 100

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)

            print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, "
                  f"Train Acc = {train_accuracy:.2f}%, Val Acc = {val_accuracy:.2f}%")

            # Check stopping criterion
            stop, reason, best_weights = stop_criterion(epoch, train_loss, val_loss, self.layers)
            if stop:
                print(f"\nTraining stopped at epoch {epoch + 1}: {reason}")
                break

        return train_losses, val_losses, train_accuracies, val_accuracies

    def evaluate(self, X_test, y_test):
        y_pred = self.forward(X_test)[0]
        test_loss = self.loss(y_test, y_pred)
        accuracy = np.mean(np.argmax(y_pred, axis=0) == np.argmax(y_test, axis=0)) * 100
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
        return test_loss, accuracy


