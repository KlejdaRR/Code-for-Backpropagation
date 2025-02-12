import numpy as np

class NeuralNetwork:

    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, y_true, y_pred, learning_rate):
        loss_grad = (y_pred - y_true)
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad, learning_rate)

    def train(self, X_train, y_train, X_val, y_val, epochs=1000, learning_rate=0.1, patience=10):
        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        no_improve_count = 0

        for epoch in range(epochs):
            y_pred = self.forward(X_train)
            train_loss = np.mean((y_pred - y_train) ** 2)
            self.backward(y_train, y_pred, learning_rate)

            val_pred = self.forward(X_val)
            val_loss = np.mean((val_pred - y_val) ** 2)

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
        y_pred = self.forward(X_test)
        test_loss = np.mean((y_pred - y_test) ** 2)
        accuracy = np.mean(np.argmax(y_pred, axis=0) == np.argmax(y_test, axis=0)) * 100
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
        return test_loss, accuracy


