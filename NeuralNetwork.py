import numpy as np
from modules.Activation import Sigmoid
from utils.StopCriterion import StopCriterion
from modules.DropoutLayer import DropoutLayer

class NeuralNetwork:
    def __init__(self, layers, task_type="classification"):
        self.layers = layers
        self.task_type = task_type
        self.sigmoid = Sigmoid()

    def loss(self, y_true, y_pred):
        if self.task_type == "classification":
            y_pred = np.clip(y_pred, 1e-12, 1.0) # To avoid numerical instability, the predicted probabilities are clipped to a small range
            if y_true.ndim == 1 or y_true.shape[0] == 1:
                return -np.mean(np.log(y_pred[np.arange(len(y_true)), y_true.astype(int)]))
            else:
                return -np.mean(y_true * np.log(y_pred))
        elif self.task_type == "regression":
            return np.mean((y_true - y_pred) ** 2)  # MSE

    def forward(self, X):
        A = [X]
        Z = []
        current_input = X

        for layer in self.layers:
            if isinstance(layer, DropoutLayer):
                Z_current = current_input # holds output of the weighted sums plus bias
                A_current = layer.forward(Z_current) # holds output of the activation function being applied to the Z_current
            else:
                Z_current = np.dot(layer.weights, current_input) + layer.biases
                A_current = layer.activation.forward(Z_current)

            Z.append(Z_current)
            A.append(A_current)
            current_input = A_current
            # A[-1] is the output of the final layer

        return A[-1], A, Z

    def backward(self, X, y_true, y_pred, A, Z, learning_rate, lambda_reg=0.001):
        """Computing gradients using backpropagation and updating weights with learning rate."""
        L = len(self.layers)

        # Computing the gradient of the loss with respect to the output based on the task type
        if self.task_type == "classification":
            d_loss = y_pred - y_true  # Gradient of cross-entropy loss
        elif self.task_type == "regression":
            d_loss = 2 * (y_pred - y_true)  # Gradient of MSE loss
        else:
            raise ValueError("Unsupported task type. Use 'classification' or 'regression'.")

        # gradient of the loss w.r.t output
        d_output = d_loss

        for l in range(L - 1, -1, -1):
            input_data = A[l] if l > 0 else X

            if isinstance(self.layers[l], DropoutLayer):
                d_output = self.layers[l].backward(d_output, Z[l], input_data, learning_rate)
                continue

            # checking if the number of rows in input_data matches the number of columns in the layer’s weights
            if input_data.shape[0] != self.layers[l].weights.shape[1]:
                input_data = input_data.T

            # checking if the number of rows in d_output matches the number of rows in the layer’s weights
            if d_output.shape[0] != self.layers[l].weights.shape[0]:
                d_output = d_output.T

            d_output = self.layers[l].backward(d_output, Z[l], input_data, learning_rate)

    def train(self, X_train, y_train, X_val, y_val, epochs=100, learning_rate=0.001, batch_size=128, patience=30):
        stop_criterion = StopCriterion(criteria=['early_stopping', 'loss_plateau', 'max_epochs'], patience=30,
                                       loss_window=30)
        stop_criterion.set_max_epochs(epochs)

        train_losses, val_losses = [], []
        train_metrics, val_metrics = [], []
        num_samples = X_train.shape[1]
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

        for epoch in range(epochs):
            permutation = np.random.permutation(num_samples)
            X_train = X_train[:, permutation] # Shuffling of data by using permutation to prevent overfitting
            y_train = y_train[:, permutation] # Shuffling of data by using permutation to prevent overfitting

            # Performing forward and backward propagation for each mini-batch
            for i in range(0, num_samples, batch_size):
                X_batch = X_train[:, i:i + batch_size] # selecting a batch of input features to improve training efficiency
                y_batch = y_train[:, i:i + batch_size] # selecting the correspondent batch labels
                y_pred, A, O = self.forward(X_batch)
                self.backward(X_batch, y_batch, y_pred, A, O, learning_rate)

            # After updating the model's weights using mini-batches,
            # we evaluate the model on the entire training set and validation set
            y_train_pred = self.forward(X_train)[0] # we get the y_pred from the forward method
            y_val_pred = self.forward(X_val)[0]

            # calculation of the loss for the entire training set and validation set
            # in order to use it for the stopping criterion to evaluate and to evaluate the model's performance
            train_loss = self.loss(y_train, y_train_pred)
            val_loss = self.loss(y_val, y_val_pred)

            train_metric = 0
            val_metric = 0
            metric_name = ""

            if self.task_type == "classification":
                train_metric = np.mean(np.argmax(y_train_pred, axis=0) == np.argmax(y_train, axis=0)) * 100
                val_metric = np.mean(np.argmax(y_val_pred, axis=0) == np.argmax(y_val, axis=0)) * 100
                metric_name = "Accuracy (%)"
            elif self.task_type == "regression":
                mse_train = train_loss  # MSE
                mse_val = val_loss

                # Using MSE as the primary metric for regression
                train_metric = mse_train
                val_metric = mse_val
                metric_name = "MSE"

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_metrics.append(train_metric)
            val_metrics.append(val_metric)

            print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, "
                  f"Train {metric_name} = {train_metric:.4f}, Val {metric_name} = {val_metric:.4f}")

            # Checking stopping criterion
            stop, reason, best_weights = stop_criterion(epoch, train_loss, val_loss, self.layers)
            if stop:
                print(f"\nTraining stopped at epoch {epoch + 1}: {reason}")
                break

        return train_losses, val_losses, train_metrics, val_metrics

    def evaluate(self, X_test, y_test):
        y_pred = self.forward(X_test)[0]
        # after training, the loss is computed for the test set to evaluate the model's performance
        test_loss = self.loss(y_test, y_pred)

        if self.task_type == "classification":
            accuracy = np.mean(np.argmax(y_pred, axis=0) == np.argmax(y_test, axis=0)) * 100
            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
            return test_loss, accuracy
        elif self.task_type == "regression":
            mse = test_loss
            mae = np.mean(np.abs(y_test - y_pred))
            rmse = np.sqrt(mse)

            print(f"Test MSE: {mse:.4f}")
            print(f"Test MAE: {mae:.4f}")
            print(f"Test RMSE: {rmse:.4f}")

            return mse, mae, rmse