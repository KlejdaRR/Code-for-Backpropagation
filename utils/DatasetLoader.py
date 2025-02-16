import numpy as np

class DatasetLoader:
    """Handles dataset loading and preprocessing for different datasets."""

    def __init__(self, dataset_type="mnist"):
        self.dataset_type = dataset_type.lower()

    def load_data(self, train_file, test_file, normalize=True, one_hot=True, num_classes=10):
        """Loads dataset based on the specified type and returns training and test sets."""
        if self.dataset_type == "mnist":
            return self._load_mnist(train_file, test_file, normalize, one_hot, num_classes)
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")

    def _load_mnist(self, train_file, test_file, normalize, one_hot, num_classes):
        """Loads MNIST-like CSV datasets, applies normalization and one-hot encoding."""
        X_train, y_train = self._load_csv(train_file, normalize, one_hot, num_classes)
        X_test, y_test = self._load_csv(test_file, normalize, one_hot, num_classes)

        val_ratio = 0.1
        val_size = int(X_train.shape[1] * val_ratio)
        X_val, y_val = X_train[:, :val_size], y_train[:, :val_size]
        X_train, y_train = X_train[:, val_size:], y_train[:, val_size:]

        return X_train, y_train, X_val, y_val, X_test, y_test

    def _load_csv(self, file_path, normalize, one_hot, num_classes):
        """Loads a CSV file and preprocesses it."""
        data = np.loadtxt(file_path, delimiter=',')
        X = data[:, 1:].T
        y = data[:, 0].astype(int)

        if normalize:
            X /= 255.0

        if one_hot:
            y_one_hot = np.zeros((num_classes, y.shape[0]))
            for i, label in enumerate(y):
                y_one_hot[label, i] = 1
            return X, y_one_hot
        else:
            return X, y