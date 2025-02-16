import os
import pickle
import numpy as np
import pandas as pd

class DatasetLoader:
    """Handles dataset loading and preprocessing for different datasets."""

    def __init__(self, dataset_type="mnist"):
        self.dataset_type = dataset_type.lower()

    def load_data(self, train_file, test_file, normalize=True, one_hot=True, num_classes=10):
        """Loading the dataset based on the specified type."""
        if self.dataset_type == "mnist":
            return self._load_mnist_csv(train_file, test_file, normalize, one_hot, num_classes)
        elif self.dataset_type == "cifar10":
            return self._load_cifar10(train_file, normalize, one_hot, num_classes)
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")

    def _load_mnist_csv(self, train_file, test_file, normalize, one_hot, num_classes):
        """Loading MNIST dataset from CSV files."""
        train_data = pd.read_csv(train_file).values
        test_data = pd.read_csv(test_file).values

        X_train, y_train = train_data[:, 1:], train_data[:, 0]
        X_test, y_test = test_data[:, 1:], test_data[:, 0]

        if normalize:
            X_train = X_train.astype("float32") / 255.0
            X_test = X_test.astype("float32") / 255.0

        if one_hot:
            y_train = self._one_hot_encode(y_train, num_classes)
            y_test = self._one_hot_encode(y_test, num_classes)

        val_size = int(X_train.shape[0] * 0.1)
        X_val, y_val = X_train[:val_size], y_train[:val_size]
        X_train, y_train = X_train[val_size:], y_train[val_size:]

        return X_train, y_train, X_val, y_val, X_test, y_test

    def _load_cifar10(self, dataset_path, normalize=True, one_hot=True, num_classes=10):
        """Loading CIFAR-10 dataset from binary files."""
        def load_batch(file):
            with open(file, 'rb') as fo:
                batch = pickle.load(fo, encoding='bytes')
            X = batch[b'data']
            y = np.array(batch[b'labels'])
            return X, y

        X_train = []
        y_train = []

        for i in range(1, 6):
            file_path = os.path.join(dataset_path, f"data_batch_{i}")
            X, y = load_batch(file_path)
            X_train.append(X)
            y_train.append(y)

        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)

        test_file_path = os.path.join(dataset_path, "test_batch")
        X_test, y_test = load_batch(test_file_path)

        if normalize:
            X_train = X_train.astype("float32") / 255.0
            X_test = X_test.astype("float32") / 255.0

        # Reshaping the data to 3D image format (32x32x3)
        X_train = X_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        X_test = X_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

        if one_hot:
            y_train = self._one_hot_encode(y_train, num_classes)
            y_test = self._one_hot_encode(y_test, num_classes)

        # Splitting the validation set (10% of training data)
        val_size = int(X_train.shape[0] * 0.1)
        X_val, y_val = X_train[:val_size], y_train[:val_size]
        X_train, y_train = X_train[val_size:], y_train[val_size:]

        return X_train, y_train, X_val, y_val, X_test, y_test

    def _one_hot_encode(self, labels, num_classes):
        one_hot = np.zeros((labels.shape[0], num_classes))
        one_hot[np.arange(labels.shape[0]), labels] = 1
        return one_hot
