import os
import pickle
import numpy as np
import pandas as pd

class DatasetLoader:
    """Handling dataset loading and preprocessing for different datasets."""

    def __init__(self, dataset_type="mnist", custom_path=None):
        self.dataset_type = dataset_type.lower()
        self.custom_path = custom_path  # Path for custom datasets

    def load_data(self, train_file=None, test_file=None, normalize=True, one_hot=True, num_classes=10):
        """Loading dataset based on type (MNIST, CIFAR-10, Wine Quality, or custom dataset)."""
        if self.dataset_type == "mnist":
            return self._load_mnist_csv(train_file, test_file, normalize, one_hot, num_classes)
        elif self.dataset_type == "cifar10":
            return self._load_cifar10(train_file, normalize, one_hot, num_classes)
        elif self.dataset_type == "wine_quality":
            return self._load_wine_quality(normalize)
        elif self.dataset_type == "custom":
            return self._load_custom_dataset(normalize, one_hot, num_classes)
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")

    def _load_wine_quality(self, normalize=True):
        """Loading the Wine Quality dataset from the wine+quality folder for the regression task experiment."""

        red_wine_path = os.path.join("wine+quality", "winequality-red.csv")
        white_wine_path = os.path.join("wine+quality", "winequality-white.csv")

        if not os.path.exists(red_wine_path) and not os.path.exists(white_wine_path):
            raise FileNotFoundError(
                f"No Wine Quality dataset files found in 'wine+quality' folder. "
                f"Please ensure 'winequality-red.csv' or 'winequality-white.csv' exists in the folder."
            )

        if os.path.exists(red_wine_path):
            print("Loading red wine dataset...")
            data = pd.read_csv(red_wine_path, delimiter=';')
        else:
            print("Loading white wine dataset...")
            data = pd.read_csv(white_wine_path, delimiter=';')

        print(f"Loaded Wine Quality dataset with shape: {data.shape}")

        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        if normalize:
            X = (X - X.mean(axis=0)) / X.std(axis=0)

        split = int(0.8 * X.shape[0])
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]

        val_size = int(X_train.shape[0] * 0.1)
        X_val, y_val = X_train[:val_size], y_train[:val_size]
        X_train, y_train = X_train[val_size:], y_train[val_size:]

        return X_train, y_train, X_val, y_val, X_test, y_test

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
            y_train = self._one_hot_encode(y_train, num_classes).T
            y_test = self._one_hot_encode(y_test, num_classes).T

        val_size = int(X_train.shape[0] * 0.1)
        X_val, y_val = X_train[:val_size], y_train[:, :val_size]
        X_train, y_train = X_train[val_size:], y_train[:, val_size:]

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

        X_train = X_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        X_test = X_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

        if one_hot:
            y_train = self._one_hot_encode(y_train, num_classes)
            y_test = self._one_hot_encode(y_test, num_classes)

        val_size = int(X_train.shape[0] * 0.1)
        X_val, y_val = X_train[:val_size], y_train[:val_size]
        X_train, y_train = X_train[val_size:], y_train[val_size:]

        return X_train, y_train, X_val, y_val, X_test, y_test

    def _load_custom_dataset(self, normalize=True, one_hot=True, num_classes=10):
        """Loading a custom dataset from CSV or NumPy format."""
        if self.custom_path is None:
            raise ValueError("Custom dataset requires a valid path.")

        if self.custom_path.endswith(".csv"):
            print("Loading custom CSV dataset...")
            data = pd.read_csv(self.custom_path, header=None).values
            X, y = data[:, 1:], data[:, 0]

            y = y.astype(int)

            if y.ndim == 2 and y.shape[1] == num_classes:
                one_hot = False  # Skip one-hot encoding
            elif one_hot:
                y = self._one_hot_encode(y, num_classes)
            else:
                y = y.reshape(1, -1)

        elif self.custom_path.endswith(".npz"):
            print("Loading custom NumPy dataset...")
            data = np.load(self.custom_path)
            X_train, y_train = data["X_train"], data["y_train"]
            X_test, y_test = data["X_test"], data["y_test"]

            y_train = y_train.astype(int)
            y_test = y_test.astype(int)

            if y_train.ndim == 2 and y_train.shape[1] == num_classes:
                one_hot = False
            elif one_hot:
                y_train = self._one_hot_encode(y_train, num_classes)
                y_test = self._one_hot_encode(y_test, num_classes)
            else:
                y_train = y_train.reshape(1, -1)
                y_test = y_test.reshape(1, -1)

            return self._prepare_data(X_train, y_train, X_test, y_test, normalize, one_hot, num_classes)

        else:
            raise ValueError("Unsupported file format. Use .csv or .npz.")

        split = int(0.8 * X.shape[0])
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]

        return self._prepare_data(X_train, y_train, X_test, y_test, normalize, one_hot, num_classes)

    def _prepare_data(self, X_train, y_train, X_test, y_test, normalize, one_hot, num_classes):
        """Preparing dataset (normalization, one-hot encoding, splitting validation)."""
        if normalize:
            X_train = X_train.astype("float32") / 255.0
            X_test = X_test.astype("float32") / 255.0

        if y_train.ndim == 2 and y_train.shape[1] == num_classes:
            one_hot = False
        elif one_hot:
            y_train = self._one_hot_encode(y_train, num_classes)
            y_test = self._one_hot_encode(y_test, num_classes)

        if y_train.ndim == 2 and y_train.shape[0] != num_classes:
            y_train = y_train.T
        if y_test.ndim == 2 and y_test.shape[0] != num_classes:
            y_test = y_test.T

        val_size = int(X_train.shape[0] * 0.1)
        X_val, y_val = X_train[:val_size], y_train[:, :val_size]
        X_train, y_train = X_train[val_size:], y_train[:, val_size:]

        return X_train, y_train, X_val, y_val, X_test, y_test

    def _one_hot_encode(self, labels, num_classes):
        """Converting labels to one-hot encoding."""
        if labels.ndim != 1:
            raise ValueError("Labels must be a 1D array of class indices.")

        labels = labels.astype(int)
        one_hot = np.zeros((labels.shape[0], num_classes))
        one_hot[np.arange(labels.shape[0]), labels] = 1
        return one_hot