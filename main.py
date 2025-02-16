import numpy as np
import argparse
import matplotlib.pyplot as plt
from utils.DatasetLoader import DatasetLoader
from modules.DenseLayer import DenseLayer
from NeuralNetwork import NeuralNetwork

def plot_results(train_losses, val_losses, dataset_name):
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f"Training Loss vs Validation Loss ({dataset_name.upper()})")
    plt.show()

if __name__ == "__main__":
    np.random.seed(0)

    parser = argparse.ArgumentParser(description="Train a neural network on MNIST or CIFAR-10.")
    parser.add_argument("--dataset", type=str, choices=["mnist", "cifar10"], default="mnist",
                        help="Choose dataset to train on (mnist or cifar10).")
    args = parser.parse_args()

    dataset_name = args.dataset
    print(f"Loading {dataset_name.upper()} dataset...")

    dataset_loader = DatasetLoader(dataset_type=dataset_name)

    if dataset_name == "mnist":
        train_file = "data/mnist_train.csv"
        test_file = "data/mnist_test.csv"
        X_train, y_train, X_val, y_val, X_test, y_test = dataset_loader.load_data(train_file, test_file)
    elif dataset_name == "cifar10":
        dataset_path = "cifar-10-python/cifar-10-batches-py"
        test_cifar = "cifar-10-python/cifar-10-batches-py/test_batch"
        X_train, y_train, X_val, y_val, X_test, y_test = dataset_loader.load_data(dataset_path, test_cifar)

    print(f"Train Set: {X_train.shape}, {y_train.shape}")
    print(f"Validation Set: {X_val.shape}, {y_val.shape}")
    print(f"Test Set: {X_test.shape}, {y_test.shape}")

    input_size = 784 if dataset_name == "mnist" else 32 * 32 * 3  # MNIST (28x28), CIFAR-10 (32x32x3)

    nn = NeuralNetwork([
        DenseLayer(input_size, 256, activation="relu", initialization="he"),
        DenseLayer(256, 128, activation="relu", initialization="he"),
        DenseLayer(128, 10, activation="softmax", initialization="xavier")
    ])

    if dataset_name == "cifar10":
        X_train = X_train.reshape(X_train.shape[0], -1).T
        X_val = X_val.reshape(X_val.shape[0], -1).T
        X_test = X_test.reshape(X_test.shape[0], -1).T
    else:
        X_train, X_val, X_test = X_train.T, X_val.T, X_test.T

    y_train, y_val, y_test = y_train.T, y_val.T, y_test.T

    train_losses, val_losses = nn.train(X_train, y_train, X_val, y_val, epochs=50, learning_rate=0.01, batch_size=64)

    plot_results(train_losses, val_losses, dataset_name)

    nn.evaluate(X_test, y_test)