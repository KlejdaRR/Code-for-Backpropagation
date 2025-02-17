import numpy as np
import argparse
import matplotlib.pyplot as plt
from utils.DatasetLoader import DatasetLoader
from modules.DenseLayer import DenseLayer
from NeuralNetwork import NeuralNetwork


def plot_results(train_losses, val_losses, train_accuracies, val_accuracies, dataset_name):
    fig, ax1 = plt.subplots(2, 1, figsize=(10, 8))

    # Loss Plot
    ax1[0].plot(train_losses, label='Training Loss', color='blue')
    ax1[0].plot(val_losses, label='Validation Loss', color='orange')
    ax1[0].set_xlabel('Epochs')
    ax1[0].set_ylabel('Loss')
    ax1[0].set_title(f"Training vs Validation Loss ({dataset_name.upper()})")
    ax1[0].legend()

    # Accuracy Plot
    ax1[1].plot(train_accuracies, label='Training Accuracy', color='green')
    ax1[1].plot(val_accuracies, label='Validation Accuracy', color='red')
    ax1[1].set_xlabel('Epochs')
    ax1[1].set_ylabel('Accuracy (%)')
    ax1[1].set_title(f"Training vs Validation Accuracy ({dataset_name.upper()})")
    ax1[1].legend()

    plt.tight_layout()
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

    input_size = 784 if dataset_name == "mnist" else 32 * 32 * 3  # CIFAR-10 needs 3072
    nn = NeuralNetwork([
        DenseLayer(input_size, 512, activation="relu", initialization="he"),  # Now correctly set for CIFAR-10
        DenseLayer(512, 256, activation="relu", initialization="he"),
        DenseLayer(256, 128, activation="relu", initialization="he"),
        DenseLayer(128, 10, activation="softmax", initialization="xavier")
    ])

    if dataset_name == "cifar10":
        X_train = X_train.reshape(X_train.shape[0], -1).T  # Reshape to (3072, num_samples)
        X_val = X_val.reshape(X_val.shape[0], -1).T
        X_test = X_test.reshape(X_test.shape[0], -1).T
    else:
        X_train, X_val, X_test = X_train.T, X_val.T, X_test.T

    y_train, y_val, y_test = y_train.T, y_val.T, y_test.T

    train_losses, val_losses, train_accuracies, val_accuracies = nn.train(
        X_train, y_train, X_val, y_val, epochs=100, learning_rate=0.001, batch_size=128, patience=10
    )

    plot_results(train_losses, val_losses, train_accuracies, val_accuracies, dataset_name)

    nn.evaluate(X_test, y_test)