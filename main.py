import argparse
import matplotlib.pyplot as plt
import numpy as np
from utils.DatasetLoader import DatasetLoader
from NeuralNetwork import NeuralNetwork
from modules.DenseLayer import DenseLayer

import matplotlib.pyplot as plt

def plot_results(train_losses, val_losses, train_accuracies, val_accuracies, dataset_name):
    """Plots training loss and accuracy with epochs on the x-axis."""
    epochs = range(1, len(train_losses) + 1)

    fig, ax1 = plt.subplots(2, 1, figsize=(10, 8))

    # Plot training and validation loss
    ax1[0].plot(epochs, train_losses, label='Training Loss', color='blue')
    ax1[0].plot(epochs, val_losses, label='Validation Loss', color='orange')
    ax1[0].set_title(f"Training vs Validation Loss ({dataset_name.upper()})")
    ax1[0].set_xlabel('Epochs')
    ax1[0].set_ylabel('Loss')
    ax1[0].legend()

    # Plot training and validation accuracy
    ax1[1].plot(epochs, train_accuracies, label='Training Accuracy', color='green')
    ax1[1].plot(epochs, val_accuracies, label='Validation Accuracy', color='red')
    ax1[1].set_title(f"Training vs Validation Accuracy ({dataset_name.upper()})")
    ax1[1].set_xlabel('Epochs')
    ax1[1].set_ylabel('Accuracy')
    ax1[1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural network on MNIST, CIFAR-10, or a custom dataset.")
    parser.add_argument("--dataset", type=str, default="mnist", help="Choose dataset (mnist, cifar10, custom).")
    parser.add_argument("--custom_path", type=str, default=None, help="Path to custom dataset file (CSV or NPZ).")
    args = parser.parse_args()

    dataset_loader = None

    if args.dataset == "mnist":
        train_file = "data/mnist_train.csv"  # Ensure this path is correct
        test_file = "data/mnist_test.csv"
        dataset_loader = DatasetLoader(dataset_type="mnist")
        X_train, y_train, X_val, y_val, X_test, y_test = dataset_loader.load_data(train_file, test_file)

    elif args.dataset == "cifar10":
        dataset_path = "cifar-10-python/cifar-10-batches-py"  # Ensure this path exists
        dataset_loader = DatasetLoader(dataset_type="cifar10")
        X_train, y_train, X_val, y_val, X_test, y_test = dataset_loader.load_data(dataset_path)

    elif args.dataset == "custom" and args.custom_path:
        dataset_loader = DatasetLoader(dataset_type="custom", custom_path=args.custom_path)
        X_train, y_train, X_val, y_val, X_test, y_test = dataset_loader.load_data()

    else:
        raise ValueError("Invalid dataset option. Use 'mnist', 'cifar10', or 'custom' with --custom_path.")

    print(f"Dataset: {args.dataset.upper()}")
    print(f"Train Set: {X_train.shape}, {y_train.shape}")
    print(f"Validation Set: {X_val.shape}, {y_val.shape}")
    print(f"Test Set: {X_test.shape}, {y_test.shape}")

    if args.dataset == "mnist":
        input_size = 784  # MNIST images are 28x28 pixels (flattened to 784)

    elif args.dataset == "cifar10":
        input_size = 32 * 32 * 3  # CIFAR-10 images are 32x32 pixels with 3 color channels (3072)

    elif args.dataset == "custom":
        input_size = X_train.shape[1]  # Use the number of features from the custom dataset

    else:
        raise ValueError("Invalid dataset option. Use 'mnist', 'cifar10', or 'custom' with --custom_path.")

    num_classes = y_train.shape[1] if y_train.ndim == 2 else len(np.unique(y_train))

    print(f"Number of output classes detected: {num_classes}")

    nn = NeuralNetwork([
        DenseLayer(input_size, 512, activation="relu", initialization="he"),
        DenseLayer(512, 256, activation="relu", initialization="he"),
        DenseLayer(256, 128, activation="relu", initialization="he"),
        DenseLayer(128, num_classes, activation="softmax", initialization="xavier")
    ])

    if args.dataset == "cifar10":
        # CIFAR-10 images need to be flattened to (3072, num_samples)
        X_train = X_train.reshape(X_train.shape[0], -1).T
        X_val = X_val.reshape(X_val.shape[0], -1).T
        X_test = X_test.reshape(X_test.shape[0], -1).T
    elif args.dataset == "custom":
        if X_train.ndim == 2:
            X_train, X_val, X_test = X_train.T, X_val.T, X_test.T
        elif X_train.ndim == 4:
            X_train = X_train.reshape(X_train.shape[0], -1).T
            X_val = X_val.reshape(X_val.shape[0], -1).T
            X_test = X_test.reshape(X_test.shape[0], -1).T
        else:
            raise ValueError(
                "Unsupported custom dataset shape. Ensure it is structured as (samples, features) or images.")
    else:
        # Default: MNIST or any dataset that is already (samples, features)
        X_train, X_val, X_test = X_train.T, X_val.T, X_test.T

    y_train, y_val, y_test = y_train.T, y_val.T, y_test.T

    best_acc = 0
    best_params = {}

    for lr in [0.001, 0.0005, 0.0003, 0.0001]:
        for batch_size in [16, 32, 64, 128]:
            print(f"Training with learning_rate={lr}, batch_size={batch_size}")

            train_losses, val_losses, train_accuracies, val_accuracies = nn.train(
                X_train, y_train, X_val, y_val, epochs=100, learning_rate=lr, batch_size=batch_size, patience=15
            )

            final_acc = val_accuracies[-1]
            print(f"Final Accuracy: {final_acc:.2f}% with lr={lr}, batch_size={batch_size}")

            if final_acc > best_acc:
                best_acc = final_acc
                best_params = {'learning_rate': lr, 'batch_size': batch_size}

    print(f"\nBest Accuracy: {best_acc:.2f}% with {best_params}")

    plot_results(train_losses, val_losses, train_accuracies, val_accuracies, args.dataset)
    nn.evaluate(X_test, y_test)