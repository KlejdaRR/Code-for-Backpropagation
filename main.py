import argparse
import matplotlib.pyplot as plt
import numpy as np
from utils.DatasetLoader import DatasetLoader
from NeuralNetwork import NeuralNetwork
from modules.DenseLayer import DenseLayer
from modules.DropoutLayer import DropoutLayer
import os

def plot_results(train_losses, val_losses, train_metrics, val_metrics, dataset_name, task_type):
    """Plot training and validation loss/metric over epochs based on task type."""
    epochs = np.arange(1, len(train_losses) + 1)

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # Plot training and validation loss
    ax[0].plot(epochs, train_losses, label='Training Loss', color='blue')
    ax[0].plot(epochs, val_losses, label='Validation Loss', color='orange')
    ax[0].set_title(f"Training vs Validation Loss ({dataset_name.upper()})")
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss (MSE for regression, Cross-Entropy for classification)')
    ax[0].legend()

    if task_type == "classification":
        # Plot accuracy for classification
        ax[1].plot(epochs, train_metrics, label='Training Accuracy', color='green')
        ax[1].plot(epochs, val_metrics, label='Validation Accuracy', color='red')
        ax[1].set_title(f"Training vs Validation Accuracy ({dataset_name.upper()})")
        ax[1].set_ylabel('Accuracy (%)')
    elif task_type == "regression":
        # Plot MSE for regression
        ax[1].plot(epochs, train_metrics, label='Training MSE', color='green')
        ax[1].plot(epochs, val_metrics, label='Validation MSE', color='red')
        ax[1].set_title(f"Training vs Validation MSE ({dataset_name.upper()})")
        ax[1].set_ylabel('Mean Squared Error (MSE)')

    ax[1].set_xlabel('Epochs')
    ax[1].legend()

    plt.tight_layout()
    plt.show()

def load_dataset(dataset_type, custom_path=None):
    """Loading the dataset based on the type and custom path."""

    if dataset_type == "mnist":
        train_file = "data/mnist_train.csv"
        test_file = "data/mnist_test.csv"
        dataset_loader = DatasetLoader(dataset_type="mnist")
        X_train, y_train, X_val, y_val, X_test, y_test = dataset_loader.load_data(train_file, test_file)

    elif dataset_type == "cifar10":
        dataset_path = "cifar-10-python/cifar-10-batches-py"
        dataset_loader = DatasetLoader(dataset_type="cifar10")
        X_train, y_train, X_val, y_val, X_test, y_test = dataset_loader.load_data(dataset_path)

    elif dataset_type == "custom":
        if custom_path is None:
            raise ValueError("Custom dataset requires --custom_path argument.")
        dataset_loader = DatasetLoader(dataset_type="custom", custom_path=custom_path)
        X_train, y_train, X_val, y_val, X_test, y_test = dataset_loader.load_data()

    else:
        raise ValueError("Invalid dataset option. Use 'mnist', 'cifar10', or 'custom' with --custom_path.")

    print(f"Dataset: {dataset_type.upper()}")
    print(f"Train Set: {X_train.shape}, {y_train.shape}")
    print(f"Validation Set: {X_val.shape}, {y_val.shape}")
    print(f"Test Set: {X_test.shape}, {y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test

def reshape_data(X_train, X_val, X_test, dataset_type):
    if dataset_type == "cifar10":
        X_train = X_train.reshape(X_train.shape[0], -1).T
        X_val = X_val.reshape(X_val.shape[0], -1).T
        X_test = X_test.reshape(X_test.shape[0], -1).T
    elif dataset_type == "wine_quality":
        X_train, X_val, X_test = X_train.T, X_val.T, X_test.T
    elif dataset_type == "custom":
        if X_train.ndim == 2:
            X_train, X_val, X_test = X_train.T, X_val.T, X_test.T
        # checking if we have image dataset
        elif X_train.ndim == 4:
            X_train = X_train.reshape(X_train.shape[0], -1).T
            X_val = X_val.reshape(X_val.shape[0], -1).T
            X_test = X_test.reshape(X_test.shape[0], -1).T
        else:
            raise ValueError("Unsupported custom dataset shape.")
    else:
        X_train, X_val, X_test = X_train.T, X_val.T, X_test.T

    return X_train, X_val, X_test

def get_input_size(X_train, dataset_type):
    """Returning the input size based on the dataset type."""
    if dataset_type == "mnist":
        return 784  # MNIST images are 28x28 pixels (flattened to 784)
    elif dataset_type == "cifar10":
        return 32 * 32 * 3  # CIFAR-10 images are 32x32 pixels with 3 color channels (3072)
    elif dataset_type == "wine_quality":
        return X_train.shape[1]
    elif dataset_type == "custom":
        return X_train.shape[1]  # Using the number of features from the custom dataset
    else:
        raise ValueError("Invalid dataset option. Use 'mnist', 'cifar10', 'wine_quality', or 'custom' with --custom_path.")

def create_model(input_size, num_classes, task_type):
    output_activation = "softmax" if task_type == "classification" else "linear"

    return NeuralNetwork([
        DenseLayer(input_size, 512, activation="relu", initialization="he"),
        DropoutLayer(0.3),
        DenseLayer(512, 256, activation="relu", initialization="he"),
        DropoutLayer(0.3),
        DenseLayer(256, 128, activation="relu", initialization="he"),
        DropoutLayer(0.3),
        DenseLayer(128, num_classes, activation=output_activation, initialization="xavier")
    ], task_type=task_type)


def main():
    parser = argparse.ArgumentParser(
        description="Train a neural network on MNIST, CIFAR-10, Wine Quality, or a custom dataset.")
    parser.add_argument("--dataset", type=str, default="mnist",
                        help="Choose dataset (mnist, cifar10, wine_quality, custom).")
    parser.add_argument("--custom_path", type=str, default=None, help="Path to custom dataset file (CSV or NPZ).")
    parser.add_argument("--task_type", type=str, default="classification",
                        help="Choose task type (classification, regression).")
    args = parser.parse_args()

    # Loading datasets

    # (X_train, y_train) serves to train the neural network model: minimize loss function and train model's parameters

    # (X_val, y_val) serves to evaluate the model on the validation dataset,
    # also the StopCriterion class uses the validation loss to determine when to stop training

    # (X_test, y_test) serves to evaluate the model on the test dataset using the evaluate method in the NeuralNetwork class

    if args.dataset == "wine_quality":
        dataset_loader = DatasetLoader(dataset_type="wine_quality", custom_path=args.custom_path)
        X_train, y_train, X_val, y_val, X_test, y_test = dataset_loader.load_data()
    else:
        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(args.dataset, args.custom_path)

    print(f"Dataset: {args.dataset.upper()}")
    print(f"Train Set: {X_train.shape}, {y_train.shape}")
    print(f"Validation Set: {X_val.shape}, {y_val.shape}")
    print(f"Test Set: {X_test.shape}, {y_test.shape}")

    # Reshaping data if necessary
    X_train, X_val, X_test = reshape_data(X_train, X_val, X_test, args.dataset)

    # Determining the number of classes (num_classes) based on the task type
    input_size = X_train.shape[0]
    if args.task_type == "classification":
        num_classes = y_train.shape[0] if y_train.ndim == 2 else len(np.unique(y_train))
    elif args.task_type == "regression":
        num_classes = 1  # For regression, the output is a single value
        # Reshaping of y_train, y_val, and y_test to (1, num_samples) for regression
        y_train = y_train.reshape(1, -1)
        y_val = y_val.reshape(1, -1)
        y_test = y_test.reshape(1, -1)
    else:
        raise ValueError("Unsupported task type. Use 'classification' or 'regression'.")

    # training the model
    best_model = None
    best_metric = 0 if args.task_type == "classification" else float('inf')
    best_params = {}

    for lr in [0.01, 0.001, 0.005, 0.0005]:
        for batch_size in [32, 64, 128]:
            print(f"Training with learning_rate={lr}, batch_size={batch_size}")

            nn = create_model(input_size, num_classes, task_type=args.task_type)
            train_losses, val_losses, train_metrics, val_metrics = nn.train(
                X_train, y_train, X_val, y_val, epochs=100, learning_rate=lr, batch_size=batch_size)

            final_metric = val_metrics[-1]
            print(f"Final Metric: {final_metric:.2f} with lr={lr}, batch_size={batch_size}")

            if (args.task_type == "classification" and final_metric > best_metric) or \
                    (args.task_type == "regression" and final_metric < best_metric):
                best_metric = final_metric
                best_params = {'learning_rate': lr, 'batch_size': batch_size}
                best_model = nn

    # Retraining the best model on the combined training and validation sets
    print("\nRetraining the best model on the combined training and validation sets...")
    X_combined = np.concatenate((X_train, X_val), axis=1)
    y_combined = np.concatenate((y_train, y_val), axis=1)

    best_model.train(X_combined, y_combined, X_test, y_test, epochs=100, learning_rate=best_params['learning_rate'],
                     batch_size=best_params['batch_size'])

    # Evaluating the best model on the test set
    print("\nEvaluating the best model on the test set...")
    best_model.evaluate(X_test, y_test)

    print(f"\nBest Metric: {best_metric:.2f} with {best_params}")
    plot_results(train_losses, val_losses, train_metrics, val_metrics, args.dataset, task_type = args.task_type)

if __name__ == "__main__":
    main()