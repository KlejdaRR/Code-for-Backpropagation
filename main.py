import numpy as np
from NeuralNetwork import NeuralNetwork
from modules.DenseLayer import DenseLayer
import matplotlib.pyplot as plt


def plot_results(train_losses, val_losses):
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def load_mnist_data(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    X = data[:, 1:] / 255.0
    y = data[:, 0].astype(int)

    y_one_hot = np.zeros((10, y.shape[0]))
    for i, label in enumerate(y):
        y_one_hot[label, i] = 1

    return X.T, y_one_hot


if __name__ == "__main__":
    np.random.seed(0)

    X_train = np.random.rand(10, 500)
    y_train = (np.sum(X_train, axis=0) > 5).astype(int).reshape(1, -1)

    X_val = np.random.rand(10, 100)
    y_val = (np.sum(X_val, axis=0) > 5).astype(int).reshape(1, -1)

    X_test = np.random.rand(10, 100)
    y_test = (np.sum(X_test, axis=0) > 5).astype(int).reshape(1, -1)

    nn = NeuralNetwork([
        DenseLayer(10, 5),
        DenseLayer(5, 1)
    ])

    train_losses, val_losses = nn.train(X_train, y_train, X_val, y_val, epochs=1000, learning_rate=0.1)

    nn.evaluate(X_test, y_test)

    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()