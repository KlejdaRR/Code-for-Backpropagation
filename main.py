import numpy as np
from NeuralNetwork import NeuralNetwork
from modules.DenseLayer import DenseLayer
from utils.DatasetLoader import DatasetLoader
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

    train_file = "data/mnist_train.csv"
    test_file = "data/mnist_test.csv"

    dataset_loader = DatasetLoader(dataset_type="mnist")
    X_train, y_train, X_val, y_val, X_test, y_test = dataset_loader.load_data(train_file, test_file)

    nn = NeuralNetwork([
        DenseLayer(784, 128, activation="relu", initialization="he"),
        DenseLayer(128, 64, activation="relu", initialization="he"),
        DenseLayer(64, 10, activation="softmax", initialization="xavier")
    ])

    train_losses, val_losses = nn.train(X_train, y_train, X_val, y_val, epochs=50, learning_rate=0.01, batch_size=64)

    plot_results(train_losses, val_losses)
    nn.evaluate(X_test, y_test)
