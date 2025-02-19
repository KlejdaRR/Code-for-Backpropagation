# Neural Network Implementation

## Overview
This repository contains a flexible and modular implementation of a neural network in Python.
The network supports various activation functions, layer types, and training techniques such as dropout and batch normalization.
It is designed to handle both classification and regression tasks and can be used with different datasets, including MNIST, CIFAR-10, Wine Quality, and custom datasets.

## Project Structure
```
Code-for-Backpropagation/
│── data/                     # Dataset files
│   │── mnist_text.csv        # MNIST test dataset
│   │── mnist_train.csv       # MNIST training dataset
│── cifar-10-python/          # CIFAR-10 dataset files
│── wine+quality/             # Wine Quality dataset files
│── modules/                  # Core modules for the neural network
│   │── Activation.py         # Activation functions (ReLU, Sigmoid, Softmax, Linear)
│   │── BaseLayer.py          # Base class for layers
│   │── DenseLayer.py         # Fully connected layer implementation
│   │── DropoutLayer.py       # Dropout layer for regularization
│── utils/                    # Utility modules
│   │── DatasetLoader.py      # Handles dataset loading (MNIST, CIFAR-10, Wine Quality, custom)
│   │── StopCriterion.py      # Early stopping and loss plateau detection
│── NeuralNetwork.py          # Neural network model implementation
│── main.py                   # Training and evaluation script
│── README.md                 # Project documentation
```

## Features
- Modular Layers: Fully connected layers with configurable activation functions.
- Activation Functions: ReLU, Sigmoid, Softmax, and Linear.
- Dataset Loader: Supports MNIST, CIFAR-10, Wine Quality, and custom datasets.
- Training Techniques:
Batch normalization.
Dropout for regularization.
L2 regularization to prevent overfitting.
- Optimization:
Gradient descent with configurable learning rate.
Early stopping and loss plateau detection for efficient training.
- Task Types: Supports both classification and regression tasks.
- Visualization: Plots training and validation loss/accuracy over epochs.

## Usage
Run the training script with:
```
python main.py --dataset <dataset_name> --task_type <task_type> --custom_path <path_to_custom_dataset>

examples:
python main.py --dataset mnist --task_type classification
python main.py --dataset custom --custom_path custom_dataset.npz
python main.py --dataset wine_quality --task_type regression
python main.py --dataset cifar10 --task_type classification
```

## Dependencies
- Python 3.10.11
- NumPy 2.2.2
- Pandas 2.2.3
- Matplotlib  3.10.0

## Authors
Klejda Rrapaj: k.rrapaj@student.unisi.it
Sildi Ricky: s.ricku@student.unisi.it

