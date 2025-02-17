# Neural Network Implementation

## Overview
This project implements a multi-layer neural network using Object-Oriented Programming (OOP) principles in Python. It supports modular layers, training, and evaluation on synthetic datasets and can be extended to real-world datasets like MNIST.

## Project Structure
```
Code-for-Backpropagation/
│── data/
│   │── mnist_text.csv
│   │── mnist_train.csv
│── cifar-10-python/
│── modules/
│   │── Activation.py       # Activation functions (ReLU, Sigmoid, Softmax)
│   │── BaseLayer.py        # Base class for layers
│   │── DenseLayer.py       # Fully connected layer implementation
│── utils/
│   │── DatasetLoader.py    # Handles dataset loading (MNIST, CIFAR-10)
│   │── StopCriterion.py    # Early stopping and loss plateau detection
│── NeuralNetwork.py        # Neural network model
│── main.py                 # Training and evaluation script
│── README.txt              # Project documentation
```

## Features
- Fully connected layers with configurable activation functions
- Activation functions: ReLU, Sigmoid, Softmax
- Dataset Loader for MNIST and CIFAR-10 datasets
- Early stopping and loss plateau detection for optimized training
- Forward and Backward Propagation with gradient descent and L2 regularization
- Training and Evaluation with visualization of loss and accuracy

## Usage
Run the training script with:
```
python main.py --dataset mnist
python main.py --dataset cifar10
```

## Dependencies
- Python 3.10.11
- NumPy 2.2.2
- Pandas 2.2.3
- Matplotlib  3.10.0

## Authors
Klejda Rrapaj: k.rrapaj@student.unisi.it
Sildi Ricky: s.ricku@student.unisi.it

