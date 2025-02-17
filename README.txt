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
│   │── Activation.py
│   │── BaseLayer.py
│   │── DenseLayer.py       # Fully connected layer implementation
│── utils/
│   │── DatasetLoader.py
│   │── StopCriterion.py
│── NeuralNetwork.py        # Neural network model
│── main.py                  # Training and evaluation script
│── README.txt                # Project documentation
```

## Features
- Modular implementation using **BaseLayer** and **DenseLayer**
- Forward and backward propagation with **sigmoid activation**
- Mean Squared Error (MSE) as loss function
- Training with **early stopping** based on validation loss
- Synthetic dataset for binary classification
- Loss visualization using **Matplotlib**

## Usage
Run the training script with:
```
python main.py --dataset mnist
python main.py --dataset cifar10
```

## Dependencies
- Python 3.9
- NumPy
- Matplotlib

## Authors
Klejda Rrapaj: k.rrapaj@student.unisi.it
Sildi Ricky: s.ricku@student.unisi.it

