import nltk as nltk
import numpy as np
import pandas as pd

# Initialize architecture
nn_architecture = [
    {"no_nodes": 2, "activation": "relu"},  # layer 1 (input layer) - 2 nodes
    {"no_nodes": 4, "activation": "relu"},  # layer 2 - 4 nodes
    {"no_nodes": 6, "activation": "relu"},  # layer 3 - 6 nodes
    {"no_nodes": 6, "activation": "relu"},  # layer 4 - 6 nodes
    {"no_nodes": 4, "activation": "sigmoid"},  # layer 5 - 4 nodes
]


# Activation functions (For entire array)
def sigmoid(Z):
    return 1/(1+np.exp(-Z))


def relu(Z):
    return np.maximum(0,Z)


def sigmoid_deriv(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)


def relu_deriv(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0;
    return dZ;

