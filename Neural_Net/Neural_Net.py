import nltk as nltk
import numpy as np
import pandas as pd

# Initialize architecture
# nn_architecture = [
#     {"no_nodes": 2, "activation": "relu"},  # layer 1 (input layer) - 2 nodes
#     {"no_nodes": 4, "activation": "relu"},  # layer 2 - 4 nodes
#     {"no_nodes": 6, "activation": "relu"},  # layer 3 - 6 nodes
#     {"no_nodes": 6, "activation": "relu"},  # layer 4 - 6 nodes
#     {"no_nodes": 4, "activation": "sigmoid"},  # layer 5 - 4 nodes
# ]


nn_architecture = [
    {"input_dim": 2, "output_dim": 4, "activation": "relu"},
    {"input_dim": 4, "output_dim": 6, "activation": "relu"},
    {"input_dim": 6, "output_dim": 6, "activation": "relu"},
    {"input_dim": 6, "output_dim": 4, "activation": "relu"},
    {"input_dim": 4, "output_dim": 1, "activation": "sigmoid"},
]


def init_layers(nn_architecture, seed=99):
    np.random.seed(seed)
    number_of_layers = len(nn_architecture)
    params_values = {}

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]

        params_values['W' + str(layer_idx)] = np.random.randn(
            layer_output_size, layer_input_size) * 0.1
        params_values['b' + str(layer_idx)] = np.random.randn(
            layer_output_size, 1) * 0.1

    return params_values

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def relu(Z):
    return np.maximum(0,Z)

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0;
    return dZ;

