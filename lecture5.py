import numpy as np
import math

class Neuron():
    def __init__(self):
        self.weight = np.random.randn(100)
        self.bias = np.random.randn(1)

    def activate_function(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, inputs):
        cell_body_sum = np.sum(inputs * self.weight) + self.bias
        activate_rate = self.activate_function(cell_body_sum)
        return activate_rate
