import numpy as np
from tensor import Tensor

class MSE():
    def __init__(self):
        self.loss = None

    def __call__(self, predictions, targets):
        assert predictions.shape == targets.shape
        return Tensor.sum((predictions-targets)*(predictions-targets)) / targets.shape[0]


class CrossEntropy():
    def __init__(self):
        self.loss = None

    def __call__(self, predictions, targets):
        assert predictions.shape == targets.shape
        return -Tensor.sum((targets * Tensor.log(predictions))) 

