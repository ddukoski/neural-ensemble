from typing import Literal

RegActivationName   = Literal['linear', 'relu', 'softplus', 'sigmoid', 'tanh']
ClassActivationName = Literal['sigmoid', 'softmax', 'relu', 'tanh']
OptimizerName       = Literal['sgd', 'rmsprop', 'adam', 'adamax', 'nadam', 'adagrad', 'adadelta']