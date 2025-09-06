from base import NeuralEnsembleBase
from config.types import RegActivationName
from typing import override


class NeuralEnsembleRegressor(NeuralEnsembleBase):
    def __init__(self, activation: RegActivationName, **kwargs):
        super().__init__(**kwargs)
        self.activation_use = activation

    @override
    def compile(self):
        raise NotImplementedError()

    @override
    def train(self):
        raise NotImplementedError()

    @override
    def predict(self):
        raise NotImplementedError()
