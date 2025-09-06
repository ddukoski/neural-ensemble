from base import NeuralEnsembleBase
from config.types import ClassActivationName
from typing import override


class NeuralEnsembleClassifier(NeuralEnsembleBase):

    def __init__(self, activation: ClassActivationName, **kwargs):
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
