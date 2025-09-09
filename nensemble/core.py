from abc import ABC, abstractmethod
from typing import List
from config.constants import available_optimizers
from config.types import OptimizerName
from utils.dataset import Dataset
from keras import Sequential
import tensorflow as tf


class NeuralEnsembleBase(ABC):

    def __init__(
        self,
        dataset: Dataset,
        n_predictors: int = 10,
        optimizer: OptimizerName = "adam",
        dropout: float = 0.0,
        lr: float = 0.001,
    ):
        NeuralEnsembleBase.__validate_ensemble_creation(
            dataset=dataset, n_predictors=n_predictors, dropout=dropout, lr=lr
        )
        tf.config.run_functions_eagerly(True)

        self.dataset = dataset
        self.n_predictors = n_predictors
        self.dropout = dropout
        self.lr = lr
        self.optimizer_use = available_optimizers[optimizer.lower()]

        self.ensemble = self.__reserve_ensemble()

    @staticmethod
    def __validate_ensemble_creation(
        dataset: Dataset, n_predictors: int, dropout: float, lr: float
    ):
        if dataset.df.empty:
            raise ValueError(
                f'Cannot create a NeuralEnsemble object type without data, the dataset being passed:\n"{dataset.df}"\nis an empty one.'
            )
        if n_predictors < 1:
            raise ValueError(f"n_predictors must be positive, got {n_predictors}.")
        if not (0 < lr <= 1):
            raise ValueError(f"Learning rate must be in the interval (0, 1], got {lr}.")
        if not (0 <= dropout < 1):
            raise ValueError(
                f"Dropout rate must be in the interval [0, 1), got {dropout}."
            )

    def __reserve_ensemble(self) -> List[Sequential]:
        ensemble = [Sequential() for _ in range(self.n_predictors)]
        for ind, ens in enumerate(ensemble):
            ens.name = f"network_{ind}"
        return ensemble

    def __str__(self):
        return f'{self.__class__} with {self.n_predictors} networks:\n{'\n'.join([str(net) for net in self.ensemble])}'

    # TODO: implement all of these in the regressor and classifier
    @abstractmethod
    def _build(self):
        pass

    @abstractmethod
    def compile(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass
