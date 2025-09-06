from abc import ABC, abstractmethod
from config.constants import available_optimizers
from config.types import OptimizerName

import pandas as pd

class NeuralEnsembleBase(ABC):

    def __init__(
        self,
        dataset: pd.DataFrame,
        n_predictors: int = 10,
        optimizer: OptimizerName = "adam",
        dropout: float = 0.0,
        lr: float = 0.001,
    ):
        NeuralEnsembleBase.__validate_ensemble_creation(
            dataset=dataset, n_predictors=n_predictors, dropout=dropout, lr=lr
        )
        self.dataset = dataset
        self.n_predictors = n_predictors
        self.dropout = dropout
        self.lr = lr
        self.optimizer_use = available_optimizers[optimizer.lower()](learning_rate=lr)

    @staticmethod
    def __validate_ensemble_creation(
        dataset: pd.DataFrame, n_predictors: int, dropout: float, lr: float
    ):
        if dataset.empty:
            raise ValueError(
                f"Cannot work without data, the dataset being passed: {dataset}, is empty."
            )
        if n_predictors < 1:
            raise ValueError(f"n_predictors must be positive, got {n_predictors}.")
        if not (0 < lr <= 1):
            raise ValueError(f"Learning rate must be in the interval (0, 1], got {lr}.")
        if not (0 <= dropout < 1):
            raise ValueError(
                f"Dropout rate must be in the interval [0, 1), got {dropout}."
            )

    # TODO: implement all of these in the regressor and classifier
    @abstractmethod
    def compile(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass
