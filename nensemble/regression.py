from nensemble.core import NeuralEnsembleBase
from config.types import RegActivationName
from config.logger import make_logger
from utils.hidden import DenseGroup
from typing import override
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import backend as K
from keras.callbacks import EarlyStopping

from sklearn.utils import resample

import numpy as np

logger = make_logger()


class NeuralEnsembleRegressor(NeuralEnsembleBase):
    def __init__(self, activation: RegActivationName, **kwargs):
        super().__init__(**kwargs)
        if self.dataset.is_classification():
            raise ValueError(
                f"The dataset passed has a classification target/response variable, extected a dataset with a regression target/response variable."
            )
        self.activation_use = activation
        self._build()

    @override
    def _build(self):
        for i, net in enumerate(self.ensemble):
            net.add(Input((self.dataset.model_input_dim(),)))

            base_sizes = [16, 32, 64, 64, 32, 16]
            for layer in DenseGroup(
                outline=base_sizes,
                activation=self.activation_use,
                dropout_range=(0.001 * i, 0.005 * i),
            ):
                net.add(layer)

            net.add(Dense(1, activation="linear"))
            logger.info(f"Sucessfully populated {net.name} with layers")

    @override
    def compile(self):
        [
            net.compile(
                optimizer=self.optimizer_use(self.lr),
                metrics=["mse", "r2_score"],
                loss="mse",
            )
            for net in self.ensemble
        ]

    @override
    def train(self):
        logger.info("Splitting dataset into train and test sets...")
        X_train, X_test, y_train, y_test = self.dataset.train_test_split()

        logger.info("Starting training of ensemble models with bootstrap sampling...")

        for i, net in enumerate(self.ensemble):
            logger.info(f"Bootstrapping training data for model {net.name}...")

            X_boot, y_boot = resample(
                X_train,
                y_train,
                replace=True,
                n_samples=len(X_train),
            )

            logger.info(f"Training model {net.name}.")
            net.fit(
                X_boot,
                y_boot,
                epochs=100,
                batch_size=64,
                verbose=True,
                validation_data=(X_test, y_test),
                callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
            )
            logger.info(f"Model {net.name} training complete.")
            K.clear_session()

    @override
    def predict(self):
        raise NotImplementedError()
