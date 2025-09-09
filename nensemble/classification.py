from keras import Input
from keras.layers import Dense
from config.constants import available_optimizers
from config.types import OptimizerName, ClassActivationName
from utils.hidden import DenseGroup
from nensemble.core import NeuralEnsembleBase
from abc import override
from config.logger import make_logger

import tensorflow as tf

tf.config.run_functions_eagerly(True)

logger = make_logger()


class NeuralEnsembleClassifier(NeuralEnsembleBase):
    def __init__(self, activation: ClassActivationName, **kwargs):
        super().__init__(**kwargs)

        if self.dataset.is_regression():
            raise ValueError(
                f"The dataset passed has a classification target/response variable, extected a dataset with a regression target/response variable."
            )

        self.activation_use = activation

    @override
    def _build(self):
        n_classes = self.dataset.num_classes()

        for i, net in enumerate(self.ensemble):
            net.add(Input((self.dataset.model_input_dim(),)))

            base_sizes = [16, 32, 64, 64, 32, 16]
            for layer in DenseGroup(
                outline=base_sizes,
                activation=self.activation_use,
                dropout_range=(0.001 * i, 0.005 * i),
            ):
                net.add(layer)

            if n_classes == 2:
                net.add(Dense(1, activation="sigmoid"))
            else:
                net.add(
                    Dense(self.dataset.unique_num_vals_response(), activation="softmax")
                )    

            logger.info(f"Successfully populated {net.name} with layers")

    @override
    def compile(self):
        n_classes = self.dataset.num_classes()

        if n_classes == 2:
            loss = "binary_crossentropy"
        else:
            loss = "sparse_categorical_crossentropy"

        for net in self.ensemble:
            optimizer_instance = available_optimizers[self.optimizer_name](
                learning_rate=self.lr
            )
            net.compile(
                optimizer=optimizer_instance,
                loss=loss,
                metrics=["accuracy", "f1_score", "recall", "precision"],
            )

    @override
    def train(self):
        X_train, X_test, y_train, y_test = self.dataset.train_test_split()

        for net in self.ensemble:
            net.fit(
                X_train,
                y_train,
                epochs=50,
                batch_size=64,
                verbose=True,
                validation_data=(X_test, y_test),
            )

    @override
    def predict(self, X):
        preds = [net.predict(X, verbose=0) for net in self.ensemble]
        return sum(preds) / len(preds)
