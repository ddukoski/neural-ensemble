from tensorflow.keras.optimizers import (
    SGD,
    RMSprop,
    Adam,
    Adamax,
    Nadam,
    Adagrad,
    Adadelta,
    Optimizer,
)

from typing import Dict, Type

available_optimizers: Dict[str, Type[Optimizer]] = {
    "sgd": SGD,
    "rmsprop": RMSprop,
    "adam": Adam,
    "adamax": Adamax,
    "nadam": Nadam,
    "adagrad": Adagrad,
    "adadelta": Adadelta,
}
