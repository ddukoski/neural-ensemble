from typing import List, Tuple, Iterator
from config.types import RegActivationName
from tensorflow.keras.layers import Dense, Dropout, Layer

import numpy as np


class DenseGroup:
    def __init__(
        self,
        outline: List[int],
        activation: RegActivationName,
        dropout_range: Tuple[float, float] = (0, 0),
    ):
        self.layers = []

        for neurons in outline:
            self.layers.append(Dense(neurons, activation=activation))
            self.layers.append(Dropout(np.random.uniform(*dropout_range)))

    def __iter__(self) -> Iterator[Layer]:
        return iter(self.layers)
