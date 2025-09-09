from sklearn.model_selection import train_test_split
from enum import Enum
from typing import List, Self

import pandas as pd


class TaskType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class Dataset:
    def __init__(self, df: pd.DataFrame, target_col: str, task_type: TaskType):
        self.df = df
        self.predictor_cols = df.drop(columns=[target_col])
        self.target_cols = df[target_col]
        self.task_type = task_type

    @classmethod
    def for_classification(cls, df: pd.DataFrame, target_col: str) -> Self:
        return cls(df, target_col, TaskType.CLASSIFICATION)

    @classmethod
    def for_regression(cls, df: pd.DataFrame, target_col: str) -> Self:
        return cls(df, target_col, TaskType.REGRESSION)

    def is_classification(self) -> bool:
        return self.task_type == TaskType.CLASSIFICATION

    def is_regression(self) -> bool:
        return self.task_type == TaskType.REGRESSION

    def model_input_dim(self) -> int:
        return self.predictor_cols.shape[1]

    def unique_num_vals_response(self) -> int:
        return len(self.target_cols.unique())

    def train_test_split(self, train_size: float = 0.8) -> List:
        return train_test_split(
            self.predictor_cols,
            self.target_cols,
            train_size=train_size,
            shuffle=True,
        )

    def num_classes(self) -> int:
        if not self.is_classification():
            raise ValueError("num_classes() is only valid for classification datasets")
        return self.target_cols.nunique()
