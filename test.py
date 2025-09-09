from nensemble.core import NeuralEnsembleBase
from nensemble.regression import NeuralEnsembleRegressor
from utils.dataset import DatasetUtil

import pandas as pd

df: pd.DataFrame = pd.read_csv("heart.csv")

dataset = DatasetUtil(df, target_col="chol")

print(dataset.target_cols)

regressor: NeuralEnsembleBase = NeuralEnsembleRegressor(
    dataset=dataset, activation="relu"
)
regressor.compile()
regressor.train()
