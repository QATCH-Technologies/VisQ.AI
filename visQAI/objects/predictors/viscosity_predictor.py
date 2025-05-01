# viscosity_predictor.py
from typing import Type
from .base_predictor import BasePredictor
from .cnn_predictor import CNNPredictor
from .xgb_predictor import XGBPredictor
from .nn_predictor import NNPredictor
import pandas as pd


class ViscosityPredictor:
    _registry = {
        "cnn": CNNPredictor,
        "xgb": XGBPredictor,
        "nn": NNPredictor,
    }

    def __init__(self, predictor_type: str, model_dir: str):
        if predictor_type not in self._registry:
            raise ValueError(f"Unknown predictor type: '{predictor_type}'")

        predictor_cls: Type[BasePredictor] = self._registry[predictor_type]
        self.predictor = predictor_cls(model_dir)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.predictor.predict(X)

    def update(self, X: pd.DataFrame, y: pd.DataFrame):
        return self.predictor.update(X, y)

    @classmethod
    def available_predictors(cls):
        return list(cls._registry.keys())
