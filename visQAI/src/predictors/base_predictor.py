# base_predictor.py
from abc import ABC, abstractmethod
import pandas as pd


class BasePredictor(ABC):
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        pass

    def update(self, X: pd.DataFrame, y: pd.DataFrame):
        raise NotImplementedError(
            "This predictor does not support fine-tuning.")
