import pandas as pd
import os
import tensorflow as tf
import joblib
from .base_predictor import BasePredictor


class CNNPredictor(BasePredictor):
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        model_path = os.path.join(model_dir, "cnn_model.keras")
        prep_path = os.path.join(model_dir, "preprocessor.pkl")

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        if not os.path.isfile(prep_path):
            raise FileNotFoundError(f"Preprocessor not found at {prep_path}")

        self.model = tf.keras.models.load_model(model_path)
        self.preprocessor = joblib.load(prep_path)

        try:
            self.input_dim = self.model.input_shape[1]
            self.output_dim = self.model.output_shape[-1]
        except Exception:
            self.input_dim = None
            self.output_dim = None

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        X_proc = self.preprocessor.transform(X)
        preds = self.model.predict(X_proc)

        target_cols = getattr(X, "_target_cols", None)
        if target_cols is None or len(target_cols) != preds.shape[1]:
            target_cols = [f"viscosity_{i}" for i in range(preds.shape[1])]

        return pd.DataFrame(preds, columns=target_cols, index=X.index)

    def update(self, X, y):
        # You could implement transfer learning logic here
        pass
