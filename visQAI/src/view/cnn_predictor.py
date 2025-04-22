import os
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf


class CNNPredictor:
    """
    Standalone predictor for a trained CNN viscosity model.

    Loads:
      - a SavedModel CNN in <model_dir>/cnn_model/
      - a scikit-learn ColumnTransformer pickle at <model_dir>/preprocessor.pkl

    Usage:
        pred = ViscosityPredictorCNN("path/to/model_dir")
        results = pred.predict(df_features)
    """

    def __init__(self, model_dir: str):
        # validate paths
        self.model_dir = model_dir
        model_path = os.path.join(model_dir, "cnn_model")
        prep_path = os.path.join(model_dir, "preprocessor.pkl")

        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"SavedModel directory not found at {model_path}")
        if not os.path.isfile(prep_path):
            raise FileNotFoundError(
                f"Preprocessor pickle not found at {prep_path}")

        # load Keras model and preprocessor
        self.model = tf.keras.models.load_model(model_path)
        self.preprocessor = joblib.load(prep_path)

        # extract expected feature names and output dims
        try:
            self.input_dim = self.model.input_shape[1]
            self.output_dim = self.model.output_shape[-1]
        except Exception:
            self.input_dim = None
            self.output_dim = None

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict viscosities for new formulations.

        Args:
            X: pd.DataFrame of shape (n_samples, n_features) with the same
               columns used during training (numeric + categorical).

        Returns:
            pd.DataFrame of shape (n_samples, n_targets) containing the predicted
            viscosities at each shear rate. Columns will be named according to
            training target order.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "X must be a pandas DataFrame with feature columns")

        # 1) apply preprocessing
        X_proc = self.preprocessor.transform(X)

        # 2) predict
        preds = self.model.predict(X_proc)

        # 3) wrap in DataFrame
        # if original X had target names stored in metadata, reuse; otherwise default idx
        try:
            target_cols = getattr(X, "_target_cols", None)
        except Exception:
            target_cols = None

        if target_cols is None or len(target_cols) != preds.shape[1]:
            # fallback generic names
            target_cols = [f"viscosity_{i}" for i in range(preds.shape[1])]

        return pd.DataFrame(preds, columns=target_cols, index=X.index)
