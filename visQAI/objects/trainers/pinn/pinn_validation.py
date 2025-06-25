import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pinn_domain import DataLoader
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Optional, Tuple, Union, List


def _extract_model_id(path: str) -> int:
    """
    Extract integer ID from filenames like 'pinn_<id>.h5'.
    """
    m = __import__('re').search(r'pinn_(\d+)\.h5$', path)
    if m:
        return int(m.group(1))
    else:
        raise ValueError(f"Invalid model filename: {path}")


class Validator:
    def __init__(
        self,
        models: Union[tf.keras.Model, List[tf.keras.Model]],
        preprocessors: Optional[List[object]] = None,
        loader: Optional[DataLoader] = None,
        target_names: Optional[List[str]] = None,
    ):
        """
        Validator that can handle a single model or an ensemble of models.

        Args:
            models: A model or list of models for prediction.
            preprocessors: List of sklearn transformers matching each model.
            loader: DataLoader instance for cleaning & transforming raw DataFrames.
            target_names: Optional list of target column names.
        """
        # ensure list of models
        self.models = models if isinstance(models, list) else [models]
        # align preprocessors
        if preprocessors is None:
            self.preprocessors = [None] * len(self.models)
        else:
            if len(preprocessors) != len(self.models):
                raise ValueError(
                    "Length of preprocessors must match number of models.")
            self.preprocessors = preprocessors
        self.loader = loader
        self.target_names = target_names

    def _prep(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        pre: Optional[object]
    ) -> np.ndarray:
        """
        Preprocess input X using DataLoader + a given transformer.
        """
        if self.loader is not None and pre is not None:
            if isinstance(X, pd.DataFrame):
                self.loader._df = X.copy()
            self.loader._preprocessor = pre
            return self.loader.get_processed_features()
        return X if isinstance(X, np.ndarray) else X.values

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """
        Predict and average across ensemble (or single model).
        """
        preds = []
        for model, pre in zip(self.models, self.preprocessors):
            Xp = self._prep(X, pre)
            preds.append(model.predict(Xp))
        stacked = np.stack(preds, axis=0)
        return stacked.mean(axis=0)

    def predict_with_uncertainty(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        n_iter: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Monte Carlo dropout across models and iterations.
        """
        all_preds = []
        for model, pre in zip(self.models, self.preprocessors):
            Xp = self._prep(X, pre)
            xt = tf.convert_to_tensor(Xp, dtype=tf.float32)
            mc_preds = [model(xt, training=True).numpy()
                        for _ in range(n_iter)]
            all_preds.extend(mc_preds)
        stacked = np.stack(all_preds, axis=0)
        return stacked.mean(axis=0), stacked.std(axis=0)

    def plot_true_vs_pred(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_name: str,
    ):
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        ax.scatter(y_true, y_pred, s=50, alpha=0.7, label="Pred vs Actual")
        vmin, vmax = float(np.min([y_true.min(), y_pred.min()])), float(
            np.max([y_true.max(), y_pred.max()]))
        xs = np.linspace(vmin, vmax, 100)
        ax.fill_between(xs, 0.9*xs, 1.1*xs, alpha=0.2, label="±10% region")
        ax.plot(xs, xs, '--', linewidth=2, label="Ideal")
        ax.text(0.05, 0.95, f"$R^2={r2:.2f}$\nMAE={mae:.2f}", transform=ax.transAxes, va='top', bbox=dict(
            facecolor='white', alpha=0.8))
        ax.set_xlabel("Actual", fontsize=14)
        ax.set_ylabel("Predicted", fontsize=14)
        ax.set_title(target_name.replace('_', ' '), fontsize=16)
        ax.grid(True, linestyle='--', linewidth=0.5)
        ax.legend(loc='lower right')
        plt.tight_layout()
        plt.show()

    def validate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.DataFrame]
    ) -> dict:
        y_true = y.values if hasattr(y, 'values') else y
        y_pred = self.predict(X)
        n_t = y_true.shape[1]
        names = self.target_names or [f"Target_{i}" for i in range(n_t)]
        results = {}
        for idx, name in enumerate(names):
            t, p = y_true[:, idx], y_pred[:, idx]
            results[name] = {"mse": mean_squared_error(
                t, p), "r2": r2_score(t, p)}
            self.plot_true_vs_pred(t, p, name)
        return results


if __name__ == "__main__":
    # load raw data
    csv_path = os.path.join('content', 'train_features.csv')
    loader = DataLoader(csv_path)
    loader.load()

    # discover ensemble
    import glob
    import re
    import joblib
    from tensorflow.keras.models import load_model
    ENSEMBLE_DIR = 'ensemble_pinn'
    model_glob = os.path.join(ENSEMBLE_DIR, 'pinn_*.h5')
    paths = sorted(glob.glob(model_glob), key=_extract_model_id)

    models, preprocessors = [], []
    for p in paths:
        mid = _extract_model_id(p)
        models.append(load_model(p, compile=False))
        tf_path = os.path.join(ENSEMBLE_DIR, f"transformer_{mid}.joblib")
        preprocessors.append(joblib.load(tf_path))

    # split raw
    X_raw, y_df = loader.split(preprocess=False)

    # validate ensemble
    validator = Validator(
        models=models,
        preprocessors=preprocessors,
        loader=loader,
        target_names=loader.TARGET_COLUMNS
    )
    metrics = validator.validate(X_raw, y_df)
    for k, v in metrics.items():
        print(f"{k}: MSE={v['mse']:.3e}, R2={v['r2']:.3f}")
