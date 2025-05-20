# validation.py
import os
import tensorflow as tf
from pinn_domain import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Optional, Tuple, Union


class Validator:
    def __init__(
        self,
        model,
        preprocessor: Optional[object] = None,
        target_names: Optional[list[str]] = None,
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.target_names = target_names

    def _prep(self, X: Union[np.ndarray, "pd.DataFrame"]) -> np.ndarray:
        if self.preprocessor is not None:
            return self.preprocessor.transform(X)
        else:
            return X if isinstance(X, np.ndarray) else X.values

    def predict(self, X: Union[np.ndarray, "pd.DataFrame"]) -> np.ndarray:
        """
        Standard point prediction.
        """
        X_np = self._prep(X)
        return self.model.predict(X_np)

    def predict_with_uncertainty(
        self,
        X: Union[np.ndarray, "pd.DataFrame"],
        n_iter: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_np = self._prep(X)
        x_tensor = tf.convert_to_tensor(X_np, dtype=tf.float32)

        preds = []
        for _ in range(n_iter):
            p = self.model(x_tensor, training=True)
            preds.append(p.numpy())
        preds = np.stack(preds, axis=0)

        mean = preds.mean(axis=0)
        std = preds.std(axis=0)
        return mean, std

    def plot_true_vs_pred(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_name: str,
    ):
        """
        Draws a scatter-plot of true vs. predicted values for one target,
        with identity line, R²/MAE annotation, grid, and legend.
        """
        # compute metrics
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)

        # create figure
        plt.figure(figsize=(10, 6))
        ax = plt.gca()

        # scatter points
        ax.scatter(
            y_true,
            y_pred,
            s=50,
            alpha=0.7,
            label="Prediction vs Actual"
        )

        # identity line
        vmin = min(y_true.min(), y_pred.min())
        vmax = max(y_true.max(), y_pred.max())
        ax.plot(
            [vmin, vmax],
            [vmin, vmax],
            linestyle="--",
            linewidth=2,
            label="Ideal (y = x)"
        )

        # metrics annotation
        textstr = f"$R^2 = {r2:.2f}$\nMAE = {mae:.2f}"
        bbox_props = dict(boxstyle="round,pad=0.5",
                          facecolor="white", alpha=0.8)
        ax.text(
            0.05,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=bbox_props,
        )

        # labels, title, grid, legend
        ax.set_xlabel("Actual Viscosity",   fontsize=14)
        ax.set_ylabel("Predicted Viscosity", fontsize=14)
        ax.set_title(
            f"{self.model.name} — {target_name.replace('_', ' ')}", fontsize=16)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.legend(loc="lower right", fontsize=12)

        plt.tight_layout()
        plt.show()

    def validate(self, X, y):
        """
        Runs predictions on (X, y), computes metrics and plots.

        Args:
            X: DataFrame or NumPy array of inputs.
            y: DataFrame or NumPy array of true targets.

        Returns:
            dict mapping each target name → {'mse':..., 'r2':...}
        """
        # 1) Prepare arrays
        y_true = y if isinstance(y, np.ndarray) else y.values
        y_pred = self.predict(X)

        n_targets = y_true.shape[1]
        names = (
            self.target_names
            if self.target_names and len(self.target_names) == n_targets
            else [f"Target_{i}" for i in range(n_targets)]
        )

        results = {}
        for i, name in enumerate(names):
            true_i = y_true[:, i]
            pred_i = y_pred[:, i]

            mse = mean_squared_error(true_i, pred_i)
            r2 = r2_score(true_i, pred_i)
            results[name] = {"mse": mse, "r2": r2}

            # 2) Plot
            self.plot_true_vs_pred(true_i, pred_i, name)

        return results


if __name__ == "__main__":
    csv_path = os.path.join('content', 'formulation_data_05152025.csv')
    loader = DataLoader(csv_path)
    loader.load()
    loader.build_preprocessor()
    X_df, y_df = loader.split(preprocess=False)  # raw features DataFrame

    model = tf.keras.models.load_model("best_pinn_model.h5", compile=False)
    # re-compile if you plan to evaluate or retrain
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae", "mse"],
    )

    validator = Validator(
        model=model,
        preprocessor=loader.preprocessor,
        target_names=loader.TARGET_COLUMNS,
    )

    metrics = validator.validate(X_df, y_df)
    for name, m in metrics.items():
        print(f"{name}: MSE={m['mse']:.3e}, R2={m['r2']:.3f}")
