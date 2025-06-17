# predictor_wrapper.py

import numpy as np
from typing import Any
import tensorflow as tf


class Predictor:
    """
    Wrapper around a Keras multi-output regression model. Implements:
      - set_model(model)              ⇒ attaches loaded Keras model
      - predict(X)                    ⇒ returns numpy array of shape (n_samples, n_targets)
      - update(X_new, y_new, model)   ⇒ performs transfer learning on the same Keras model
    """

    def __init__(self):
        self.model: tf.keras.Model | None = None

    def set_model(self, model: Any):
        """
        Called by Generic Predictor.__init__ to attach the loaded model.
        """
        self.model = model

    def predict(self, X: Any) -> np.ndarray:
        """
        Given X (DataFrame or NumPy array), return the predicted regression outputs
        as an (n_samples × n_targets) NumPy array.
        """
        if self.model is None:
            raise ValueError("No model set. Call set_model() first.")
        X_arr = np.array(X)
        preds = self.model.predict(X_arr)
        # If the model outputs a single target, `preds` might be shape (n_samples,), so ensure 2D
        preds = np.atleast_2d(preds)
        return preds

    def update(self, X_new: Any, y_new: Any, model: Any):
        """
        Fine-tune the existing model on new data (X_new, y_new).
        We simply re-compile with a smaller learning rate and call model.fit for a few epochs.
        """
        if self.model is None:
            raise ValueError("No model set. Call set_model() first.")

        X_arr = np.array(X_new)
        y_arr = np.array(y_new)

        # Re-compile with small lr (assuming original was lr=1e-3)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss="mse",
            metrics=["mae"]
        )

        # Fine-tune for a few epochs
        self.model.fit(
            X_arr,
            y_arr,
            epochs=5,
            batch_size=16,
            verbose=0
        )
        # No return needed; weights updated in place
