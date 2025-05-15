import os
from typing import Union, List, Dict, Tuple

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from custom_layers import ReverseCumsum


class ViscosityPredictor:
    def __init__(self,
                 model_path: str,
                 preprocessor_path: str,
                 mc_samples: int = 50):
        if not os.path.isfile(preprocessor_path):
            raise FileNotFoundError(
                f"Could not find preprocessor at {preprocessor_path}")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Could not find model at {model_path}")

        self.preprocessor = joblib.load(preprocessor_path)

        # Tell Keras how to resolve your custom layer

        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                # for safety, map both just-in-case…
                "ReverseCumsum": ReverseCumsum,
                "custom_layers>ReverseCumsum": ReverseCumsum,
            }
        )

        self.mc_samples = mc_samples

    def predict(
        self,
        data: Union[pd.DataFrame, Dict[str, List], List[Dict[str, float]]],
        return_confidence: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Run inference on new samples, optionally returning a per‐prediction confidence score ∈ [0,1].

        Args:
            data: 
              - pd.DataFrame with the expected feature columns, or
              - dict mapping column→list of values, or
              - list of feature‐dicts.
            return_confidence: If True, returns a tuple (preds, confs).

        Returns:
            preds: np.ndarray of shape (n_samples, n_targets)  
            confs (optional): np.ndarray of same shape, values in [0,1],
                where higher means “more certain.”
        """
        # 1) coerce to DataFrame
        if isinstance(data, (dict, list)):
            df_new = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df_new = data.copy()
        else:
            raise TypeError(
                "`data` must be a DataFrame, dict of lists, or list of dicts.")

        # 2) preprocess
        X_proc = self.preprocessor.transform(df_new)

        # 3) plain predictions
        preds = self.model.predict(X_proc, verbose=0)

        if not return_confidence:
            return preds

        # 4) compute MC‐Dropout only if model has Dropout layers
        if any(isinstance(layer, tf.keras.layers.Dropout) for layer in self.model.layers):
            # stack MC samples: shape (mc_samples, n, targets)
            mc_preds = np.stack([
                self.model(X_proc, training=True).numpy()
                for _ in range(self.mc_samples)
            ], axis=0)
            # mean and standard deviation across MC passes
            mean = mc_preds.mean(axis=0)
            std = mc_preds.std(axis=0)
        else:
            # no stochastic layers: treat as deterministic
            mean, std = preds, np.zeros_like(preds)

        # 5) map std → confidence in [0,1]
        #    Here we use: conf = 1 / (1 + std), which compresses larger uncertainty
        conf = 1.0 / (1.0 + std)

        return mean, conf
