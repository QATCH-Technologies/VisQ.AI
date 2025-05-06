import os
from typing import Union, List, Dict

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf


class ViscosityPredictor:

    def __init__(self,
                 model_path: str,
                 preprocessor_path: str):
        # sanity checks
        if not os.path.isfile(preprocessor_path):
            raise FileNotFoundError(
                f"Could not find preprocessor at {preprocessor_path}")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Could not find model at {model_path}")

        # load them correctly
        self.preprocessor = joblib.load(preprocessor_path)
        self.model = tf.keras.models.load_model(model_path)

    def predict(self,
                data: Union[pd.DataFrame, Dict[str, List],
                            List[Dict[str, float]]]
                ) -> np.ndarray:
        """
        Run inference on new samples.

        Args:
            data:
              - a DataFrame with exactly the columns your pipeline expects, or
              - a dict of lists(column→values), or
              - a list of feature‐dicts.

        Returns:
            A NumPy array of shape(n_samples, n_targets) with the predicted viscosities.
        """
        # coerce input into a DataFrame
        if isinstance(data, dict):
            df_new = pd.DataFrame(data)
        elif isinstance(data, list):
            df_new = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df_new = data.copy()
        else:
            raise TypeError(
                "`data` must be a DataFrame, dict of lists, or list of dicts.")

        # apply the exact same transforms you used in training
        X_proc = self.preprocessor.transform(df_new)

        # Keras expects a NumPy array
        preds = self.model.predict(X_proc, verbose=0)
        return preds
