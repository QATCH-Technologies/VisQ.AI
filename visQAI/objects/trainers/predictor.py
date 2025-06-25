import os
from typing import Union, List, Dict, Tuple

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from custom_layers import ReverseCumsum
import pickle


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
        self.model_path = model_path
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
        # 1) coerce to DataFrame
        # if isinstance(data, (dict, list)):
        #     df_new = pd.DataFrame(data)
        if isinstance(data, pd.DataFrame):
            df_new = data.copy()
        else:
            raise TypeError(
                "`data` must be a DataFrame, dict of lists, or list of dicts."
            )

        # 2) preprocess
        X_proc = self.preprocessor.transform(df_new)

        # — if transform() returns a tuple, grab just the first element —
        if isinstance(X_proc, tuple):
            X_proc = X_proc[0]

        # — if it’s still a DataFrame, convert to ndarray —
        if hasattr(X_proc, "values"):
            X_proc = X_proc.values

        # 3) plain predictions
        preds = self.model.predict(X_proc, verbose=0)

        if not return_confidence:
            return preds

        # 4) compute MC-Dropout only if model has Dropout layers
        if any(isinstance(layer, tf.keras.layers.Dropout) for layer in self.model.layers):
            mc_preds = np.stack([
                # keep dropout active at inference time
                self.model(X_proc, training=True).numpy()
                for _ in range(self.mc_samples)
            ], axis=0)
            mean = mc_preds.mean(axis=0)
            std = mc_preds.std(axis=0)
            print("[Status] Computing Error")
        else:
            # no stochastic layers → zero uncertainty
            mean = preds
            std = np.zeros_like(preds)
            print("[Warn] Skipping Error")

        # 5) return the standard deviation as your “confidence” metric
        #    (you can rename conf→std if you prefer)
        return mean, std

    def update(
        self,
        new_data: pd.DataFrame,
        new_targets: np.ndarray,
        epochs: int = 1,
        batch_size: int = 32,
        save: bool = True,
    ):
        """
        Incrementally update both the preprocessor and the model
        on just the new samples.
        """
        # if not hasattr(self.preprocessor, "partial_fit"):
        #     raise RuntimeError(
        #         "Preprocessor does not support partial_fit; "
        #         "you need an incremental transformer."
        #     )
        # self.preprocessor.partial_fit(new_data)
        X_new = self.preprocessor.transform(new_data)
        if isinstance(X_new, tuple):
            X_new = X_new[0]
        if hasattr(X_new, "values"):
            X_new = X_new.values
        self.model.fit(
            X_new,
            new_targets,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
        )

        if save:
            self.model.save(self.model_path)
            # joblib.dump(self.preprocessor, self.preprocessor_path)


class EnsembleViscosityPredictor:
    def __init__(
        self,
        base_dir: str,
        mc_samples: int = 50,
        model_filename: str = "model.keras",
        preprocessor_filename: str = "preprocessor.pkl",
    ):
        """
        base_dir/
            member_1/
                model.h5
                preprocessor.pkl
            member_2/
                ...
        """
        self.members: List[ViscosityPredictor] = []
        for name in sorted(os.listdir(base_dir)):
            member_dir = os.path.join(base_dir, name)
            if not (os.path.isdir(member_dir) and name.startswith("member_")):
                continue

            mdl = os.path.join(member_dir, model_filename)
            pre = os.path.join(member_dir, preprocessor_filename)

            if not os.path.isfile(mdl):
                raise FileNotFoundError(f"Missing model file at {mdl}")
            if not os.path.isfile(pre):
                raise FileNotFoundError(f"Missing preprocessor at {pre}")

            self.members.append(
                ViscosityPredictor(model_path=mdl,
                                   preprocessor_path=pre,
                                   mc_samples=mc_samples)
            )

        if not self.members:
            raise ValueError(f"No ensemble members found in {base_dir!r}")

    def predict(
        self,
        data: Union[pd.DataFrame, Dict[str, List], List[Dict[str, float]]],
        return_confidence: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        # 1) Collect member outputs
        if not return_confidence:
            # simple ensemble mean
            preds = [m.predict(data, return_confidence=False)
                     for m in self.members]
            # shape: (n_members, n_samples, n_targets)
            return np.stack(preds, axis=0).mean(axis=0)

        # gather each member’s mean & std
        means: List[np.ndarray] = []
        stds:  List[np.ndarray] = []
        for m in self.members:
            mu, sigma = m.predict(data, return_confidence=True)
            means.append(mu)
            stds.append(sigma)

        means = np.stack(means, axis=0)  # (M, N, T)
        stds = np.stack(stds,  axis=0)  # (M, N, T)

        # inter-model variance
        var_between = np.var(means, axis=0, ddof=0)
        # average intra-model variance
        var_within = np.mean(stds**2, axis=0)
        total_var = var_between + var_within

        ensemble_mean = np.mean(means, axis=0)
        ensemble_std = np.sqrt(total_var)

        return ensemble_mean, ensemble_std

    def update(
        self,
        new_data: pd.DataFrame,
        new_targets: np.ndarray,
        epochs: int = 1,
        batch_size: int = 32,
        save: bool = True,
    ):
        """
        Update each member of the ensemble on the new samples.
        """
        for member in self.members:
            member.update(
                new_data,
                new_targets,
                epochs=epochs,
                batch_size=batch_size,
                save=save
            )
