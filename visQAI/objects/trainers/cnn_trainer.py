import os
import pandas as pd
import numpy as np
import joblib
import optuna

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold
import tensorflow as tf
from keras import layers, models, optimizers, losses, metrics, callbacks
import pandas as pd
from visQ_data_processor import VisQDataProcessor


import pandas as pd


class ViscosityTrainerCNN:
    def __init__(self,
                 features=None,
                 cv_splits=3,
                 random_state=42):
        self.features = features
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.preprocessor = ColumnTransformer([
            ("num", StandardScaler(), self.features),
        ])
        self.input_dim = None
        self.output_dim = None
        self.best_params = None
        self.model = None

    def _build_model(self, params):
        inp = layers.Input(shape=(self.input_dim,), name="features")
        x = layers.Reshape((self.input_dim, 1))(inp)
        x = layers.Conv1D(params["filters"],
                          params["kernel_size"],
                          activation="relu",
                          padding="same")(x)
        x = layers.Conv1D(params["filters"],
                          params["kernel_size"],
                          activation="relu",
                          padding="same")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(params["dense_units"], activation="relu")(x)
        out = layers.Dense(self.output_dim,
                           activation="linear",
                           name="viscosities")(x)

        model = models.Model(inputs=inp, outputs=out, name="viscosity_cnn")
        model.compile(
            optimizer=optimizers.Adam(learning_rate=params["learning_rate"]),
            loss=losses.MeanSquaredError(),
            metrics=[metrics.RootMeanSquaredError(name="rmse")]
        )
        return model

    def tune(self, X, y, n_trials=20, epochs=20, batch_size=32):

        X_proc = self.preprocessor.fit_transform(X)
        y_arr = y.values if hasattr(y, "values") else np.asarray(y)
        self.input_dim, self.output_dim = X_proc.shape[1], y_arr.shape[1]

        def objective(trial):
            params = {
                "filters": trial.suggest_categorical("filters", [16, 32, 64]),
                "kernel_size": trial.suggest_int("kernel_size", 1, 3),
                "dense_units": trial.suggest_categorical("dense_units", [32, 64, 128]),
                "learning_rate": trial.suggest_loguniform("learning_rate", 1e-4, 1e-2),
            }
            cv = KFold(n_splits=self.cv_splits,
                       shuffle=True,
                       random_state=self.random_state)
            rmses = []
            for train_idx, val_idx in cv.split(X_proc):
                X_tr, X_val = X_proc[train_idx], X_proc[val_idx]
                y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

                m = self._build_model(params)
                m.fit(
                    X_tr, y_tr,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_val, y_val),
                    callbacks=[callbacks.EarlyStopping(patience=3,
                                                       restore_best_weights=True)],
                    verbose=0
                )
                # evaluate returns [loss, rmse]
                rmses.append(m.evaluate(X_val, y_val, verbose=0)[1])
            return float(np.mean(rmses))

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        study.optimize(objective, n_trials=n_trials)
        self.best_params = study.best_params
        return study

    def cross_validate(self, X, y,
                       epochs=50,
                       batch_size=32):
        """
        Runs K-Fold CV training with self.best_params and returns per-fold RMSE.
        Then retrains a final model on all data.
        """
        if self.best_params is None:
            raise RuntimeError("Call tune() before cross_validate().")

        # preprocess
        X_proc = self.preprocessor.fit_transform(X)
        y_arr = y.values if hasattr(y, "values") else np.asarray(y)
        self.input_dim, self.output_dim = X_proc.shape[1], y_arr.shape[1]

        cv = KFold(n_splits=self.cv_splits,
                   shuffle=True,
                   random_state=self.random_state)
        fold_rmses = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X_proc), 1):
            X_tr, X_val = X_proc[train_idx], X_proc[val_idx]
            y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

            m = self._build_model(self.best_params)
            m.fit(
                X_tr, y_tr,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=[callbacks.EarlyStopping(patience=5,
                                                   restore_best_weights=True)],
                verbose=1
            )
            rmse = m.evaluate(X_val, y_val, verbose=0)[1]
            print(f"Fold {fold} RMSE: {rmse:.4f}")
            fold_rmses.append(rmse)

        mean_rmse = np.mean(fold_rmses)
        std_rmse = np.std(fold_rmses)
        print(f"→ CV RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")

        # retrain on full data
        print("Retraining on full dataset...")
        self.model = self._build_model(self.best_params)
        self.model.fit(
            X_proc, y_arr,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[callbacks.EarlyStopping(patience=5,
                                               restore_best_weights=True)],
            verbose=1
        )
        return fold_rmses

    def predict(self, X_new):
        if self.model is None:
            raise RuntimeError(
                "Model not trained. Call cross_validate() or fit() first.")
        Xp = self.preprocessor.transform(X_new)
        return self.model.predict(Xp)

    def save(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        self.model.save(os.path.join(model_dir, "cnn_model.keras"))
        joblib.dump(self.preprocessor, os.path.join(
            model_dir, "preprocessor.pkl"))


if __name__ == "__main__":
    DATA_PATH = 'content/forumlation_data_05052025.csv'
    SAVE_PATH = 'visQAI/objects/cnn_regressor'

    df = pd.read_csv(DATA_PATH)
    target_cols = [
        "Viscosity100",
        "Viscosity1000",
        "Viscosity10000",
        "Viscosity100000",
        "Viscosity15000000"
    ]

    X, y = VisQDataProcessor.process(DATA_PATH)
    X.drop(columns=['Temperature_norm'], inplace=True)
    # iterate through columns
    for col in X.columns:
        # find the row indices where this column is NaN
        null_rows = X.index[X[col].isna()].tolist()
        for row in null_rows:
            print(f"Row {row}, Column '{col}', Value {X.loc[row, col]}")
    features = X.columns
    trainer = ViscosityTrainerCNN(features=features, cv_splits=4)
    print("Tuning hyperparameters…")
    study = trainer.tune(X, y, n_trials=30, epochs=15, batch_size=16)
    print("Best params:", study.best_params)

    print("Cross‑validating final model…")
    rmses = trainer.cross_validate(X, y, epochs=50, batch_size=16)

    print(f"Saving final model to {SAVE_PATH}")
    trainer.save(SAVE_PATH)
    print("Done.")
