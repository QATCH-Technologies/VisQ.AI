import os
import pandas as pd
import numpy as np
import optuna
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold


class ViscosityRegressorLM:
    """
    Build, tune, train, predict, and save viscosity profiles
    using a multi-output Ridge linear model.
    """

    def __init__(self,
                 numeric_features=None,
                 categorical_features=None,
                 cv_splits=5,
                 random_state=42):
        self.numeric_features = numeric_features or [
            "Protein", "Temperature", "Sugar (M)", "TWEEN"]
        self.categorical_features = categorical_features or [
            "Protein type", "Buffer", "Sugar", "Surfactant"]
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.preprocessor = ColumnTransformer([
            ("num", StandardScaler(), self.numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical_features),
        ])
        self.best_params = None
        self.pipeline = None

    def _build_pipeline(self, params):
        ridge = Ridge(
            alpha=params["alpha"],
            fit_intercept=params.get("fit_intercept", True),
            random_state=self.random_state
        )
        multi = MultiOutputRegressor(ridge)
        return Pipeline([
            ("preprocessor", self.preprocessor),
            ("regressor", multi)
        ])

    def tune(self, X, y, n_trials=20):
        def objective(trial):
            params = {
                "alpha": trial.suggest_float("alpha", 1e-3, 1e2, log=True),
                "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False])
            }
            pipe = self._build_pipeline(params)
            cv = KFold(n_splits=self.cv_splits, shuffle=True,
                       random_state=self.random_state)
            scores = cross_val_score(
                pipe, X, y,
                cv=cv,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1
            )
            return scores.mean()

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        study.optimize(objective, n_trials=n_trials)
        self.best_params = study.best_params
        return study

    def fit(self, X, y):
        if self.best_params is None:
            raise RuntimeError("Call tune() before fit().")
        self.pipeline = self._build_pipeline(self.best_params)
        self.pipeline.fit(X, y)
        return self

    def predict(self, X_new):
        if self.pipeline is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return self.pipeline.predict(X_new)

    def save(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.pipeline, os.path.join(model_dir, "pipeline_lm.pkl"))


if __name__ == "__main__":
    DATA_PATH = os.path.join('content', 'formulation_data_04222025_2.csv')
    SAVE_PATH = os.path.join('visQAI', 'objects', 'linear_regressor')
    df = pd.read_csv(DATA_PATH)
    feature_cols = ["Protein type", "Protein", "Temperature", "Buffer",
                    "Sugar", "Sugar (M)", "Surfactant", "TWEEN"]
    target_cols = ["Viscosity100", "Viscosity1000",
                   "Viscosity10000", "Viscosity100000", "Viscosity15000000"]

    X = df[feature_cols]
    y = df[target_cols]

    reg = ViscosityRegressorLM()
    print("Tuning linear model hyperparameters...")
    reg.tune(X, y, n_trials=20)
    print("Training on full data...")
    reg.fit(X, y)
    print(f"Saving pipeline to {SAVE_PATH}...")
    reg.save(SAVE_PATH)
    print("Done.")
