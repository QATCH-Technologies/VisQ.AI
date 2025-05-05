import os
import joblib
import xgboost as xgb
import optuna
import pandas as pd
import argparse
import json
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold


class XGBTrainer:
    """
    Trainer for XGBoost regressors across multiple viscosity targets,
    with Optuna hyperparameter tuning support.
    """

    def __init__(
        self,
        model_dir: str,
        feature_columns=None,
        target_columns=None,
        numeric_features=None,
        categorical_features=None,
        random_state: int = 42,
    ):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        # default feature/target sets
        self.feature_columns = feature_columns or [
            "Protein type",
            "MW(kDa)",
            "PI_mean", "PI_range",
            "Protein", "Temperature",
            "Buffer", "Sugar",
            "Sugar(M)", "Surfactant",
            "Concentration"
        ]
        self.target_columns = target_columns or [
            "Viscosity100", "Viscosity1000",
            "Viscosity10000", "Viscosity100000",
            "Viscosity15000000"
        ]
        self.numeric_features = numeric_features or [
            "Protein", "Temperature", "Sugar (M)", "TWEEN"]
        self.categorical_features = categorical_features or [
            "Protein type", "Buffer", "Sugar", "Surfactant"]
        self.random_state = random_state

        # preprocessing pipeline
        self.preprocessor = ColumnTransformer([
            ("num", StandardScaler(), self.numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical_features),
        ])

        # storage for tuned params and boosters
        self.best_params = None
        self.boosters = {}

    def tune(self, X: pd.DataFrame, y: pd.DataFrame, n_trials: int = 20, cv: int = 5):
        """
        Run Optuna tuning to find best XGBoost hyperparameters (minimize RMSE).
        Stores best parameters in self.best_params.
        """
        # prepare transformed features
        self.preprocessor.fit(X)
        X_mat = self.preprocessor.transform(X)

        def objective(trial):
            params = {
                "n_estimators":     trial.suggest_int("n_estimators", 50, 500),
                "max_depth":        trial.suggest_int("max_depth", 3, 12),
                "learning_rate":    trial.suggest_loguniform("learning_rate", 1e-3, 0.3),
                "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
                "objective":        "reg:squarederror",
                "seed":             self.random_state,
            }
            kf = KFold(n_splits=cv, shuffle=True,
                       random_state=self.random_state)
            rmses = []
            for target in y.columns:
                dtrain = xgb.DMatrix(X_mat, label=y[target].values)
                cv_res = xgb.cv(
                    params,
                    dtrain,
                    num_boost_round=params["n_estimators"],
                    folds=kf,
                    metrics="rmse",
                    seed=self.random_state,
                    as_pandas=True,
                    verbose_eval=False
                )
                rmses.append(cv_res["test-rmse-mean"].iloc[-1])
            return float(pd.Series(rmses).mean())

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        self.best_params = study.best_params
        return study

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, num_boost_round: int = 100):
        """
        Train XGBoost boosters on full data using self.best_params.
        Must call tune() first.
        """
        if self.best_params is None:
            raise RuntimeError("Call tune() before fit().")
        X_mat = self.preprocessor.transform(X)
        for target in y.columns:
            dtrain = xgb.DMatrix(X_mat, label=y[target].values)
            params = self.best_params.copy()
            params.update({"objective": "reg:squarederror",
                          "seed": self.random_state})
            rounds = params.get("n_estimators", num_boost_round)
            self.boosters[target] = xgb.train(
                params,
                dtrain,
                num_boost_round=rounds
            )
        return self

    def predict(self, X_new: pd.DataFrame) -> pd.DataFrame:
        """
        Predict viscosity targets for new data.
        """
        if not self.boosters:
            raise RuntimeError("Model not trained. Call fit() first.")
        X_mat = self.preprocessor.transform(X_new)
        dmat = xgb.DMatrix(X_mat)
        preds = {t: b.predict(dmat) for t, b in self.boosters.items()}
        return pd.DataFrame(preds, index=X_new.index)

    def save(self):
        """
        Save preprocessor and boosters to self.model_dir.
        """
        joblib.dump(self.preprocessor, os.path.join(
            self.model_dir, "preprocessor.pkl"))
        joblib.dump(self.boosters, os.path.join(
            self.model_dir, "boosters.pkl"))

    def __repr__(self):
        return f"<XGBTrainer(model_dir={self.model_dir}, tuned={self.best_params is not None})>"


if __name__ == "__main__":
    # Default paths
    DATA_PATH = os.path.join('content', 'formulation_data_04222025_2.csv')
    SAVE_PATH = os.path.join('visQAI', 'objects', 'xgb_regressor')

    # Load data
    df = pd.read_csv(DATA_PATH)
    feature_cols = ["Protein type", "Protein", "Temperature", "Buffer",
                    "Sugar", "Sugar (M)", "Surfactant", "TWEEN"]
    target_cols = ["Viscosity100", "Viscosity1000",
                   "Viscosity10000", "Viscosity100000", "Viscosity15000000"]
    X = df[feature_cols]
    y = df[target_cols]

    # Train and save
    trainer = XGBTrainer(model_dir=SAVE_PATH)
    print("Tuning XGBoost hyperparameters...")
    trainer.tune(X, y, n_trials=20, cv=5)
    print("Training on full dataset...")
    trainer.fit(X, y, num_boost_round=trainer.best_params.get(
        'n_estimators', 100))
    print(f"Saving model artifacts to {SAVE_PATH}...")
    trainer.save()
    print("Done.")
