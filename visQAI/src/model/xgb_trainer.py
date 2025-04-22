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
    Trainer class for fitting and saving XGBoost regressors for multiple targets.

    Attributes:
        model_dir (str): Directory to save preprocessor and booster artifacts.
        feature_columns (list): List of feature column names.
        target_columns (list): List of target column names.
        numeric_features (list): Subset of features that are numeric.
        categorical_features (list): Subset of features that are categorical.
        random_state (int): Random seed for reproducibility.
        preprocessor (ColumnTransformer): Fitted transformer for preprocessing.
        boosters (dict): Dictionary of trained xgb.Booster objects, keyed by target name.
    """

    def __init__(
        self,
        model_dir,
        feature_columns=None,
        target_columns=None,
        numeric_features=None,
        categorical_features=None,
        random_state: int = 42,
    ):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        # Define features
        self.feature_columns = feature_columns or [
            "Protein type", "Protein", "Temperature", "Buffer",
            "Sugar", "Sugar (M)", "Surfactant", "TWEEN"
        ]
        self.target_columns = target_columns or [
            "Viscosity100", "Viscosity1000",
            "Viscosity10000", "Viscosity100000",
            "Viscosity15000000"
        ]
        self.numeric_features = numeric_features or [
            "Protein", "Temperature", "Sugar (M)", "TWEEN"
        ]
        self.categorical_features = categorical_features or [
            "Protein type", "Buffer", "Sugar", "Surfactant"
        ]
        self.random_state = random_state

        # Build preprocessor pipeline
        self.preprocessor = ColumnTransformer([
            ("num", StandardScaler(), self.numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical_features),
        ])

        # Placeholder for trained boosters
        self.boosters = {}

    def _tune_with_optuna(self, X_mat: pd.DataFrame, y_df: pd.DataFrame, n_trials: int, cv: int) -> dict:
        """
        Run an Optuna study to minimize average RMSE across targets.
        """
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
            rmse_scores = []
            kf = KFold(n_splits=cv, shuffle=True,
                       random_state=self.random_state)
            for target in y_df.columns:
                dtrain = xgb.DMatrix(X_mat, label=y_df[target].values)
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
                rmse_scores.append(cv_res["test-rmse-mean"].iloc[-1])
            return float(pd.Series(rmse_scores).mean())

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        return study.best_params

    def fit(
        self,
        df: pd.DataFrame,
        tune: bool = False,
        n_trials: int = 20,
        cv: int = 5,
        xgb_params: dict = None,
        num_boost_round: int = 100,
    ):
        """
        Fit the preprocessor and train boosters on the provided DataFrame.

        Args:
            df (pd.DataFrame): Input data containing features and targets.
            tune (bool): Whether to run hyperparameter tuning.
            n_trials (int): Number of Optuna trials if tuning.
            cv (int): Number of CV folds for tuning.
            xgb_params (dict): Parameters for training if not tuning.
            num_boost_round (int): Rounds of boosting when not tuning.
        """
        # Split features and targets
        X = df[self.feature_columns]
        y = df[self.target_columns]

        # Fit and transform features
        self.preprocessor.fit(X)
        X_mat = self.preprocessor.transform(X)

        # Determine params
        if tune:
            best_params = self._tune_with_optuna(X_mat, y, n_trials, cv)
        else:
            best_params = xgb_params or {
                "objective": "reg:squarederror", "seed": self.random_state}

        # Train boosters per target
        for target in self.target_columns:
            dtrain = xgb.DMatrix(X_mat, label=y[target].values)
            params = best_params.copy()
            params.update({"objective": "reg:squarederror",
                          "seed": self.random_state})
            rounds = params.get("n_estimators", num_boost_round)
            booster = xgb.train(
                params,
                dtrain,
                num_boost_round=rounds
            )
            self.boosters[target] = booster

        # Save artifacts
        joblib.dump(self.preprocessor, os.path.join(
            self.model_dir, "preprocessor.pkl"))
        joblib.dump(self.boosters, os.path.join(
            self.model_dir, "boosters.pkl"))

    def __repr__(self):
        return (
            f"<XGBTrainer(model_dir={self.model_dir}, "
            f"n_targets={len(self.target_columns)})>"
        )


if __name__ == "__main__":

    DATA_PATH = os.path.join('content', 'formulation_data_04222025_2.csv')
    SAVE_PATH = os.path.join('visQAI', 'objects', 'nn_regressor')
    df = pd.read_csv(DATA_PATH)
    feature_cols = ["Protein type", "Protein", "Temperature", "Buffer",
                    "Sugar", "Sugar (M)", "Surfactant", "TWEEN"]
    target_cols = ["Viscosity100", "Viscosity1000",
                   "Viscosity10000", "Viscosity100000", "Viscosity15000000"]

    # Initialize trainer and fit
    trainer = XGBTrainer(
        model_dir=args.model_dir
    )
    trainer.fit(
        df,
        tune=args.tune,
        n_trials=args.n_trials,
        cv=args.cv,
        xgb_params=xgb_params,
        num_boost_round=args.num_boost_round
    )

    print(f"Training complete. Artifacts saved to {args.model_dir}")
