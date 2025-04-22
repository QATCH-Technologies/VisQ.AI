import os
import joblib
import xgboost as xgb
import optuna
import pandas as pd
from sklearn.model_selection import KFold


class XGBPredictor:

    def __init__(self, model_dir,
                 feature_columns=None,
                 target_columns=None):
        self.model_dir = model_dir

        # load preprocessor (must be a fitted ColumnTransformer or Pipeline)
        self.preprocessor = joblib.load(
            os.path.join(model_dir, "preprocessor.pkl")
        )

        # load dict of { target_name: xgb.Booster }
        self.boosters = joblib.load(
            os.path.join(model_dir, "boosters.pkl")
        )

        # which columns to pull from incoming data
        self.feature_columns = feature_columns or [
            "Protein type", "Protein", "Temperature", "Buffer",
            "Sugar", "Sugar (M)", "Surfactant", "TWEEN"
        ]
        self.target_columns = target_columns or [
            "Viscosity100", "Viscosity1000",
            "Viscosity10000", "Viscosity100000",
            "Viscosity15000000"
        ]

    def predict(self, df_new: pd.DataFrame) -> pd.DataFrame:
        X = df_new[self.feature_columns]
        X_mat = self.preprocessor.transform(X)
        dmat = xgb.DMatrix(X_mat)
        preds = {
            target: booster.predict(dmat)
            for target, booster in self.boosters.items()
        }
        return pd.DataFrame(preds, index=df_new.index)

    def _tune_with_optuna(self, X_mat, y_df, n_trials: int, cv: int):
        """Run an Optuna study using xgb.cv across targets to minimize avg RMSE."""
        def objective(trial):
            # Suggest hyperparameters
            params = {
                "n_estimators":    trial.suggest_int("n_estimators",    50, 500),
                "max_depth":       trial.suggest_int("max_depth",       3, 12),
                "learning_rate":   trial.suggest_loguniform("learning_rate", 1e-3, 0.3),
                "subsample":       trial.suggest_float("subsample",       0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "gamma":           trial.suggest_float("gamma",           0.0, 5.0),
                "objective":       "reg:squarederror",
                "seed":            42,
            }
            rmse_scores = []
            kf = KFold(n_splits=cv, shuffle=True, random_state=42)
            # For each target, do a CV and record final RMSE
            for target in y_df.columns:
                dtrain = xgb.DMatrix(X_mat, label=y_df[target].values)
                cv_res = xgb.cv(
                    params,
                    dtrain,
                    num_boost_round=params["n_estimators"],
                    folds=kf,
                    metrics="rmse",
                    seed=42,
                    as_pandas=True,
                    verbose_eval=False
                )
                rmse_scores.append(cv_res["test-rmse-mean"].iloc[-1])
            # Objective is the mean RMSE across targets
            return float(pd.Series(rmse_scores).mean())

        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        return study.best_params

    def update(self,
               df_new: pd.DataFrame,
               xgb_params: dict = None,
               num_boost_round: int = 10,
               save: bool = True,
               tune: bool = False,
               n_trials: int = 20,
               cv: int = 5):
        # -- 1) Preprocessor update --
        X_new = df_new[self.feature_columns]
        if hasattr(self.preprocessor, "transformers_"):
            for _, transformer, cols in self.preprocessor.transformers_:
                if hasattr(transformer, "partial_fit"):
                    transformer.partial_fit(X_new[cols])
        else:
            self.preprocessor.fit(X_new)

        # Transform once for both incremental and tuning steps
        X_mat = self.preprocessor.transform(X_new)

        # -- 3) Optional Optuna tuning + full retrain --
        if tune:
            # run CV to find best_params
            best_params = self._tune_with_optuna(X_mat, df_new[self.target_columns],
                                                 n_trials=n_trials, cv=cv)
            # retrain boosters from scratch on this batch
            new_boosters = {}
            for target in self.target_columns:
                dtrain = xgb.DMatrix(X_mat, label=df_new[target].values)
                params = best_params.copy()
                params.update({"objective": "reg:squarederror", "seed": 42})
                new_boosters[target] = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=params["n_estimators"]
                )
            self.boosters = new_boosters
        else:
            for target in self.target_columns:
                y_new = df_new[target].values
                dmat = xgb.DMatrix(X_mat, label=y_new)
                booster = self.boosters[target]
                # continue training from existing booster
                self.boosters[target] = xgb.train(
                    params=xgb_params or {
                        "objective": "reg:squarederror", "seed": 42},
                    dtrain=dmat,
                    num_boost_round=num_boost_round,
                    xgb_model=booster
                )

        # -- 4) Save artifacts --
        if save:
            joblib.dump(self.preprocessor,
                        os.path.join(self.model_dir, "preprocessor.pkl"))
            joblib.dump(self.boosters,
                        os.path.join(self.model_dir, "boosters.pkl"))
