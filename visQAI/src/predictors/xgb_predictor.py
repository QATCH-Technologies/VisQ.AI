# xgb_predictor.py
from base_predictor import BasePredictor
import os
import joblib
import pandas as pd
import xgboost as xgb
import optuna
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score


class XGBPredictor(BasePredictor):
    def __init__(self, model_dir,
                 feature_columns=None,
                 target_columns=None):
        self.model_dir = model_dir
        self.preprocessor = joblib.load(
            os.path.join(model_dir, "preprocessor.pkl")
        )
        self.boosters = joblib.load(
            os.path.join(model_dir, "boosters.pkl")
        )

        self.feature_columns = feature_columns or [
            "Protein type", "Protein", "Temperature", "Buffer",
            "Sugar", "Sugar (M)", "Surfactant", "TWEEN"
        ]
        self.target_columns = target_columns or [
            "Viscosity100", "Viscosity1000", "Viscosity10000",
            "Viscosity100000", "Viscosity15000000"
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

    def update(self,
               df_new: pd.DataFrame,
               xgb_params: dict,
               num_boost_round: int = 10,
               tune: bool = False,
               n_trials: int = 10,
               cv: int = 3,
               save: bool = True):
        X = df_new[self.feature_columns]

        # Update preprocessor if it supports incremental learning
        if hasattr(self.preprocessor, "transformers_"):
            for _, transformer, cols in self.preprocessor.transformers_:
                if hasattr(transformer, "partial_fit"):
                    transformer.partial_fit(X[cols])
        else:
            self.preprocessor.fit(X)

        X_mat = self.preprocessor.transform(X)

        for target in self.target_columns:
            y = df_new[target]
            dmat = xgb.DMatrix(X_mat, label=y)
            booster = self.boosters[target]
            updated = xgb.train(
                params=xgb_params,
                dtrain=dmat,
                num_boost_round=num_boost_round,
                xgb_model=booster
            )
            self.boosters[target] = updated

        if tune:
            for target in self.target_columns:
                y = df_new[target].values
                best = self._tune_target(X_mat, y, n_trials=n_trials, cv=cv)
                model = XGBRegressor(
                    **best, n_estimators=num_boost_round, use_label_encoder=False)
                model.fit(X_mat, y)
                self.boosters[target] = model.get_booster()

        if save:
            joblib.dump(self.preprocessor,
                        os.path.join(self.model_dir, "preprocessor.pkl"))
            joblib.dump(self.boosters,
                        os.path.join(self.model_dir, "boosters.pkl"))

    def _tune_target(self, X_mat, y, n_trials: int, cv: int):
        def objective(trial):
            params = {
                "eta": trial.suggest_loguniform("eta", 1e-3, 0.3),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
                "objective": "reg:squarederror",
                "verbosity": 0
            }
            model = XGBRegressor(**params, n_estimators=100,
                                 use_label_encoder=False)
            scores = cross_val_score(
                model, X_mat, y,
                cv=cv,
                scoring="neg_mean_squared_error",
                n_jobs=-1
            )
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        return study.best_params
