import os
import joblib
import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score


class LinearPredictor:
    def __init__(self, model_dir,
                 feature_columns=None,
                 target_columns=None):
        self.model_dir = model_dir
        self.pipeline = joblib.load(
            os.path.join(model_dir, "pipeline_lm.pkl")
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
        preds = self.pipeline.predict(X)
        return pd.DataFrame(preds,
                            columns=self.target_columns,
                            index=df_new.index)

    def _tune_with_optuna(self, X, y, n_trials: int, cv: int):
        """Run an Optuna study to tune the final regressor via neg‑MSE CV."""
        def objective(trial):
            # suggest some hyperparameters for a generic linear regressor
            params = {
                # if your reg is e.g. Ridge or ElasticNet
                "regressor__alpha": trial.suggest_loguniform("alpha", 1e-6, 1e2),
                # toggle intercept fitting
                "regressor__fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
                # learning rate (if using SGDRegressor)
                "regressor__eta0": trial.suggest_loguniform("eta0", 1e-4, 1e-1),
            }
            # apply to pipeline
            self.pipeline.set_params(**params)
            # evaluate with k‐fold CV on the raw features
            scores = cross_val_score(
                self.pipeline,
                X, y,
                cv=cv,
                scoring="neg_mean_squared_error",
                n_jobs=-1
            )
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # set the best params and do a full refit on this batch
        self.pipeline.set_params(**study.best_params)
        self.pipeline.fit(X, y)

    def update(self,
               df_new: pd.DataFrame,
               save: bool = True,
               tune: bool = False,
               n_trials: int = 10,
               cv: int = 3):
        """
        Incrementally update the preprocessing + linear regressor on new data,
        then optionally run an Optuna CV tuning sweep.

        Args:
          df_new: DataFrame with feature_columns + target_columns.
          save:   If True, overwrite pipeline_lm.pkl on disk.
          tune:   If True, run Optuna tuning (n_trials, cv) AFTER the weight update.
          n_trials: Number of Optuna trials.
          cv:     Number of cross‑validation folds.
        """
        # 1) Extract X/y
        X_new = df_new[self.feature_columns]
        y_new = df_new[self.target_columns]

        # 2) Incrementally update any partial_fit transformers
        if hasattr(self.pipeline, "named_steps") and "preprocessor" in self.pipeline.named_steps:
            preproc = self.pipeline.named_steps["preprocessor"]
            if hasattr(preproc, "transformers_"):
                for _, transformer, cols in preproc.transformers_:
                    if hasattr(transformer, "partial_fit"):
                        transformer.partial_fit(X_new[cols])
            elif hasattr(preproc, "partial_fit"):
                preproc.partial_fit(X_new)

        # 3) Incrementally update the regressor if possible
        reg_name, reg = self.pipeline.steps[-1]
        if hasattr(reg, "partial_fit"):
            X_trans = self.pipeline[:-1].transform(X_new)
            # many linear regressors take 2D y
            reg.partial_fit(X_trans, y_new.values)
        else:
            # full retrain on new batch
            self.pipeline.fit(X_new, y_new)

        # 4) Optional Optuna tuning on this batch
        if tune:
            # tune on raw X/y so pipeline’s preprocessor is included in CV
            self._tune_with_optuna(X_new, y_new, n_trials=n_trials, cv=cv)

        # 5) Save updated pipeline
        if save:
            joblib.dump(self.pipeline,
                        os.path.join(self.model_dir, "pipeline_lm.pkl"))
