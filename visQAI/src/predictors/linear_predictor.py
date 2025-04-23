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
            params = {
                "regressor__alpha": trial.suggest_loguniform("alpha", 1e-6, 1e2),
                "regressor__fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
                "regressor__eta0": trial.suggest_loguniform("eta0", 1e-4, 1e-1),
            }
            # apply to pipelineWF
            self.pipeline.set_params(**params)
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

        self.pipeline.set_params(**study.best_params)
        self.pipeline.fit(X, y)

    def update(self,
               df_new: pd.DataFrame,
               save: bool = True,
               tune: bool = False,
               n_trials: int = 10,
               cv: int = 3):
        X_new = df_new[self.feature_columns]
        y_new = df_new[self.target_columns]

        if hasattr(self.pipeline, "named_steps") and "preprocessor" in self.pipeline.named_steps:
            preproc = self.pipeline.named_steps["preprocessor"]
            if hasattr(preproc, "transformers_"):
                for _, transformer, cols in preproc.transformers_:
                    if hasattr(transformer, "partial_fit"):
                        transformer.partial_fit(X_new[cols])
            elif hasattr(preproc, "partial_fit"):
                preproc.partial_fit(X_new)

        reg_name, reg = self.pipeline.steps[-1]
        if hasattr(reg, "partial_fit"):
            X_trans = self.pipeline[:-1].transform(X_new)
            reg.partial_fit(X_trans, y_new.values)
        else:
            self.pipeline.fit(X_new, y_new)

        if tune:
            self._tune_with_optuna(X_new, y_new, n_trials=n_trials, cv=cv)
        if save:
            joblib.dump(self.pipeline,
                        os.path.join(self.model_dir, "pipeline_lm.pkl"))
