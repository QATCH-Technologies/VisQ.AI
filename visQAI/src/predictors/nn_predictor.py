# nn_predictor.py
from base_predictor import BasePredictor
import os
import joblib
import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score


class NNPredictor(BasePredictor):
    def __init__(self, model_dir, target_columns=None):
        self.model_dir = model_dir
        self.pipeline = joblib.load(os.path.join(model_dir, "pipeline_nn.pkl"))
        self.target_columns = target_columns or [
            "Viscosity100", "Viscosity1000", "Viscosity10000",
            "Viscosity100000", "Viscosity15000000"
        ]

    def predict(self, df_new: pd.DataFrame) -> pd.DataFrame:
        preds = self.pipeline.predict(df_new)
        return pd.DataFrame(preds, columns=self.target_columns, index=df_new.index)

    def update(self,
               df_new: pd.DataFrame,
               epochs: int = 1,
               tune: bool = False,
               n_trials: int = 10,
               cv: int = 3,
               save: bool = True):

        X_new = df_new.drop(columns=self.target_columns)
        y_new = df_new[self.target_columns]

        # Optional: Update scalers with new data
        if hasattr(self.pipeline, "named_steps") and "preprocessor" in self.pipeline.named_steps:
            preproc = self.pipeline.named_steps["preprocessor"]
            if hasattr(preproc, "transformers_"):
                for _, transformer, cols in preproc.transformers_:
                    if hasattr(transformer, "partial_fit"):
                        transformer.partial_fit(X_new[cols])
            elif hasattr(preproc, "partial_fit"):
                preproc.partial_fit(X_new)

        # Update or refit the regressor
        reg_name, reg = self.pipeline.steps[-1]
        if hasattr(reg, "partial_fit"):
            X_trans = self.pipeline[:-1].transform(X_new)
            for _ in range(epochs):
                reg.partial_fit(X_trans, y_new.values)
        else:
            self.pipeline.fit(X_new, y_new)

        # Optional tuning with Optuna
        if tune:
            X_trans = self.pipeline[:-1].transform(X_new)
            self._tune_with_optuna(X_trans, y_new.values,
                                   n_trials=n_trials, cv=cv)

        # Save model after update
        if save:
            joblib.dump(self.pipeline, os.path.join(
                self.model_dir, "pipeline_nn.pkl"))

    def _tune_with_optuna(self, X, y, n_trials: int, cv: int):
        def objective(trial):
            params = {
                "regressor__hidden_layer_sizes":
                    trial.suggest_categorical("hidden_layer_sizes", [
                                              (50,), (100,), (50, 50)]),
                "regressor__alpha":
                    trial.suggest_loguniform("alpha", 1e-6, 1e-2),
                "regressor__learning_rate_init":
                    trial.suggest_loguniform("lr_init", 1e-4, 1e-2)
            }
            self.pipeline.set_params(**params)
            scores = cross_val_score(
                self.pipeline, X, y, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1)
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        self.pipeline.set_params(**study.best_params)
        self.pipeline.fit(X, y)
