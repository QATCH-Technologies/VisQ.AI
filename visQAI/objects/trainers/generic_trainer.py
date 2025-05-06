import os
import numpy as np
import joblib
import keras_tuner as kt
from keras_tuner import Objective
import tensorflow as tf
from sklearn.model_selection import KFold
from keras import callbacks
from generic_hypermodel import GenericHyperModel
from visQ_data_processor import VisQDataProcessor
from typing import Callable, Dict, Any, Union


class GenericTrainer:
    """Tuner + CV + save workflow for *any* Keras architecture."""

    def __init__(
        self,
        df,
        builder: Callable[..., tf.keras.Model],
        hyperparam_space: Dict[str, Dict[str, Any]],
        compile_args: Union[Dict[str, Any],
                            Callable[[Dict[str, Any]], Dict[str, Any]]],
        cv_splits: int = 3,
        random_state: int = 42,
    ):
        # 1) preprocessor + data
        self.processor = VisQDataProcessor()
        X_df, y_df = self.processor.fit(df)
        self.X = X_df.values
        self.y = y_df.values
        self.input_dim = X_df.shape[1]
        self.output_dim = y_df.shape[1]

        # 2) architecture + tuner config
        self.builder = builder
        self.hyperparam_space = hyperparam_space
        self.compile_args = compile_args
        self.cv_splits = cv_splits
        self.random_state = random_state

        self.best_hp: Optional[kt.HyperParameters] = None
        self.model: Optional[tf.keras.Model] = None

    def tune(
        self,
        max_trials: int = 20,
        executions_per_trial: int = 1,
        epochs: int = 20,
        batch_size: int = 32,
        validation_split: float = 0.2,
        directory: str = "visQAI/objects",
        project_name: str = "generic_tuner",
        objective_name: str = "val_rmse",
    ) -> kt.Tuner:
        """Search for best HPs using KerasTuner.RandomSearch."""
        hypermodel = GenericHyperModel(
            self.input_dim,
            self.output_dim,
            builder=self.builder,
            hyperparam_space=self.hyperparam_space,
            compile_args=self.compile_args,
        )
        tuner = kt.RandomSearch(
            hypermodel,
            objective=Objective(objective_name, direction="min"),
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            directory=directory,
            project_name=project_name,
            seed=self.random_state,
        )

        tuner.search(
            x=self.X,
            y=self.y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[callbacks.EarlyStopping(
                patience=3, restore_best_weights=True)],
            verbose=1,
        )
        self.best_hp = tuner.get_best_hyperparameters(1)[0]
        print("Best HPs:", self.best_hp.values)
        return tuner

    def cross_validate(
        self,
        epochs: int = 50,
        batch_size: int = 32,
    ) -> list[float]:
        """K-Fold CV with the best HPs, then retrain final model on all data."""
        if self.best_hp is None:
            raise RuntimeError("You must call tune() first.")
        rmses = []
        cv = KFold(
            n_splits=self.cv_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        for fold, (tr, val) in enumerate(cv.split(self.X), start=1):
            X_tr, X_val = self.X[tr], self.X[val]
            y_tr, y_val = self.y[tr], self.y[val]

            model = GenericHyperModel(
                self.input_dim,
                self.output_dim,
                self.builder,
                self.hyperparam_space,
                self.compile_args,
            ).build(self.best_hp)  # rebuild with best HP
            model.fit(
                X_tr,
                y_tr,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=[callbacks.EarlyStopping(
                    patience=5, restore_best_weights=True)],
                verbose=1,
            )
            rmse = model.evaluate(X_val, y_val, verbose=0)[1]
            print(f"Fold {fold} RMSE: {rmse:.4f}")
            rmses.append(rmse)

        # final retrain
        print("Retraining on full dataset…")
        self.model = GenericHyperModel(
            self.input_dim,
            self.output_dim,
            self.builder,
            self.hyperparam_space,
            self.compile_args,
        ).build(self.best_hp)
        self.model.fit(
            self.X,
            self.y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[callbacks.EarlyStopping(
                patience=5, restore_best_weights=True)],
            verbose=1,
        )
        return rmses

    def save(self, model_dir: str):
        """Save final model + preprocessor."""
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model.keras")
        self.model.save(model_path)
        print(f"[INFO] Saved model to {model_path}")

        prep_path = os.path.join(model_dir, "preprocessor.pkl")
        joblib.dump(self.processor, prep_path)
        print(f"[INFO] Saved preprocessor to {prep_path}")
