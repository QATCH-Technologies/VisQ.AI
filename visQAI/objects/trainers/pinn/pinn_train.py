# trainer.py
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
from keras_tuner import BayesianOptimization
from pinn_domain import DataLoader
from pinn_net import MLPHyperModel
from pinn_loss import composite_loss
from pinn_optimizer import PINNBayes
from tensorflow.keras.optimizers.schedules import CosineDecay


class PINNTrainer:
    """
    Trainer class for physics-informed neural network with hyperparameter tuning and
    physics-based retraining loops.
    """

    def __init__(
        self,
        csv_path: str,
        tuner_dir: str = "tuner_logs",
        project_name: str = "pinn_viscosity_bayes",
        test_size: float = 0.2,
        random_state: int = 42,
        strat_bins: int = 10,
        max_trials: int = 20,
        initial_random_trials: int = 5,
        tuner_overwrite: bool = True,
    ):
        # Paths and parameters
        self.csv_path = csv_path
        self.test_size = test_size
        self.random_state = random_state
        self.strat_bins = strat_bins

        # Tuner config
        self.tuner_dir = tuner_dir
        self.project_name = project_name
        self.max_trials = max_trials
        self.initial_random_trials = initial_random_trials
        self.tuner_overwrite = tuner_overwrite

        # Placeholders
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.feature_names = []
        self.hypermodel = None
        self.tuner = None
        self.best_models = []

        # Global optimizer for priming slots
        self.global_optimizer = tf.keras.optimizers.Adam()

    def load_and_split_data(self) -> None:
        """
        Load CSV, preprocess, and split into train/validation.
        """
        loader = DataLoader(self.csv_path)
        loader.load()
        loader.build_preprocessor()
        X, y = loader.split(preprocess=True)

        y_mean = np.mean(y, axis=1)
        y_bins = pd.qcut(
            y_mean,
            q=self.strat_bins,
            labels=False,
            duplicates='drop'
        )
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self.test_size,
            random_state=self.random_state
        )
        train_idx, val_idx = next(splitter.split(X, y_bins))

        # assign
        self.X_train = X[train_idx].astype("float32")
        self.y_train = y[train_idx].astype("float32")
        self.X_val = X[val_idx].astype("float32")
        self.y_val = y[val_idx].astype("float32")
        self.feature_names = loader.NUMERIC_FEATURES

    def setup_hypermodel_and_tuner(self) -> None:
        """
        Instantiate HyperModel and BayesianOptimization tuner.
        """
        input_dim = self.X_train.shape[1]
        output_dim = self.y_train.shape[1]
        self.hypermodel = MLPHyperModel(
            input_dim,
            output_dim,
            self.feature_names
        )
        self.tuner = PINNBayes(
            loss_fn=composite_loss,
            constraints=getattr(self.hypermodel, '_physics_constraints', []),
            data_weight=1.0,
            metrics=['MeanAbsoluteError'],
            hypermodel=self.hypermodel,
            objective='val_loss',
            max_trials=self.max_trials,
            num_initial_points=self.initial_random_trials,
            alpha=1e-4,
            beta=2.6,
            executions_per_trial=1,
            directory=self.tuner_dir,
            project_name=self.project_name,
            overwrite=self.tuner_overwrite,
        )

    def prime_optimizer_slots(self) -> None:
        """
        Prime optimizer slot variables by applying zero gradients.
        """
        # prime global
        space = self.tuner.oracle.get_space()
        dummy_model = self.hypermodel.build(space)
        zero_grads = [tf.zeros_like(v)
                      for v in dummy_model.trainable_variables]
        self.global_optimizer.apply_gradients(
            zip(zero_grads, dummy_model.trainable_variables)
        )

    def search_hyperparameters(
        self,
        epochs: int = 5,
        batch_size: int = 16
    ) -> None:
        """
        Run hyperparameter search.
        """
        self.tuner.search(
            X=self.X_train,
            y=self.y_train,
            X_val=self.X_val,
            y_val=self.y_val,
            epochs=epochs,
            batch_size=batch_size
        )

    @tf.function
    def train_step(self, model, x_batch, y_batch, constraints):
        y_batch = tf.cast(y_batch, tf.float32)
        with tf.GradientTape() as tape:
            total_loss, data_loss, phys_loss = composite_loss(
                model, x_batch, y_batch,
                constraints=constraints,
                data_weight=1.0
            )
        grads = tape.gradient(total_loss, model.trainable_variables)
        self.global_optimizer.apply_gradients(
            zip(grads, model.trainable_variables)
        )
        return total_loss, data_loss, phys_loss

    def run_trial_with_physics(
        self,
        model: tf.keras.Model,
        extreme_fraction: float = 0.2,
        epochs: int = 100,
        batch_size: int = 16
    ) -> tf.keras.Model:
        """
        Retrain a built model with physics-informed batching and cosine LR.
        """
        constraints = getattr(model, '_physics_constraints', [])
        # compute extremes
        deviations = np.linalg.norm(
            self.y_train - self.y_train.mean(axis=0), axis=1
        )
        n_extreme = int(extreme_fraction * len(deviations))
        extreme_idx = np.argsort(deviations)[-n_extreme:]
        normal_idx = np.setdiff1d(np.arange(len(self.y_train)), extreme_idx)

        # lr schedule
        steps = int(np.ceil(len(self.X_train) / batch_size)) * epochs
        initial_lr = float(self.global_optimizer.learning_rate)
        lr_sched = CosineDecay(initial_lr, decay_steps=steps, alpha=0.0)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sched)
        zero_grads = [tf.zeros_like(v) for v in model.trainable_variables]
        optimizer.apply_gradients(
            zip(zero_grads, model.trainable_variables)
        )

        best_val, wait = float('inf'), 0
        n_ext_batch = max(1, int(extreme_fraction * batch_size))

        for epoch in range(1, epochs+1):
            np.random.shuffle(extreme_idx)
            np.random.shuffle(normal_idx)
            ptr_ext, ptr_norm = 0, 0
            while ptr_norm < len(normal_idx):
                if ptr_ext + n_ext_batch > len(extreme_idx):
                    np.random.shuffle(extreme_idx)
                    ptr_ext = 0
                batch_ext = extreme_idx[ptr_ext:ptr_ext+n_ext_batch]
                batch_norm = normal_idx[ptr_norm:ptr_norm +
                                        (batch_size-n_ext_batch)]
                ptr_ext += n_ext_batch
                ptr_norm += (batch_size-n_ext_batch)
                idx = np.concatenate([batch_ext, batch_norm])
                np.random.shuffle(idx)
                x_b = tf.convert_to_tensor(self.X_train[idx], tf.float32)
                y_b = tf.convert_to_tensor(self.y_train[idx], tf.float32)
                self.train_step(model, x_b, y_b, constraints)
            # validate
            val_total, val_data, val_phys = composite_loss(
                model, self.X_val, self.y_val, constraints=constraints
            )
            print(f"Epoch {epoch:03d} | val_data={val_data:.4e}"
                  f" | val_phys={val_phys:.4e} | lr={optimizer._decayed_lr(tf.float32):.3e}")
            if val_data < best_val:
                best_val, wait = val_data, 0
            else:
                wait += 1
                if wait >= 10:
                    print("Early stopping at epoch", epoch)
                    break
        return model

    def retrain_best_models(
        self,
        num_models: int = 3,
        extreme_fraction: float = 0.2,
        epochs: int = 100,
        batch_size: int = 16
    ) -> None:
        """
        Build, prime, and retrain the top-n hyperparameter models.
        """
        for trial in self.tuner.oracle.get_best_trials(num_trials=num_models):
            hp = trial.hyperparameters
            model = self.hypermodel.build(hp)
            # prime
            zero_grads = [tf.zeros_like(v) for v in model.trainable_variables]
            self.global_optimizer.apply_gradients(
                zip(zero_grads, model.trainable_variables)
            )
            # retrain
            retrained = self.run_trial_with_physics(
                model,
                extreme_fraction=extreme_fraction,
                epochs=epochs,
                batch_size=batch_size
            )
            self.best_models.append(retrained)

    def save_best_model(self, path: str) -> None:
        """
        Save the final best model to disk.
        """
        if not self.best_models:
            raise ValueError(
                "No models available. Run retrain_best_models first.")
        self.best_models[-1].save(path)

    def train_full_pipeline(
        self,
        search_epochs: int = 5,
        search_batch: int = 16,
        retrain_epochs: int = 100,
        retrain_batch: int = 16,
        retrain_extreme: float = 0.2,
        num_final_models: int = 3,
        save_path: str = 'best_pinn_model.h5'
    ) -> None:
        """
        Execute full workflow: load data, tune, retrain, and save.
        """
        self.load_and_split_data()
        self.setup_hypermodel_and_tuner()
        self.prime_optimizer_slots()
        self.search_hyperparameters(
            epochs=search_epochs,
            batch_size=search_batch
        )
        self.retrain_best_models(
            num_models=num_final_models,
            extreme_fraction=retrain_extreme,
            epochs=retrain_epochs,
            batch_size=retrain_batch
        )
        self.save_best_model(save_path)


if __name__ == '__main__':
    trainer = PINNTrainer(
        csv_path=os.path.join('content', 'formulation_data_05222025.csv')
    )
    trainer.train_full_pipeline()
