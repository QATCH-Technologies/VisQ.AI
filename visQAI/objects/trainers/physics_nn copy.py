from tensorflow.keras import layers
import os
import pickle
from typing import Dict
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
from keras_tuner import RandomSearch, HyperModel
from keras_tuner.engine.hyperparameters import HyperParameters
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from custom_layers import ReverseCumsum

# Feature and target definitions
TARGET_COLS = [
    "Viscosity_100",
    "Viscosity_1000",
    "Viscosity_10000",
    "Viscosity_100000",
    "Viscosity_15000000"
]
NUMERIC_FEATURES = [
    "MW",
    "PI_mean",
    "PI_range",
    "Protein_concentration",
    "Temperature",
    "Sugar_concentration",
    "Surfactant_concentration",
    "Buffer_pH"
]
CATEGORICAL_FEATURES = [
    "Protein_type",
    "Buffer_type",
    "Sugar_type",
    "Surfactant_type"
]

# Penalty layers encapsulate custom TF logic


class MonotonicPenaltyLayer(layers.Layer):
    def call(self, y_pred):
        diff = y_pred[:, :-1] - y_pred[:, 1:]
        violations = tf.nn.relu(-diff)
        return tf.reduce_mean(violations)


class PhysicsPenaltyLayer(layers.Layer):
    def __init__(self, feature_mins, feature_maxs, **kwargs):
        super().__init__(**kwargs)

        # ---- store plain NumPy arrays for get_config() ----
        self.feature_mins = np.array(feature_mins, dtype=np.float32)
        self.feature_maxs = np.array(feature_maxs, dtype=np.float32)
        self.n_numeric = self.feature_mins.shape[0]

        # ---- create TF constants for the actual call() ----
        self._mins_tensor = tf.constant(self.feature_mins, dtype=tf.float32)
        self._maxs_tensor = tf.constant(self.feature_maxs, dtype=tf.float32)

    def call(self, inputs):
        full_scaled, y_pred = inputs

        # slice off just the numeric portion
        numeric_scaled = full_scaled[:, :self.n_numeric]

        # invert the MinMax on just the numeric slice
        numeric_raw = (numeric_scaled * (self._maxs_tensor - self._mins_tensor)
                       + self._mins_tensor)

        # grab the individual raw-unit features
        mw = numeric_raw[:, 0:1]
        pi_mean = numeric_raw[:, 1:2]
        pi_range = numeric_raw[:, 2:3]
        temp = numeric_raw[:, 4:5]
        sugar = numeric_raw[:, 5:6]
        surf = numeric_raw[:, 6:7]

        # build your physics penalty
        terms = [
            tf.square(y_pred - mw * 0.01),
            tf.square(y_pred - 25*(temp - 25)),
            tf.square(y_pred - pi_mean * 0.05),
            tf.square(y_pred - pi_range*0.03),
            tf.square(y_pred - sugar * 0.04),
            tf.square(y_pred - surf * 0.02),
        ]
        return tf.reduce_mean(tf.add_n(terms))

    def get_config(self):
        config = super().get_config()
        # now feature_mins/feature_maxs are NumPy arrays, so .tolist() works
        config.update({
            "feature_mins": self.feature_mins.tolist(),
            "feature_maxs": self.feature_maxs.tolist(),
        })
        return config


class PhysicsHyperModel(HyperModel):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 feature_mins: np.ndarray,
                 feature_maxs: np.ndarray):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.feature_mins = feature_mins
        self.feature_maxs = feature_maxs

    def build(self, hp: HyperParameters):
        # hyperparameters
        interaction_units = hp.Int('interaction_units', 8, 128, sampling='log')
        dense_units = hp.Int('dense_units', 16, 256, sampling='log')
        bottleneck_units = hp.Int('bottleneck_units', 4, 64, sampling='log')
        dropout_rate = hp.Float('dropout_rate', 0.0, 0.5)
        l2_reg = hp.Float('l2_reg', 1e-6, 1e-2, sampling='log')
        learning_rate = hp.Float('learning_rate', 1e-5, 1e-2, sampling='log')
        # batch size is needed in fit, still register choice here
        hp.Choice('batch_size', [16, 32, 64])

        # model architecture
        inp = layers.Input(shape=(self.input_dim,), name='features')
        interaction = layers.Dense(
            interaction_units,
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg)
        )(inp)
        x = layers.Concatenate()([inp, interaction])
        for _ in range(2):
            x = layers.Dense(dense_units,
                             activation='relu',
                             kernel_regularizer=regularizers.l2(l2_reg))(x)
            x = layers.Dropout(dropout_rate)(x)
        bottleneck = layers.Dense(
            bottleneck_units,
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg)
        )(x)
        raw = layers.Dense(self.output_dim,
                           activation=None,
                           kernel_regularizer=regularizers.l2(l2_reg))(bottleneck)
        pos = layers.Activation('relu')(raw)
        mono_out = ReverseCumsum()(pos)

        # after you have `mono_out`:
        mono_pen = MonotonicPenaltyLayer()(mono_out)

        # instantiate the penalty layer with your min/max
        phys_pen = PhysicsPenaltyLayer(self.feature_mins,
                                       self.feature_maxs)([inp, mono_out])

        model = models.Model(inputs=inp, outputs=mono_out)
        model.add_loss(0.1 * mono_pen)
        model.add_loss(0.01 * phys_pen)
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            run_eagerly=True
        )
        return model

    def fit(self, hp: HyperParameters, model: tf.keras.Model, x, y, **kwargs):
        # retrieve batch size choice
        batch_size = hp.get('batch_size')
        return model.fit(x, y, batch_size=batch_size, **kwargs)


class PhysicsNNTrainer:
    def __init__(
        self,
        data_path: str,
        test_size: float = 0.2,
        random_state: int = 42
    ):
        self.data_path = data_path
        self.test_size = test_size
        self.rs = random_state
        self._load_data()

    def _load_data(self) -> None:
        df = pd.read_csv(self.data_path)
        # drop rows with missing target values to avoid NaNs in training
        df.dropna(subset=TARGET_COLS, inplace=True)
        for col in NUMERIC_FEATURES:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        missing = set(TARGET_COLS + NUMERIC_FEATURES +
                      CATEGORICAL_FEATURES) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        y = df[TARGET_COLS].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.rs
        )
        # preprocessing pipelines
        num_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', MinMaxScaler())
        ])
        cat_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        pre = ColumnTransformer([
            ('num', num_pipe, NUMERIC_FEATURES),
            ('cat', cat_pipe, CATEGORICAL_FEATURES)
        ])
        self.X_train = pre.fit_transform(self.X_train)
        num_scaler = pre.named_transformers_['num'].named_steps['scaler']

        self.feature_mins = num_scaler.data_min_    # shape (8,)
        self.feature_maxs = num_scaler.data_max_    # shape (8,)
        self.preprocessor = pre

    def tune(
        self,
        max_trials: int = 30,
        executions_per_trial: int = 1,
        directory: str = 'ktuner_logs'
    ) -> HyperParameters:
        hypermodel = PhysicsHyperModel(
            input_dim=self.X_train.shape[1],
            output_dim=self.y_train.shape[1],
            feature_mins=self.feature_mins,
            feature_maxs=self.feature_maxs,
        )
        tuner = RandomSearch(
            hypermodel=hypermodel,
            objective='val_loss',
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            directory=directory,
            project_name='physics_nn',
            overwrite=True
        )
        tuner.search(
            self.X_train,
            self.y_train,
            validation_split=0.2,
            epochs=30,
            verbose=1
        )
        return tuner.get_best_hyperparameters(num_trials=1)[0]

    def train_final(
        self,
        best_hp: HyperParameters,
        epochs: int = 100,
        output_dir: str = './models'
    ) -> None:
        # ensure that data + preprocessor are loaded
        if not hasattr(self, 'preprocessor'):
            # this will (re-)load X_train, y_train and set self.preprocessor
            self._load_data()

        os.makedirs(output_dir, exist_ok=True)

        # build & train your model as before
        model = PhysicsHyperModel(
            input_dim=self.X_train.shape[1],
            output_dim=self.y_train.shape[1],
            feature_mins=self.feature_mins,
            feature_maxs=self.feature_maxs
        ).build(best_hp)

        batch_size = best_hp.get('batch_size')
        model.fit(
            self.X_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size
        )

        # save the model
        model.save(os.path.join(output_dir, 'physics_nn.h5'))

        # now pickle the pipeline you loaded above
        with open(os.path.join(output_dir, 'pipeline.pkl'), 'wb') as f:
            pickle.dump(self.preprocessor, f)

    def test(
        self,
        model_path: str = './models/physics_nn.h5',
        target_cols: list = TARGET_COLS
    ) -> None:
        import matplotlib.pyplot as plt
        from tensorflow.keras.models import load_model

        # 1) load model
        model = load_model(
            model_path,
            custom_objects={
                'ReverseCumsum': ReverseCumsum,
                'MonotonicPenaltyLayer': MonotonicPenaltyLayer,
                'PhysicsPenaltyLayer': PhysicsPenaltyLayer
            }
        )

        # 2) read in the full DataFrame
        df = pd.read_csv(self.data_path)

        # 3) coerce numeric features exactly as in _load_data()
        for col in NUMERIC_FEATURES:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 4) now transform
        X_full = self.preprocessor.transform(
            df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        )
        y_full = df[TARGET_COLS].values

        # 5) plot Predictions vs Actual
        for idx in range(len(X_full)):
            x_row = X_full[idx:idx+1]
            actual = y_full[idx]
            pred = model.predict(x_row).flatten()
            print(pred)
            plt.figure(figsize=(6, 4))
            plt.plot(range(len(target_cols)), pred,
                     marker='o', label='Predicted')
            plt.plot(range(len(target_cols)), actual,
                     marker='x', linestyle='--', label='Actual')
            plt.xticks(range(len(target_cols)), target_cols, rotation=45)
            plt.xlabel('Shear rate index')
            plt.ylabel('Viscosity')
            plt.title(f'Sample #{idx}: Pred vs Actual Profile')
            plt.legend()
            plt.tight_layout()
            plt.show()


if __name__ == '__main__':
    trainer = PhysicsNNTrainer(
        data_path='content/formulation_data_05152025.csv')
    print("Starting hyperparameter tuning...")
    best_hp = trainer.tune(max_trials=30)
    print("Best hyperparameters:", {k: best_hp.get(k)
          for k in best_hp.values.keys()})
    print("Training final model...")
    trainer.train_final(best_hp=best_hp)
    print("Done. Artifacts saved to ./models")
    trainer.test()
