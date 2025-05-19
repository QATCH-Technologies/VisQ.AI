import os
import pickle
from typing import Dict
import pandas as pd
import numpy as np
import tensorflow as tf
from keras import layers, models, regularizers
from keras.optimizers import Adam
from keras_tuner import RandomSearch, HyperModel
from keras_tuner.engine.hyperparameters import HyperParameters
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from custom_layers import ReverseCumsum
from typing import List
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
from keras.models import load_model
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
        # store for get_config()
        self.feature_mins = np.array(feature_mins, dtype=np.float32)
        self.feature_maxs = np.array(feature_maxs, dtype=np.float32)
        self.n_numeric = self.feature_mins.shape[0]

        # build TF constants
        self._mins = tf.constant(self.feature_mins, dtype=tf.float32)
        self._maxs = tf.constant(self.feature_maxs, dtype=tf.float32)

        # your original k-coefficients
        self.k_mw = tf.constant(0.01, dtype=tf.float32)
        self.k_prot = tf.constant(0.05, dtype=tf.float32)
        self.k_sugar = tf.constant(0.04, dtype=tf.float32)
        self.k_surf = tf.constant(-0.02, dtype=tf.float32)
        self.k_pH = tf.constant(1.0, dtype=tf.float32)
        self.epsilon = tf.constant(1e-3, dtype=tf.float32)

        # new physics constants
        self.slope_target = tf.constant(-0.7, dtype=tf.float32)
        self.log_shear_rates = tf.constant(
            np.log([100, 1e3, 1e4, 1e5, 1.5e7], dtype=np.float32),
            dtype=tf.float32
        )
        self.arrh_slope = tf.constant(5000.0, dtype=tf.float32)
        self.c_star = tf.constant(150.0, dtype=tf.float32)
        self.ev_n = tf.constant(2.0, dtype=tf.float32)
        self.NA = tf.constant(6.022e23, dtype=tf.float32)
        self.rho = tf.constant(1e3, dtype=tf.float32)
        self.gauss_sigma = tf.constant(0.5, dtype=tf.float32)
        self.k_I = tf.constant(1.0, dtype=tf.float32)

    def call(self, inputs):
        full_scaled, y_pred = inputs

        # 1) un-scale numeric features
        num = full_scaled[:, :self.n_numeric]
        raw = num * (self._maxs - self._mins) + self._mins

        # 2) slice out each feature
        mw = raw[:, 0:1]
        pi_m = raw[:, 1:2]
        pi_r = raw[:, 2:3]
        prot = raw[:, 3:4]
        temp = raw[:, 4:5]
        sugar = raw[:, 5:6]
        surf = raw[:, 6:7]
        pH = raw[:, 7:8]

        # 3) build all target curves
        targ_mw = mw * self.k_mw
        targ_prot = prot * self.k_prot
        targ_sugar = sugar * self.k_sugar
        targ_surf = surf * self.k_surf
        dpH = tf.abs(pH - pi_m) + self.epsilon
        targ_pH = self.k_pH / dpH
        targ_temp = (temp - 25.0) * 25.0
        targ_pi_r = pi_r * 0.03

        # 4) assemble *scalar* penalties
        penalties = [
            tf.reduce_mean(tf.square(y_pred - targ_mw)),
            tf.reduce_mean(tf.square(y_pred - targ_prot)),
            tf.reduce_mean(tf.square(y_pred - targ_sugar)),
            tf.reduce_mean(tf.square(y_pred - targ_surf)),
            tf.reduce_mean(tf.square(y_pred - targ_pH)),
            tf.reduce_mean(tf.square(y_pred - targ_temp)),
            tf.reduce_mean(tf.square(y_pred - targ_pi_r)),
        ]

        # — shear-thinning exponent penalty —
        ly = tf.math.log(y_pred + 1e-6)
        dlog = ly[:, 1:] - ly[:, :-1]
        dlogg = self.log_shear_rates[1:] - self.log_shear_rates[:-1]
        slopes = dlog / dlogg
        penalties.append(tf.reduce_mean(tf.square(slopes - self.slope_target)))

        # # — Arrhenius T dependence —
        # invT = 1.0 / (temp + 273.15)
        # logy = tf.math.log(y_pred + 1e-6)
        # cov_IT = tf.reduce_mean((invT - tf.reduce_mean(invT)) *
        #                         (logy - tf.reduce_mean(logy)),  axis=0)
        # arrh_pen = tf.reduce_mean(tf.square(cov_IT - self.arrh_slope))
        # penalties.append(arrh_pen)

        # # — excluded-volume divergence —
        # ev_targ = tf.pow(1 - prot/self.c_star, -self.ev_n)
        # penalties.append(tf.reduce_mean(tf.square(y_pred - ev_targ)))

        # # — Einstein dilute limit —
        # phi = prot * mw / (self.rho * self.NA + self.epsilon)
        # ein_targ = 1 + 2.5 * phi
        # penalties.append(tf.reduce_mean(tf.square(y_pred - ein_targ)))

        # — Gaussian bell around pI —
        gauss_t = tf.exp(-tf.square(pH - pi_m)/(2*self.gauss_sigma**2))
        penalties.append(tf.reduce_mean(tf.square(y_pred - gauss_t)))

        # — ionic strength screen (if you slice I above) —
        # ion_t   = 1.0/(1 + self.k_I * tf.sqrt(I + self.epsilon))
        # penalties.append(tf.reduce_mean(tf.square(y_pred - ion_t)))

        # 5) sum all scalars into one loss
        return tf.add_n(penalties)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "feature_mins": self.feature_mins.tolist(),
            "feature_maxs": self.feature_maxs.tolist(),
            # add any of the new constants here if you like
        })
        return cfg


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
        raw = layers.Dense(self.output_dim, activation=None,
                           kernel_regularizer=regularizers.l2(l2_reg))(bottleneck)
        pos = layers.Activation('relu')(raw)

        # instantiate our enriched physics penalty
        phys_pen = PhysicsPenaltyLayer(self.feature_mins,
                                       self.feature_maxs)([inp, pos])
        phys_weight = hp.Float('phys_weight', 1e-6, 1e-2, sampling='log')

        # # optional: keep monotonicity too
        # mono_pen = MonotonicPenaltyLayer()(pos)

        model = models.Model(inputs=inp, outputs=pos)
        # model.add_loss(0.1 * mono_pen)
        model.add_loss(phys_weight * phys_pen)

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
        target_cols: List[str] = TARGET_COLS,
    ) -> None:
        # 1) load model
        model = load_model(
            model_path,
            custom_objects={
                'ReverseCumsum': ReverseCumsum,
                'PhysicsPenaltyLayer': PhysicsPenaltyLayer
            }
        )

        # 2) read full data
        df = pd.read_csv(self.data_path)

        # 3) coerce numeric features
        for col in NUMERIC_FEATURES:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 4) transform inputs
        X_full = self.preprocessor.transform(
            df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
        y_full = df[target_cols].values  # shape = (n_samples, n_shearrates)

        # 5) predict in one go
        preds_full = model.predict(X_full, verbose=0)

        # 6) loop over each shear‐rate
        for idx, rate in enumerate(target_cols):
            y_true = y_full[:, idx]
            y_pred = preds_full[:, idx]

            # mask out any nan or infinite
            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            if not mask.any():
                print(f"→ No valid data for {rate}, skipping.")
                continue

            y_true = y_true[mask]
            y_pred = y_pred[mask]

            # compute metrics
            r2 = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)

            # plot
            plt.figure(figsize=(6, 4))
            plt.scatter(y_true, y_pred, alpha=0.6, label='Data points')
            mn, mx = min(y_true.min(), y_pred.min()), max(
                y_true.max(), y_pred.max())
            plt.plot([mn, mx], [mn, mx], ls='--', label='Ideal (y = x)')
            plt.text(
                0.05, 0.95,
                f'$R^2$ = {r2:.2f}\nMAE = {mae:.2f}',
                transform=plt.gca().transAxes,
                va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
            plt.xlabel('Actual Viscosity')
            plt.ylabel('Predicted Viscosity')
            plt.title(f'physics_nn — Pred vs Actual @ {rate}')
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
