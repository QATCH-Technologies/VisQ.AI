# predictor_wrapper.py

import numpy as np
from typing import Any, Union, Tuple
import tensorflow as tf
from keras.layers import TFSMLayer
from keras import Model, Input
from tensorflow.keras import Input, Model, layers


class Predictor:
    """
    Wrapper around a Keras multi-output regression model. Implements:
      - set_model(model)              ⇒ attaches loaded Keras model
      - predict(X)                    ⇒ returns numpy array of shape (n_samples, n_targets)
      - update(X_new, y_new, model)   ⇒ performs transfer learning on the same Keras model
    """

    def __init__(self):
        self.model: tf.keras.Model | None = None

    def set_model(self, model: Any):
        """
        Called by Generic Predictor.__init__ to attach the loaded model.
        """
        self.model = model

    def predict(
        self,
        X: Any,
        return_uncertainty: bool = False,
        n_samples: int = 50
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if self.model is None:
            raise ValueError("No model set. Call set_model() first.")

        X_arr = np.array(X)

        if not return_uncertainty:
            preds = self.model.predict(X_arr)
            preds = np.atleast_2d(preds)
            return preds

        preds = []
        for _ in range(n_samples):
            y = self.model(X_arr, training=True)
            if isinstance(y, tf.Tensor):
                y = y.numpy()
            preds.append(y)
        samples = np.stack(preds, axis=0)

        mean_preds = samples.mean(axis=0)
        std_preds = samples.std(axis=0)

        return mean_preds, std_preds

    def update(self, X_new, y_new, model_path, train_full=False):
        """
        Update (fine-tune) a SavedModel on new data.

        Args:
            X_new: array-like of shape (n_samples, n_features_new)
            y_new: array-like of shape (n_samples, ...)
            model_path: path to a TF SavedModel on disk
            train_full: if True, fine-tune the entire SavedModel backbone;
                        if False, only train the adapter layer mapping new->orig features.

        Returns:
            A tf.keras.Model instance after fine-tuning.
        """
        # (1) Load SavedModel and inspect original input dimension
        saved = tf.saved_model.load(model_path)
        if "serving_default" not in saved.signatures:
            raise ValueError(
                f"No 'serving_default' signature in {model_path!r}")
        sig = saved.signatures["serving_default"]
        orig_dim = sig.inputs[0].shape[-1]

        # (2) Prepare data
        X_arr = np.asarray(X_new)
        y_arr = np.asarray(y_new)
        new_dim = X_arr.shape[1]

        # (3) Build adapter + wrapped model
        inp = Input(shape=(new_dim,), name="model_input")
        x = inp
        if new_dim != orig_dim:
            # trainable adapter mapping new_dim -> orig_dim
            x = layers.Dense(
                orig_dim,
                activation=None,
                name="input_adapter"
            )(x)

        # Attach the SavedModel as a non-Keras layer
        out_dict = TFSMLayer(
            model_path,
            call_endpoint="serving_default"
        )(x)
        # unwrap single-output dict
        out = next(iter(out_dict.values())) if isinstance(
            out_dict, dict) else out_dict

        model = Model(inputs=inp, outputs=out, name="wrapped_saved_model")

        # (4) Initialize variables and patch optimizer hooks
        model(tf.zeros([1, new_dim]))
        for v in model.trainable_variables:
            # ensure regularizer attribute exists
            try:
                if not hasattr(v, 'regularizer'):
                    setattr(v, 'regularizer', None)
            except Exception:
                object.__setattr__(v, 'regularizer', None)
            # ensure overwrite_with_gradient flag for optimizer
            try:
                setattr(v, 'overwrite_with_gradient', True)
            except Exception:
                object.__setattr__(v, 'overwrite_with_gradient', True)

        # (5) Freeze backbone if not fine-tuning full model
        if not train_full:
            for layer in model.layers:
                if isinstance(layer, TFSMLayer):
                    try:
                        layer.trainable = False
                    except (AttributeError, ValueError):
                        object.__setattr__(layer, '_trainable', False)

        # (6) Compile & fit
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss="mse",
            metrics=["mae"]
        )
        model.fit(
            X_arr,
            y_arr,
            epochs=5,
            batch_size=16,
            verbose=1
        )

        return model


if __name__ == "__main__":

    model_path = r"VisQAI-base/package/model"  # adjust if needed
    print(f"Loading SavedModel from: {model_path}")
    loaded = tf.saved_model.load(model_path)
    sig = loaded.signatures["serving_default"]
    print("\nSignature Inputs:")
    for inp in sig.structured_input_signature[1].values():
        print(" ", inp.shape, inp.dtype)
    print("Signature Outputs:", sig.structured_outputs)

    class SMWrapper:
        def __init__(self, fn, output_key):
            self.fn = fn
            self.output_key = output_key

        def predict(self, x: np.ndarray) -> np.ndarray:
            tf_in = tf.constant(x)
            out_dict = self.fn(tf_in)
            return out_dict[self.output_key].numpy()

    out_key = next(iter(sig.structured_outputs.keys()))
    model = SMWrapper(sig, out_key)

    inp_spec = sig.structured_input_signature[1]
    feat_dim = list(inp_spec.values())[0].shape[-1]
    batch_size = 1
    X_mock = np.random.randn(batch_size, feat_dim).astype(np.float32)
    print(f"\nMock input batch shape: {X_mock.shape}")
    predictor = Predictor()
    predictor.set_model(model)
    preds = predictor.predict(X_mock)

    print(f"\npreds.shape = {preds.shape}")
    print("Sample preds:\n", preds[:2])
    assert preds.ndim == 2, f"Expected 2D array, got ndim={preds.ndim}"
    assert preds.shape[1] == 5, f"Expected 5 outputs, got {preds.shape[1]}"
    print("Success: predictor returned 5 outputs per sample.")
