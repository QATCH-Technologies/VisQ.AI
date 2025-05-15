
from keras import layers, models, optimizers, losses, metrics, backend as K
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="custom_layers")
class ReverseCumsum(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        y_rev = inputs[:, ::-1]
        csum_rev = K.cumsum(y_rev, axis=1)
        return csum_rev[:, ::-1]

    def get_config(self):
        # no extra args, so just delegate
        return super().get_config()
