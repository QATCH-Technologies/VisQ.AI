# losses.py

import tensorflow as tf
from typing import List, Tuple
from pinn_constraints import BaseConstraint


def composite_loss(
    model: tf.keras.Model,
    x: tf.Tensor,
    y_true: tf.Tensor,
    constraints: List[BaseConstraint],
    data_weight: float = 1.0,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    if x is None:
        raise ValueError("composite_loss: received x=None")
    x = tf.debugging.check_numerics(
        x, message="composite_loss: x contains NaN or Inf")
    y_pred = model(x, training=True)
    y_true = tf.cast(y_true, y_pred.dtype)
    data_loss = tf.reduce_mean(tf.square(y_pred - y_true))
    phys_losses = []
    for c in constraints:
        idxs = getattr(c, "_indices", None)
        if isinstance(idxs, list) and len(idxs) == 0:
            continue
        phys_losses.append(c(model, x))
    if phys_losses:
        phys_loss = tf.add_n(phys_losses)
    else:
        phys_loss = tf.constant(0.0, dtype=y_pred.dtype)
    total_loss = data_weight * data_loss + phys_loss
    return total_loss, data_loss, phys_loss
