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
    if constraints is None:
        constraints = getattr(model, "_physics_constraints", [])

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


class CompositeLossWithUncertainty:
    def __init__(self, n_phys: int):
        init = tf.zeros((n_phys + 1,), dtype=tf.float32)
        self.log_sigmas = tf.Variable(init, trainable=True, name="log_sigmas")

    def __call__(self,
                 y_true:    tf.Tensor,
                 y_pred:    tf.Tensor,
                 model:     tf.keras.Model,
                 x:         tf.Tensor,
                 constraints: List[BaseConstraint],
                 ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # 1) data term
        data_loss = tf.reduce_mean(
            tf.square(y_pred - tf.cast(y_true, y_pred.dtype)))

        # 2) physics terms
        phys_losses = []
        for c in constraints:
            phys_losses.append(c(model, x))
        phys_sum = tf.add_n(phys_losses) if phys_losses else tf.constant(
            0.0, dtype=y_pred.dtype)

        # 3) uncertainty weighting
        all_losses = [data_loss] + phys_losses
        total = 0.0
        for i, L in enumerate(all_losses):
            sigma_sq = tf.exp(self.log_sigmas[i])  # σ²
            total += L / (2.0 * sigma_sq) + 0.5 * tf.math.log(sigma_sq)

        return total, data_loss, phys_sum
