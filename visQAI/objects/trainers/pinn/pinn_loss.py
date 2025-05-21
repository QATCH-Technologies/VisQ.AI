import tensorflow as tf
from typing import List, Tuple
from pinn_constraints import BaseConstraint


def composite_loss(
    model: tf.keras.Model,
    x: tf.Tensor,
    y_true: tf.Tensor,
    constraints: List[BaseConstraint],
    data_weight: float = 1.0,
    eps: float = 1e-8
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Returns (total_loss, data_loss, phys_loss) with phys_loss scaled so that
    data_weight * data_loss ≈ scaled_phys_loss each batch.
    """
    y_pred = model(x, training=True)
    y_true = tf.cast(y_true, y_pred.dtype)
    data_loss = tf.reduce_mean(tf.square(y_pred - y_true))

    phys_terms = []
    for c in constraints or getattr(model, "_physics_constraints", []):
        idxs = getattr(c, "_indices", None)
        if isinstance(idxs, list) and len(idxs) == 0:
            continue
        phys_terms.append(c(model, x))
    phys_loss = tf.add_n(phys_terms) if phys_terms else tf.constant(
        0.0, dtype=y_pred.dtype)
    w_phys = (data_weight * data_loss) / (phys_loss + eps)
    total_loss = data_weight * data_loss + w_phys * phys_loss

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
        data_loss = tf.reduce_mean(
            tf.square(y_pred - tf.cast(y_true, y_pred.dtype)))
        phys_losses = []
        for c in constraints:
            phys_losses.append(c(model, x))
        phys_sum = tf.add_n(phys_losses) if phys_losses else tf.constant(
            0.0, dtype=y_pred.dtype)
        all_losses = [data_loss] + phys_losses
        total = 0.0
        for i, L in enumerate(all_losses):
            sigma_sq = tf.exp(self.log_sigmas[i])  # σ²
            total += L / (2.0 * sigma_sq) + 0.5 * tf.math.log(sigma_sq)

        return total, data_loss, phys_sum
