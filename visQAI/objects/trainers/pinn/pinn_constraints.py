from abc import ABC, abstractmethod
from typing import Sequence, List, Optional
import tensorflow as tf


def _safe_gather_cols(tensor: tf.Tensor, indices: tf.Tensor) -> tf.Tensor:
    # Gather only valid indices from tensor along axis=1
    D = tf.shape(tensor)[1]
    valid = tf.boolean_mask(indices, indices < D)
    def no_valid(): return tf.zeros(
        [tf.shape(tensor)[0], 0], dtype=tensor.dtype)
    gathered = tf.cond(tf.size(valid) > 0,
                       lambda: tf.gather(tensor, valid, axis=1),
                       no_valid)
    return gathered


def _safe_slice_col(tensor: tf.Tensor, idx: int) -> tf.Tensor:
    # Safely slice column idx from tensor; return zeros if out-of-bounds
    D = tf.shape(tensor)[1]
    def slice_ok(): return tensor[:, idx]
    def slice_zero(): return tf.zeros_like(tensor[:, 0])
    return tf.cond(idx < D, slice_ok, slice_zero)


class BaseConstraint(ABC):
    def __init__(self, weight: float = 1.0):
        self.weight = weight

    @abstractmethod
    def __call__(
        self,
        model: tf.keras.Model,
        x: tf.Tensor,
        y_pred: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        ...


class MonotonicIncreasingConstraint(BaseConstraint):
    def __init__(self, feature_names: Sequence[str], to_enforce: Sequence[str], weight: float = 1.0):
        super().__init__(weight)
        fn = list(feature_names)
        missing = [f for f in to_enforce if f not in fn]
        if missing:
            raise ValueError(
                f"[MonotonicIncreasing] unknown features: {missing}")
        self._indices: List[int] = [fn.index(f) for f in to_enforce]
        self.indices_tensor = tf.constant(self._indices, dtype=tf.int32)

    def __call__(self, model, x, y_pred=None):
        x = tf.convert_to_tensor(x)
        if not self._indices:
            return tf.constant(0.0, dtype=x.dtype)

        with tf.GradientTape() as tape:
            tape.watch(x)
            u = model(x, training=True)
        du_dx = tape.gradient(u, x)

        selected = _safe_gather_cols(du_dx, self.indices_tensor)
        if tf.shape(selected)[1] == 0:
            return tf.constant(0.0, dtype=du_dx.dtype)

        violations = tf.nn.relu(-selected)
        loss = tf.reduce_mean(tf.square(violations))
        return self.weight * loss


class MonotonicDecreasingConstraint(BaseConstraint):
    def __init__(self, feature_names: Sequence[str], to_enforce: Sequence[str], weight: float = 1.0):
        super().__init__(weight)
        fn = list(feature_names)
        missing = [f for f in to_enforce if f not in fn]
        if missing:
            raise ValueError(
                f"[MonotonicDecreasing] unknown features: {missing}")
        self._indices: List[int] = [fn.index(f) for f in to_enforce]
        self.indices_tensor = tf.constant(self._indices, dtype=tf.int32)

    def __call__(self, model, x, y_pred=None):
        x = tf.convert_to_tensor(x)
        if not self._indices:
            return tf.constant(0.0, dtype=x.dtype)

        with tf.GradientTape() as tape:
            tape.watch(x)
            u = model(x, training=True)
        du_dx = tape.gradient(u, x)

        selected = _safe_gather_cols(du_dx, self.indices_tensor)
        if tf.shape(selected)[1] == 0:
            return tf.constant(0.0, dtype=du_dx.dtype)

        violations = tf.nn.relu(selected)
        loss = tf.reduce_mean(tf.square(violations))
        return self.weight * loss


class FlatSlopeConstraint(BaseConstraint):
    def __init__(self, feature_names: Sequence[str], feature: str, weight: float = 1.0):
        super().__init__(weight)
        if feature not in feature_names:
            raise ValueError(f"[FlatSlope] unknown feature: {feature}")
        self.idx = feature_names.index(feature)

    def __call__(self, model, x, y_pred=None):
        x = tf.convert_to_tensor(x)
        with tf.GradientTape() as tape:
            tape.watch(x)
            u = model(x, training=True)
        du_dx = tape.gradient(u, x)

        slope = _safe_slice_col(du_dx, self.idx)
        loss = tf.reduce_mean(tf.square(slope))
        return self.weight * loss


class ShearThinningConstraint(BaseConstraint):
    def __init__(self, weight: float = 1.0):
        super().__init__(weight)

    def __call__(self, model, x, y_pred=None):
        x = tf.convert_to_tensor(x)
        u = y_pred if y_pred is not None else model(x, training=True)

        num = tf.shape(u)[1]

        def _penalty():
            diffs = u[:, :-1] - u[:, 1:]
            violations = tf.nn.relu(-diffs)
            return tf.reduce_mean(tf.square(violations))
        loss = tf.cond(num > 1, _penalty,
                       lambda: tf.constant(0.0, dtype=u.dtype))
        return self.weight * loss


class ArrheniusConstraint(BaseConstraint):
    def __init__(self, feature_names: Sequence[str], T_feature: str, weight: float = 1.0):
        super().__init__(weight)
        if T_feature not in feature_names:
            raise ValueError(f"[Arrhenius] unknown T_feature: {T_feature}")
        self.idx_T = feature_names.index(T_feature)

    def __call__(self, model, x, y_pred=None):
        x = tf.convert_to_tensor(x)
        with tf.GradientTape() as tape:
            tape.watch(x)
            u = model(x, training=True)
        du_dx = tape.gradient(u, x)

        du_dT = _safe_slice_col(du_dx, self.idx_T)
        T = _safe_slice_col(x, self.idx_T)
        u_avg = tf.reduce_mean(u, axis=1)

        G = - T**2 * du_dT / (u_avg + 1e-8)
        mean_G = tf.reduce_mean(G)
        loss = tf.reduce_mean((G - mean_G)**2)
        return self.weight * loss


class GaussianBellAroundPIConstraint(BaseConstraint):
    def __init__(
        self,
        feature_names: Sequence[str],
        pH_feature: str,
        pI_feature: str,
        weight: float = 1.0
    ):
        super().__init__(weight)
        names = list(feature_names)
        missing = [f for f in (pH_feature, pI_feature) if f not in names]
        if missing:
            raise ValueError(f"[GaussianBell] unknown features: {missing}")
        self.idx_pH = names.index(pH_feature)
        self.idx_pI = names.index(pI_feature)

    def __call__(
        self,
        model: tf.keras.Model,
        x: tf.Tensor,
        y_pred: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        x = tf.convert_to_tensor(x)

        with tf.GradientTape() as tape1:
            tape1.watch(x)
            u_all = model(x, training=True)
            u_avg = tf.reduce_mean(u_all, axis=1)
        du_dx = tape1.gradient(u_avg, x)

        with tf.GradientTape() as tape2:
            tape2.watch(x)
            u_all2 = model(x, training=True)
            u_avg2 = tf.reduce_mean(u_all2, axis=1)
        hess = tape2.jacobian(u_avg2, x)
        diag_hess = tf.linalg.diag_part(hess)

        d2u_pH2 = _safe_slice_col(diag_hess, self.idx_pH)
        pH = _safe_slice_col(x, self.idx_pH)
        pI = _safe_slice_col(x, self.idx_pI)

        mask = tf.exp(-((pH - pI) ** 2) * 50.0)
        slope_pen = tf.square(_safe_slice_col(du_dx, self.idx_pH)) * mask
        concave_pen = tf.square(tf.nn.relu(d2u_pH2)) * mask
        loss = tf.reduce_mean(slope_pen + concave_pen)
        return self.weight * loss


class EinsteinDiluteLimitConstraint(BaseConstraint):
    def __init__(self, feature_names: Sequence[str], conc_feature: str, threshold: float = 0.05, weight: float = 1.0):
        super().__init__(weight)
        if conc_feature not in feature_names:
            raise ValueError(
                f"[EinsteinLimit] unknown conc_feature: {conc_feature}")
        self.idx = feature_names.index(conc_feature)
        self.thresh = threshold

    def __call__(self, model, x: tf.Tensor, y_pred=None) -> tf.Tensor:
        x = tf.convert_to_tensor(x)
        phi = _safe_slice_col(x, self.idx)
        u_all = model(x, training=True)
        u_avg = tf.reduce_mean(u_all, axis=1)

        limit = 1.0 + 2.5 * phi
        mask = tf.cast(phi < self.thresh, tf.float32)
        diff = (u_avg - limit) * mask
        loss = tf.reduce_sum(tf.square(diff)) / (tf.reduce_sum(mask) + 1e-6)
        return self.weight * loss


class ExcludedVolumeDivergenceConstraint(BaseConstraint):
    def __init__(self, feature_names: Sequence[str], conc_feature: str, weight: float = 1.0):
        super().__init__(weight)
        if conc_feature not in feature_names:
            raise ValueError(
                f"[ExclVolume] unknown conc_feature: {conc_feature}")
        self.idx = feature_names.index(conc_feature)

    def __call__(self, model, x, y_pred=None):
        x = tf.convert_to_tensor(x)
        u = y_pred if y_pred is not None else model(x, training=True)

        u_phi = _safe_slice_col(u, self.idx)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            du_dx = tape.gradient(u_phi, x)
            du_dphi = _safe_slice_col(du_dx, self.idx)
        d2u = tape.gradient(du_dphi, x)
        d2u_phi2 = _safe_slice_col(d2u, self.idx)
        violations = tf.nn.relu(-d2u_phi2)
        loss = tf.reduce_mean(tf.square(violations))
        return self.weight * loss
