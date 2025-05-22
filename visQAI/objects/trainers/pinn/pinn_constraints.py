# physics_constraints.py

from abc import ABC, abstractmethod
from typing import Sequence, List, Optional
import tensorflow as tf


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


# ──────────────────────────────────────────────────────────────────────────────
class MonotonicIncreasingConstraint(BaseConstraint):
    """
    A TensorFlow constraint that penalizes a model whenever its output decreases
    with respect to one or more input features.

    This constraint computes the gradient of the model's prediction with respect to
    its inputs, selects the components corresponding to the specified features,
    and adds to the loss any negative slopes (i.e. where dy/dx < 0).  In effect, it
    enforces that the model's output be monotonically non-decreasing along each
    of the "to_enforce" features.

    Args:
        feature_names (Sequence[str]):
            The full, ordered list of input feature names that the model expects.
        to_enforce (Sequence[str]):
            A subset of `feature_names` for which monotonicity should be enforced.
            For each feature in this list, the constraint will penalize negative
            partial derivatives.
        weight (float, optional):
            A scaling factor for the penalty.  Defaults to 1.0.  The returned loss
            is multiplied by this weight.

    Raises:
        ValueError:
            If any feature in `to_enforce` is not found in `feature_names`.

    Attributes:
        _indices (List[int]):
            The integer indices of `to_enforce` features within `feature_names`.
        indices_tensor (tf.Tensor):
            A 1D int32 Tensor containing `_indices`, used to gather the relevant
            gradient components in the compute graph.
    """

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
        selected = tf.gather(du_dx, self.indices_tensor, axis=1)
        violations = tf.nn.relu(-selected)
        loss = tf.reduce_mean(tf.square(violations))
        return self.weight * loss


# ──────────────────────────────────────────────────────────────────────────────
class MonotonicDecreasingConstraint(BaseConstraint):
    """
    A TensorFlow constraint that penalizes a model whenever its output increases
    with respect to one or more input features.

    This constraint computes the gradient of the model's prediction with respect to
    its inputs, selects the components corresponding to the specified features,
    and adds to the loss any positive slopes (i.e. where dy/dx > 0).  In effect, it
    enforces that the model's output be monotonically non-increasing along each
    of the `to_enforce` features.

    Args:
        feature_names (Sequence[str]):
            The full, ordered list of input feature names that the model expects.
        to_enforce (Sequence[str]):
            A subset of `feature_names` for which monotonic decrease should be enforced.
            For each feature in this list, the constraint will penalize positive
            partial derivatives.
        weight (float, optional):
            A scaling factor for the penalty.  Defaults to 1.0.  The returned loss
            is multiplied by this weight.

    Raises:
        ValueError:
            If any feature in `to_enforce` is not found in `feature_names`.

    Attributes:
        _indices (List[int]):
            The integer indices of `to_enforce` features within `feature_names`.
        indices_tensor (tf.Tensor):
            A 1D int32 Tensor containing `_indices`, used to gather the relevant
            gradient components in the compute graph.
    """

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

        selected = tf.gather(du_dx, self.indices_tensor, axis=1)
        violations = tf.nn.relu(selected)
        loss = tf.reduce_mean(tf.square(violations))
        return self.weight * loss


# ──────────────────────────────────────────────────────────────────────────────
class FlatSlopeConstraint(BaseConstraint):
    """
    A TensorFlow constraint that penalizes any non'zero slope of the models output
    with respect to a single input feature, enforcing that the output be flat
    (i.e. zero gradient) along that feature.

    Args:
        feature_names (Sequence[str]):
            The full, ordered list of input feature names that the model expects.
        feature (str):
            The name of the one feature for which the model's partial derivative
            should be driven toward zero.
        weight (float, optional):
            A scaling factor for the penalty. Defaults to 1.0. The final loss
            contribution is multiplied by this weight.

    Raises:
        ValueError:
            If `feature` is not found in `feature_names`.

    Attributes:
        idx (int):
            The index of `feature` in `feature_names`, used to select the
            correct partial derivative.
    """

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
        slope = du_dx[:, self.idx]
        loss = tf.reduce_mean(tf.square(slope))
        return self.weight * loss

# ──────────────────────────────────────────────────────────────────────────────


class ShearThinningConstraint(BaseConstraint):
    """
    A TensorFlow constraint that penalizes a model whenever its predicted viscosity
    does not decrease with increasing shear rate (i.e. enforces shear‐thinning behavior).

    In many complex fluids (e.g., polymer solutions, protein suspensions), viscosity
    drops as shear rate increases.  This constraint computes adjacent differences
    between the model's multiple viscosity outputs (ordered by ascending shear rate)
    and penalizes any cases where viscosity would increase instead.

    Args:
        weight (float, optional):
            A scaling factor for the penalty. Defaults to 1.0. The returned loss
            is multiplied by this weight.

    Attributes:
        None beyond those inherited from BaseConstraint.
    """

    def __init__(self, weight: float = 1.0):
        super().__init__(weight)

    def __call__(self, model, x, y_pred=None):
        """
        Compute the shear-thinning penalty for a batch of inputs.

        Args:
            model: A callable that maps inputs `x` to a tensor `u` of shape
                (batch_size, num_shear_rates), representing predicted viscosities
                at increasing shear rates.
            x (tf.Tensor): The input tensor of shape (batch_size, num_features).
            y_pred (tf.Tensor, optional):
                If provided, uses these precomputed predictions instead of calling
                `model(x, training=True)`. Must be shape (batch_size, num_shear_rates).

        Returns:
            tf.Tensor: A scalar tensor equal to `weight * mean(violation^2)` where
            `violation` is any case where u_i < u_{i+1} (viscosity increases with shear).
        Method:
            1. Obtain predictions `u` (either via `y_pred` or `model(x, training=True)`).
            2. Compute adjacent differences: diffs = u[:, :-1] - u[:, 1:].
               Positive diffs mean viscosity_i ≥ viscosity_{i+1}; negative diffs
               indicate an increase in viscosity with shear (violation).
            3. Apply `tf.nn.relu(-diffs)` to isolate only negative diffs (violations).
            4. Square, average over all entries, and multiply by `weight`.
        """
        x = tf.convert_to_tensor(x)
        u = y_pred if y_pred is not None else model(x, training=True)
        diffs = u[:, :-1] - u[:, 1:]
        violations = tf.nn.relu(-diffs)
        loss = tf.reduce_mean(tf.square(violations))
        return self.weight * loss


# ──────────────────────────────────────────────────────────────────────────────
class ArrheniusConstraint(BaseConstraint):
    """
    A TensorFlow constraint that enforces an Arrhenius-type temperature dependence
    on a predicted property (e.g., viscosity).  In Arrhenius behavior, the log of
    the property varies linearly with inverse temperature, so
    G = -T^2/u ⋅ ∂u/∂T should be constant.

    This constraint computes G for each sample in the batch,
    then penalizes its variance so that G -> const across temperatures.

    Args:
        feature_names (Sequence[str]):
            The full, ordered list of input feature names that the model expects.
        T_feature (str):
            The name of the temperature feature in `feature_names`.
        weight (float, optional):
            A scaling factor for the penalty.  Defaults to 1.0.

    Raises:
        ValueError:
            If `T_feature` is not found in `feature_names`.

    Attributes:
        idx_T (int):
            The index of the temperature feature within `feature_names`.
    """

    def __init__(self, feature_names: Sequence[str], T_feature: str, weight: float = 1.0):
        super().__init__(weight)
        if T_feature not in feature_names:
            raise ValueError(f"[Arrhenius] unknown T_feature: {T_feature}")
        self.idx_T = feature_names.index(T_feature)

    def __call__(self, model, x, y_pred=None):
        """
        Compute the Arrhenius penalty for a batch of inputs.

        Args:
            model: A callable that maps inputs `x` to predictions `u`.
            x (tf.Tensor): Input tensor of shape (batch_size, num_features).
            y_pred (tf.Tensor, optional):
                If provided, use these precomputed predictions instead of
                calling `model(x, training=True)`.

        Returns:
            tf.Tensor:
                A scalar tensor equal to `weight * Var(G)`, where
                G = -T^2/u ⋅ du/dT and Var(G) is its variance over the batch.

        Method:
            1. Compute predictions `u` via `y_pred` or `model(x, training=True)`.
            2. Record `x` for gradient computation; use GradientTape to get
               du/dx, then extract ∂u/∂T at index `idx_T`.
            3. Extract T from `x[:, idx_T]`.
            4. Form G = -T^2 * (du/dT) / (u + epsilon) with epsilon=1e-8 for numerical stability.
            5. Compute the mean of G over the batch, then compute
               Var(G) = mean((G - mean(G))^2).
            6. Multiply by `weight` to produce the final penalty.
        """
        x = tf.convert_to_tensor(x)
        with tf.GradientTape() as tape:
            tape.watch(x)
            u = model(x, training=True)
        du_dx = tape.gradient(u, x)

        T = x[:, self.idx_T]
        du_dT = du_dx[:, self.idx_T]
        T_exp = tf.expand_dims(T,    axis=1)
        du_dT_exp = tf.expand_dims(du_dT, axis=1)

        G = - T_exp**2 * du_dT_exp / (u + 1e-8)
        mean_G = tf.reduce_mean(G)
        loss = tf.reduce_mean((G - mean_G)**2)
        return self.weight * loss


# ──────────────────────────────────────────────────────────────────────────────
class GaussianBellAroundPIConstraint(BaseConstraint):
    """
    Enforces a Gaussian-bell shaped viscosity peak around the protein's isoelectric point (pI).

    When predicting viscosity as a function of pH, proteins often show a pronounced peak at
    their isoelectric point: net charge is zero, intermolecular repulsion is minimized, and
    aggregation (hence viscosity) is maximized. This constraint nudges the model so that, in a
    narrow Gaussian window around pI:

      1. The first derivative of the predicted average viscosity w.r.t. pH is zero
         (i.e. a flat extremum at the peak).
      2. The second derivative w.r.t. pH is negative (concave-down, enforcing a "peak"
         rather than a trough).

    Outside this window, the constraint decays exponentially and has virtually no effect.

    Args:
        feature_names (Sequence[str]):
            List of all input feature names in the same order as your model expects.
        pH_feature (str):
            Name of the pH feature in `feature_names`.
        pI_feature (str):
            Name of the isoelectric point feature in `feature_names`.
        weight (float, optional):
            Multiplier for scaling this constraint term in the total loss. Defaults to 1.0.

    Raises:
        ValueError:
            If either `pH_feature` or `pI_feature` is not present in `feature_names`.

    Call Signature:
        (model: tf.keras.Model, x: tf.Tensor, y_pred: Optional[tf.Tensor] = None) -> tf.Tensor

    Returns:
        tf.Tensor:
            A scalar tensor giving the mean penalty over the batch:

              loss = mean[ 
                  (du/dpH)^2 * exp(-50*(pH - pI)^2)
                + (max(0, d^2u/dpH^2))^2 * exp(-50*(pH - pI)^2)
              ]

            Multiplying by `weight` scales how strongly this prior is enforced.
    """

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
            u_all = model(x, training=True)
            u_avg = tf.reduce_mean(u_all, axis=1)
        hess = tape2.jacobian(u_avg, x)
        diag_hess = tf.linalg.diag_part(hess)
        d2u_pH2 = diag_hess[:, self.idx_pH]
        pH = x[:, self.idx_pH]
        pI = x[:, self.idx_pI]
        mask = tf.exp(-((pH - pI) ** 2) * 50.0)
        slope_pen = tf.square(du_dx[:, self.idx_pH]) * mask
        concave_pen = tf.square(tf.nn.relu(d2u_pH2)) * mask
        loss = tf.reduce_mean(slope_pen + concave_pen)
        return self.weight * loss

# ──────────────────────────────────────────────────────────────────────────────


class EinsteinDiluteLimitConstraint(BaseConstraint):
    """
    Enforces the classical Einstein dilute-suspension limit: as the particle volume
    fraction phi->0, the relative viscosity u approaches u = 1 + 2.5·phi.

    In the dilute limit, interactions between suspended particles are negligible and
    hydrodynamic theory predicts a linear increase of viscosity with concentration:
      u(phi) = n/n_o ~ 1 + 2.5 phi.
    This constraint penalizes deviations from that line for phi below a small threshold.

    Args:
        feature_names (Sequence[str]):
            List of all input feature names in the same order as your model expects.
        conc_feature (str):
            Name of the concentration feature (phi) in `feature_names`.
        threshold (float, optional):
            Maximum phi below which to enforce the dilute limit (default: 0.05).
        weight (float, optional):
            Multiplier for scaling this constraint term in the total loss (default: 1.0).

    Raises:
        ValueError:
            If `conc_feature` is not found in `feature_names`.

    Call Signature:
        (model: tf.keras.Model, x: tf.Tensor, y_pred: Optional[tf.Tensor] = None) -> tf.Tensor

    Returns:
        tf.Tensor:
            A scalar tensor giving the mean squared deviation from the Einstein relation
            over all samples with φ < threshold:

            Multiplying by `weight` scales how strongly this physical prior is enforced.
    """

    def __init__(self, feature_names: Sequence[str], conc_feature: str, threshold: float = 0.05, weight: float = 1.0):
        super().__init__(weight)
        if conc_feature not in feature_names:
            raise ValueError(
                f"[EinsteinLimit] unknown conc_feature: {conc_feature}")
        self.idx = feature_names.index(conc_feature)
        self.thresh = threshold

    def __call__(self, model: tf.keras.Model, x: tf.Tensor, y_pred=None) -> tf.Tensor:
        x = tf.convert_to_tensor(x)
        u_all = model(x, training=True)
        u_avg = tf.reduce_mean(u_all, axis=1)
        phi = x[:, self.idx]
        limit = 1.0 + 2.5 * phi
        mask = tf.cast(phi < self.thresh, tf.float32)
        diff = (u_avg - limit) * mask
        loss = tf.reduce_sum(tf.square(diff)) / (tf.reduce_sum(mask) + 1e-6)
        return self.weight * loss


# ──────────────────────────────────────────────────────────────────────────────
class ExcludedVolumeDivergenceConstraint(BaseConstraint):
    """
    Enforces convex divergence of viscosity at high particle concentrations.

    As the volume fraction phi approaches its maximum packing limit, hydrodynamic
    and crowding effects cause viscosity to rise sharply and with positive curvature.
    Mathematically, we require:

      ddu(phi) / dphi^2 ≥ 0

    across the batch.  Any region where the second derivative goes negative (concave
    down) is penalized.

    Args:
        feature_names (Sequence[str]):
            List of all input feature names in the same order your model expects.
        conc_feature (str):
            Name of the concentration feature (phi) in `feature_names`.
        weight (float, optional):
            Scalar weight to scale this constraint term in the total loss.
            Defaults to 1.0.

    Raises:
        ValueError:
            If `conc_feature` is not found in `feature_names`.

    Call Signature:
        (model: tf.keras.Model, x: tf.Tensor, y_pred: Optional[tf.Tensor] = None) -> tf.Tensor

    Returns:
        tf.Tensor:
            A scalar tensor representing the mean squared penalty for any
            violation of convexity in phi:

              loss = mean[ (max(0, - ddu/dphi^2))^2 ]

            Multiplying by `weight` controls the strength of enforcement.
    """

    def __init__(self, feature_names: Sequence[str], conc_feature: str, weight: float = 1.0):
        super().__init__(weight)
        if conc_feature not in feature_names:
            raise ValueError(
                f"[ExclVolume] unknown conc_feature: {conc_feature}")
        self.idx = feature_names.index(conc_feature)

    def __call__(self, model, x, y_pred=None):
        # one tape, persistent so we can do two .gradient() calls
        x = tf.convert_to_tensor(x)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            u = y_pred if y_pred is not None else model(x, training=True)
            u_phi = u[:, self.idx]
            du_dx = tape.gradient(u_phi, x)
            du_dphi = du_dx[:, self.idx]
        d2u = tape.gradient(du_dphi, x)
        d2u_phi2 = d2u[:, self.idx]
        violations = tf.nn.relu(-d2u_phi2)
        loss = tf.reduce_mean(tf.square(violations))
        del tape
        return self.weight * loss
