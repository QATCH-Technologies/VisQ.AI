import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sample_collector import SampleCollector
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import math
from scipy.optimize import minimize
from scipy.stats import gaussian_kde
from sklearn.manifold import TSNE


def optimizer_fn(obj_func, initial_theta, bounds):
    def func(theta):
        loss, _ = obj_func(theta)
        return loss

    def grad(theta):
        _, gradient = obj_func(theta)
        return gradient

    res = minimize(
        fun=func,
        x0=initial_theta,
        jac=grad,
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": 10000}
    )
    return res.x, res.fun


class ExperimentSuggester:
    def __init__(self, collector, target_cols, kappa=2.0, n_restarts_optimizer=20, eval_frac_window=0.1):
        self.collector = collector
        self.target_cols = target_cols
        self.kappa = kappa
        self.eval_frac_window = eval_frac_window

        # Immutable inputs
        self.fixed_features = {
            'Protein_type', 'MW', 'PI_mean', 'PI_range', 'Buffer_pH'
        }

        # All possible feature names (cats + nums) minus IDs
        all_feats = set(collector.categorical_features +
                        collector.numeric_features)
        # Remove the viscosity targets from consideration altogether
        viscosity_feats = {c for c in all_feats if c.startswith("Viscosity_")}
        # The ones we CAN vary:
        self.mutable_features = list(
            all_feats - self.fixed_features - viscosity_feats)

        # Split numeric vs categorical mutables
        self.mutable_numeric = [
            f for f in self.mutable_features if f in collector.numeric_features]
        self.mutable_categorical = [
            f for f in self.mutable_features if f in collector.categorical_features]

        # 1) Build a broader‐bounds kernel:
        kernel = (
            C(1.0, (1e-3, 1e4))    # signal variance from 1e-3 to 1e4
            * RBF(1.0, (1e-3, 1e4))  # length‐scale from 1e-3 to 1e4
            + WhiteKernel(1e-5, (1e-12, 1.0))  # noise from 1e-12 to 1
        )

        # 2) Prepare scaler for X
        self.scaler_X = StandardScaler()

        # 4) Instantiate the GP with scaling and restarts
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=20,
            optimizer=optimizer_fn
        )

        # Placeholders for baseline eval grid
        self._eval_grid = None
        self._baseline_unc_sum = None

    def _fit_model(self):
        # Encode & scale X
        X_df = self.collector.encode_dataframe(self.known_df)
        X_scaled = self.scaler_X.fit_transform(X_df.values)

        # Build scalar target
        y = self.known_df[self.target_cols].mean(axis=1).values

        # Fit GP (now with proper scaling & hyperparameter search)
        self.gp.fit(X_scaled, y)

    def compute_baseline(self, initial_known_df):
        """
        Call this ONCE with your original CSV-loaded DataFrame
        before you enter your loop.  It:
          - builds the full eval grid around your initial sample
          - fits the GP on that initial data
          - computes & stores the baseline total σ
        """
        self.known_df = initial_known_df.copy()
        # fit GP on initial data
        self._fit_model()

        # build eval grid from the first sample in that DF
        first_sample = self.collector.collect(
            self.known_df.iloc[0].to_dict()
        ).iloc[0]
        self._eval_grid = self._create_eval_grid(first_sample)

        # compute baseline σ sum
        X_eval_scaled = self._scale_df(self._eval_grid)
        _, sigma_eval_base = self.gp.predict(X_eval_scaled, return_std=True)
        self._baseline_unc_sum = float(np.sum(sigma_eval_base))

    def update_known(self, new_known_df):
        """
        After each iteration, call this to refit the GP on your
        growing dataset.  It will preserve the baseline.
        """
        self.known_df = new_known_df.copy()
        self._fit_model()

    def _scale_df(self, df):
        """Helper to encode & scale any DataFrame of features."""
        X = self.collector.encode_dataframe(df).values
        return self.scaler_X.transform(X)

    def compute_update_vector(
        self,
        current: pd.Series,
        suggestion: pd.Series,
        info_score: float
    ) -> dict:
        """
        Only compute deltas for numeric mutables so you never try to float() a string.
        """
        step_scale = 1 - info_score
        update = {}
        for f in self.mutable_numeric:
            raw_delta = float(suggestion[f]) - float(current[f])
            scaled_delta = raw_delta * step_scale
            direction = np.sign(raw_delta)  # +1, -1 or 0
            update[f] = {
                "delta":     scaled_delta,
                "direction": int(direction)
            }
        return update

    def plan_next_sample(
        self,
        current: pd.Series,
        suggestion: pd.Series,
        info_score: float
    ) -> dict:
        update_vec = self.compute_update_vector(
            current, suggestion, info_score)
        next_vals = {}
        for f, params in update_vec.items():
            # Sanity check
            if f not in self.mutable_numeric:
                raise ValueError(f"Unexpected feature in update vector: {f}")

            raw = float(current[f]) + params["delta"]

            if f == "Protein_concentration":
                val = int(round(raw))
                val = max(val, 0)

            elif f == "Sugar_concentration":
                val = round(raw / 0.1) * 0.1
                val = min(max(val, 0.0), 1.0)

            elif f == "Surfactant_concentration":
                val = round(raw / 0.01) * 0.01
                val = min(max(val, 0.0), 0.1)

            else:
                val = raw

            next_vals[f] = val

        return next_vals

    def _create_eval_grid(self, sample):
        base = sample.to_dict()
        p0 = sample['Protein_concentration']
        lo = math.floor(p0 * (1 - self.eval_frac_window))
        hi = math.ceil(p0 * (1 + self.eval_frac_window))
        prot_vals = list(range(lo, hi + 1))
        sugar_vals = [i * 0.1 for i in range(11)]
        surf_vals = [i * 0.01 for i in range(11)]

        rows = []
        cols = self.collector.categorical_features + self.collector.numeric_features
        for p in prot_vals:
            for s in sugar_vals:
                for sf in surf_vals:
                    cand = base.copy()
                    cand['Protein_concentration'] = p
                    cand['Sugar_concentration'] = s
                    cand['Surfactant_concentration'] = sf
                    rows.append(cand)
        return pd.DataFrame(rows, columns=cols)

    def _generate_candidates(self, sample: pd.Series, n_candidates: int, frac_window: float) -> pd.DataFrame:
        """
        Vary *any* mutable feature:
          - Numeric mutables: uniform in ±frac_window window
          - Categorical mutables: random known category
        Fixed_features remain at their original values.
        """
        base = sample.to_dict()
        cols = self.collector.categorical_features + self.collector.numeric_features
        rows = []

        # Precompute numeric windows
        windows = {}
        for f in self.mutable_numeric:
            v = sample[f]
            lo, hi = v * (1-frac_window), v * (1+frac_window)
            windows[f] = (lo, hi)

        # Precompute category lists
        cat_lists = {f: list(self.collector._cat_maps[f].keys())
                     for f in self.mutable_categorical}

        for _ in range(n_candidates):
            cand = base.copy()
            # numeric jitter
            for f in self.mutable_numeric:
                lo, hi = windows[f]
                cand[f] = float(np.random.uniform(lo, hi))
            # categorical resample
            for f in self.mutable_categorical:
                cand[f] = np.random.choice(cat_lists[f])
            rows.append(cand)

        return pd.DataFrame(rows, columns=cols)

    def suggest(self, raw_sample, n_candidates=200, frac_window=0.1):

        # encode sample
        sample_df = self.collector.collect(raw_sample)
        sample = sample_df.iloc[0]

        # generate & predict on candidates
        cand_df = self._generate_candidates(sample, n_candidates, frac_window)
        X_cand_scaled = self._scale_df(cand_df)
        mu, sigma = self.gp.predict(X_cand_scaled, return_std=True)
        ucb = mu + self.kappa * sigma
        best_idx = int(np.argmax(ucb))
        suggestion = cand_df.iloc[best_idx][self.mutable_features].copy()

        # recompute total uncertainty on the fixed eval grid
        X_eval_scaled = self._scale_df(self._eval_grid)
        _, sigma_eval = self.gp.predict(X_eval_scaled, return_std=True)
        curr_unc_sum = float(np.sum(sigma_eval))

        # monotonic 0–1 score
        info_score = 1 - (curr_unc_sum / self._baseline_unc_sum)
        info_score = float(np.clip(info_score, 0, 1))

        # suggestion['info_score'] = info_score
        return suggestion, info_score


def visualize_density_surface_new(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    collector,
    grid_size: int = 50
):
    """
    Create a 3D density surface over the first two PCA components,
    showing 'before' space and overlaying only the *new* 'after' points.
    """
    # 1) Identify new points: rows in after_df not in before_df
    merged = after_df.merge(
        before_df.drop_duplicates(),
        how='left',
        indicator=True
    )
    new_df = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])

    # 2) Encode inputs (drop viscosity targets) for before + new
    def encode_inputs(df):
        X_enc = collector.encode_dataframe(df)
        cols = [c for c in X_enc.columns if not c.startswith("Viscosity_")]
        return X_enc[cols].values

    X_bef = encode_inputs(before_df)
    X_new = encode_inputs(new_df)
    X_all = np.vstack([X_bef, X_new])

    # 3) PCA reduction to 2D for surface axes
    pca = PCA(n_components=2)
    comps_all = pca.fit_transform(X_all)
    n_bef = X_bef.shape[0]
    comps_bef = comps_all[:n_bef]
    comps_new = comps_all[n_bef:]

    # 4) KDE on combined 2D embedding
    kde = gaussian_kde(comps_all.T)

    # 5) Build grid over PCA space
    x_min, x_max = comps_all[:, 0].min(), comps_all[:, 0].max()
    y_min, y_max = comps_all[:, 1].min(), comps_all[:, 1].max()
    xi = np.linspace(x_min, x_max, grid_size)
    yi = np.linspace(y_min, y_max, grid_size)
    Xg, Yg = np.meshgrid(xi, yi)
    grid_coords = np.vstack([Xg.ravel(), Yg.ravel()])
    Zg = kde(grid_coords).reshape(Xg.shape)

    # 6) Plot surface + overlay
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Xg, Yg, Zg, cmap='viridis', alpha=0.6)
    # Known before
    zb = kde(comps_bef.T)
    ax.scatter(comps_bef[:, 0], comps_bef[:, 1], zb,
               c='blue', marker='o', label='Before')
    # Only new after
    zn = kde(comps_new.T)
    ax.scatter(comps_new[:, 0], comps_new[:, 1], zn,
               c='red', marker='^', s=100, label='New')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('Density')
    ax.legend()
    plt.tight_layout()
    plt.show()


def visualize_density_tsne(before_df: pd.DataFrame, after_df: pd.DataFrame, collector, grid_size=50, perplexity=30):
    """
    Create a 3D surface of sample-density over a 2D t-SNE embedding,
    and overlay only the 'before' and *new* 'after' sample points (no duplicates).
    """
    # 1) Identify new points: rows in after_df not present in before_df
    merged = after_df.merge(
        before_df.drop_duplicates(),
        how='left',
        indicator=True
    )
    new_df = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])

    # 2) Encode inputs (drop viscosity targets)
    def encode_inputs(df):
        X_enc = collector.encode_dataframe(df)
        cols = [c for c in X_enc.columns if not c.startswith("Viscosity_")]
        return X_enc[cols].values

    X_bef = encode_inputs(before_df)
    X_new = encode_inputs(new_df)
    X_all = np.vstack([X_bef, X_new])

    # 3) t-SNE to 2D on combined data
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    comps_all = tsne.fit_transform(X_all)
    n_bef = X_bef.shape[0]
    comps_bef = comps_all[:n_bef]
    comps_new = comps_all[n_bef:]

    # 4) Fit KDE on the combined 2D embedding
    kde = gaussian_kde(comps_all.T)

    # 5) Create a grid over the t-SNE space
    x_min, x_max = comps_all[:, 0].min(), comps_all[:, 0].max()
    y_min, y_max = comps_all[:, 1].min(), comps_all[:, 1].max()
    xi = np.linspace(x_min, x_max, grid_size)
    yi = np.linspace(y_min, y_max, grid_size)
    Xg, Yg = np.meshgrid(xi, yi)
    grid_coords = np.vstack([Xg.ravel(), Yg.ravel()])
    Zg = kde(grid_coords).reshape(Xg.shape)

    # 6) Plot surface + overlay known & new points
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Xg, Yg, Zg, cmap='viridis', alpha=0.6)
    zb = kde(comps_bef.T)
    ax.scatter(comps_bef[:, 0], comps_bef[:, 1], zb,
               c='blue', marker='o', label='Known')
    zn = kde(comps_new.T)
    ax.scatter(comps_new[:, 0], comps_new[:, 1],
               zn, c='red', marker='^', label='New')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_zlabel('Density')
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 1) Load history & set up
    initial_df = pd.read_csv("content/formulation_data_05272025.csv")
    target_cols = [
        "Viscosity_100", "Viscosity_1000",
        "Viscosity_10000", "Viscosity_100000",
        "Viscosity_15000000"
    ]
    known_before = initial_df.copy()

    # 2) Seed sample (includes initial viscosities)
    sample = {
        "ID": "exp_1",
        "Protein_type": "EXP",
        "Buffer_type": "PBS",
        "Sugar_type": "Sucrose",
        "Surfactant_type": "tween-20",
        "MW": 100,
        "PI_mean": 5,
        "PI_range": 1,
        "Protein_concentration": 100,
        "Temperature": 25,
        "Buffer_pH": 7.4,
        "Buffer_conc": 10,
        "NaCl": 140,
        "Sugar_concentration": 0.1,
        "Surfactant_concentration": 0.01,
        "Viscosity_100": 10,
        "Viscosity_1000": 9,
        "Viscosity_10000": 9,
        "Viscosity_100000": 9,
        "Viscosity_15000000": 8
    }
    # 1) load initial data
    initial_df = pd.read_csv("content/formulation_data_05272025.csv")
    collector = SampleCollector(initial_df)

    # 2) instantiate + compute baseline once
    suggester = ExperimentSuggester(collector, target_cols)
    suggester.compute_baseline(initial_df)

    known_df = initial_df.copy()
    n_iters = int(input("How many automated experiments to run? "))
    history = []
    for i in range(n_iters):
        # a) suggest
        suggestion, info_score = suggester.suggest(sample)
        print(f"Iteration {i+1}, info_score = {info_score:.3f}",
              suggestion.to_dict())
        current_series = pd.Series(sample)
        history.append(suggestion.to_dict())

        suggestion_series = pd.Series(suggestion.to_dict())
        info = info_score
        updates = suggester.compute_update_vector(
            current_series, suggestion_series, info
        )
        print("Update vector:", updates)

        # plan the next sample values
        next_concs = suggester.plan_next_sample(
            current_series, suggestion_series, info
        )
        print("Next concentrations:", next_concs)

        start = sample['Viscosity_100']
        end = sample['Viscosity_15000000']
        measured_vals = np.linspace(start, end, num=len(target_cols))
        measured = dict(zip(target_cols, measured_vals))

        # Now update your sample with the suggestion and these monotonic measurements:
        sample.update(suggestion.to_dict())
        sample.update(measured)

        # c) append + update GP (but never reset baseline)
        known_df = pd.concat(
            [known_df, pd.DataFrame([sample])], ignore_index=True)
        suggester.update_known(known_df)
    known_after = known_df

    history_df = pd.DataFrame(history)

    # Create 3 separate subplots in one figure
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(6, 9))
    for ax, col in zip(axes, history_df.columns):
        ax.plot(history_df.index + 1, history_df[col], marker='o')
        ax.set_ylabel(col)
        ax.set_title(f'{col} over Iterations')
    axes[-1].set_xlabel('Iteration')
    plt.tight_layout()
    plt.show()

    visualize_density_surface_new(known_before, known_after, collector)
    visualize_density_tsne(known_before, known_after, collector, perplexity=40)

    print(f"\nDone! Collected {len(known_df)} total experiments.")
