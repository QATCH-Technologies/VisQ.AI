# eda_checks.py

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from pinn_domain import DataLoader
import os


def violation_rate_monotonic(x: np.ndarray, y: np.ndarray, increasing: bool = True) -> float:
    order = np.argsort(x)
    diffs = np.diff(y[order])
    return np.mean(diffs < 0) if increasing else np.mean(diffs > 0)


def mean_abs_slope(x: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    dx = np.diff(xs)
    dy = np.diff(ys)
    mask = dx != 0
    return np.mean(np.abs(dy[mask] / dx[mask])) if mask.any() else np.nan


def shear_thinning_rate(U: np.ndarray) -> float:
    # fraction of k where U[:,k] < U[:,k+1]
    diffs = np.diff(U, axis=1)
    return np.mean(diffs > 0)


def arrhenius_stats(T: np.ndarray, U: np.ndarray):
    invT = 1.0 / T
    Uavg = U.mean(axis=1)
    lnU = np.log(Uavg + 1e-8)
    coef, intercept = np.polyfit(invT, lnU, 1)
    pred = coef*invT + intercept
    return coef, intercept, r2_score(lnU, pred), invT, lnU, pred


def gaussian_bell_check(pH: np.ndarray, pI: np.ndarray, U: np.ndarray):
    d = pH - pI
    Uavg = U.mean(axis=1)
    df = pd.DataFrame({"d": d, "u": Uavg})
    dfg = df.groupby(pd.cut(df["d"], 30))["u"].mean().reset_index()
    mids = [iv.mid for iv in dfg["d"]]
    ys = dfg["u"].values
    d1 = np.diff(ys) / np.diff(mids)
    d2 = np.diff(d1) / np.diff(mids[:-1])
    concave_frac = np.mean(d2 < 0)
    return mids, ys, concave_frac


def excluded_volume_check(phi: np.ndarray, U: np.ndarray):
    # convexity: d2u/dphi2 > 0
    Ucol = U[:, 0]   # pick one viscosity output, or loop all of them
    df = pd.DataFrame({"phi": phi, "u": Ucol}).sort_values("phi")
    xs, ys = df["phi"].values, df["u"].values
    d1 = np.diff(ys) / np.diff(xs)
    d2 = np.diff(d1) / np.diff(xs[:-1])
    return np.mean(d2 > 0), xs, ys


def main(csv_path: str):
    loader = DataLoader(csv_path)
    df = loader.load()
    X = loader.get_raw_features()
    U = loader.get_targets()
    targets = loader.TARGET_COLUMNS

    print("\n1) Monotonic-Increasing in Protein_concentration & Sugar_concentration")
    for feat in ["Protein_conc", "Stabilizer_conc"]:
        print(f"  • {feat}:")
        for i, col in enumerate(targets):
            rate = violation_rate_monotonic(
                X[feat].values, U[:, i], increasing=True)
            print(f"      {col:15s}: {100*rate:5.2f}% violations")

    print("\n2) Monotonic-Decreasing in Temperature & Surfactant_concentration")
    for feat in ["Temperature", "Surfactant_conc"]:
        print(f"  • {feat}:")
        for i, col in enumerate(targets):
            rate = violation_rate_monotonic(
                X[feat].values, U[:, i], increasing=False)
            print(f"      {col:15s}: {100*rate:5.2f}% violations")

    print("\n3) Flat-Slope in Buffer_pH (mean |dU/dpH|)")
    for i, col in enumerate(targets):
        mas = mean_abs_slope(X["Buffer_pH"].values, U[:, i])
        print(f"      {col:15s}: mean |dU/dpH| = {mas:.4g}")

    print("\n4) Shear-Thinning (viscosity ↓ with shear rate)")
    rate = shear_thinning_rate(U)
    print(f"      Avg violation rate across all samples: {rate:.4f}")
    plt.hist(shear_thinning_rate(U) *
             np.ones(U.shape[0]), bins=20, edgecolor="k")
    plt.title("Shear-thinning violation rate per sample")
    plt.xlabel("violations fraction")
    plt.show()

    print("\n5) Arrhenius on Temperature")
    slope, inter, r2, invT, lnU, pred = arrhenius_stats(
        X["Temperature"].values, U)
    print(f"      slope={slope:.4f}, intercept={inter:.4f}, R²={r2:.4f}")
    plt.scatter(invT, lnU, s=10, alpha=0.5)
    plt.plot(invT, pred, lw=2)
    plt.title("Arrhenius: ln(U_avg) vs 1/T")
    plt.xlabel("1/Temperature")
    plt.ylabel("ln(U_avg)")
    plt.show()

    print("\n6) Gaussian-Bell around pI (Buffer_pH vs U_avg)")
    mids, ys, concave_frac = gaussian_bell_check(
        X["Buffer_pH"].values, X["PI_mean"].values, U
    )
    print(
        f"      fraction of bins with negative 2nd derivative: {concave_frac:.3f}")
    plt.plot(mids, ys, "-o")
    plt.axvline(0, ls=":", c="gray")
    plt.title("Mean viscosity vs (pH - pI)")
    plt.xlabel("pH - pI")
    plt.ylabel("mean U")
    plt.show()

    print("\n7) Excluded-Volume Divergence (convexity in Protein_concentration)")
    convex_frac, xs, ys = excluded_volume_check(
        X["Protein_conc"].values, U)
    print(f"      fraction of intervals with d2u/dφ2 > 0: {convex_frac:.3f}")
    plt.plot(xs, ys, "-")
    plt.title("Viscosity vs Protein_concentration")
    plt.xlabel("Protein_concentration")
    plt.ylabel("U")
    plt.show()


if __name__ == "__main__":
    csv_path = os.path.join('content', 'train_features.csv')
    main(csv_path)
