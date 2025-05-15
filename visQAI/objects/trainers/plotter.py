import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

from predictor import ViscosityPredictor   # ← adjust to your real import

# ─── CONFIG ────────────────────────────────────────────────────────────────────
ARCH_ROOT = "visQAI/objects/architectures"
CSV_PATH = "content/formulation_data_05072025.csv"
MODEL_FN = "model.keras"
PREP_FN = "preprocessor.pkl"

SHEARS = [100, 1000, 10000, 100000, 15000000]
TARGET_COLS = [f"Viscosity_{s}" for s in SHEARS]
df = pd.read_csv(CSV_PATH)
FEATURE_DF = df.drop(columns=TARGET_COLS)
archs = [d for d in os.listdir(ARCH_ROOT)
         if os.path.isdir(os.path.join(ARCH_ROOT, d))]

if not archs:
    print(f"[ERROR] no subfolders in {ARCH_ROOT}", file=sys.stderr)
    sys.exit(1)

for arch in archs:
    arch_path = os.path.join(ARCH_ROOT, arch)
    model_path = os.path.join(arch_path, MODEL_FN)
    prep_path = os.path.join(arch_path, PREP_FN)

    # sanity check
    if not os.path.isfile(model_path) or not os.path.isfile(prep_path):
        print(
            f"[WARN] skipping {arch}: missing {MODEL_FN} or {PREP_FN}", file=sys.stderr)
        continue

    # load predictor
    try:
        predictor = ViscosityPredictor(model_path, prep_path)
    except Exception as e:
        print(f"[ERROR] loading {arch}: {e}", file=sys.stderr)
        continue

    # run through all rows once
    preds = predictor.predict(FEATURE_DF)   # shape: (n_samples, 5)
    # ensure it's a numpy array
    preds = np.asarray(preds)
    if preds.ndim != 2 or preds.shape[1] != len(SHEARS):
        print(
            f"[ERROR] {arch} output shape {preds.shape} ≠ (n, {len(SHEARS)})", file=sys.stderr)
        continue
    for idx, shear in enumerate(SHEARS):
        col = TARGET_COLS[idx]
        y_true = df[col].values
        y_pred = preds[:, idx]
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        if not mask.all():
            dropped = np.count_nonzero(~mask)
            print(f"Dropped {dropped} NaN entries for shear={shear}")
        y_true_f = y_true[mask]
        y_pred_f = y_pred[mask]
        if y_true_f.size == 0:
            print(f"No valid data for shear={shear}, skipping plot.")
            continue
        r2 = r2_score(y_true_f, y_pred_f)
        mae = mean_absolute_error(y_true_f, y_pred_f)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(
            y_true_f,
            y_pred_f,
            s=30,
            alpha=0.5,
            edgecolor='k',
            linewidth=0.5,
            label="Prediction vs Actual"
        )
        lo, hi = min(y_true_f.min(), y_pred_f.min()), max(
            y_true_f.max(), y_pred_f.max())
        ax.plot([lo, hi], [lo, hi], linestyle='--',
                linewidth=1.5, label="Ideal (y = x)")
        ax.text(
            0.05, 0.95,
            f"$R^2$ = {r2:.2f}\nMAE = {mae:.2f}",
            transform=ax.transAxes,
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", alpha=0.2)
        )
        ax.set_xlabel("Actual Viscosity")
        ax.set_ylabel("Predicted Viscosity")
        ax.set_title(f"{arch} — viscosity @ {shear} s⁻¹")
        ax.grid(which="both", linestyle=":", linewidth=0.5, alpha=0.7)
        ax.legend(loc="lower right", fontsize="small")
        fig.tight_layout()
        out_name = f"{arch}_viscosity_{shear}.png"
        out_path = os.path.join(arch_path, out_name)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        print(f"Saved: {out_path}")
