# tests/test_regressors_per_sample_rmse_improved.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from linear_predictor import LinearPredictor
from nn_predictor import NNPredictor
from xgb_predictor import XGBPredictor
CSV_PATH = os.path.join("content", "formulation_data_04222025.csv")
TARGET_COLS = [
    "Viscosity100", "Viscosity1000",
    "Viscosity10000", "Viscosity100000",
    "Viscosity15000000"
]
SHEAR_RATES = np.array([100, 1000, 10000, 100000, 15000000])

OUTPUT_DIR = "per_sample_plots_rmse_improved"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def predict_viscosity(params, predictor_type: str):
    df = pd.DataFrame({**params, 'shear_rate': SHEAR_RATES})
    if predictor_type == "XGB":
        pred = XGBPredictor(os.path.join(
            'visqAI', 'objects', 'xgb_regressor')).predict(df)
    elif predictor_type == "Neural Net":
        pred = NNPredictor(os.path.join(
            'visqAI', 'objects', 'nn_regressor')).predict(df)
    else:
        pred = LinearPredictor(os.path.join(
            'visqAI', 'objects', 'linear_regressor')).predict(df)
    return np.array(pred).flatten()[:len(SHEAR_RATES)]


def compute_rmse(actual, pred):
    return np.sqrt(np.mean((actual - pred) ** 2))


def test_per_sample_rmse_improved():
    # load data
    df = pd.read_csv(CSV_PATH)
    sample_ids = df['ID'].astype(str).tolist()  # pull ID column for labels
    X = df.drop(columns=TARGET_COLS)
    y = df[TARGET_COLS]

    models = ["Linear", "Neural Net", "XGB"]
    errors = {m: [] for m in models}

    # per-sample RMSE & individual plots
    for idx, row in X.iterrows():
        # don't pass ID into predict
        params = row.drop(labels=['ID']).to_dict()
        actual = y.loc[idx].values
        preds = {m: predict_viscosity(params, m) for m in models}

        for m in models:
            errors[m].append(compute_rmse(actual, preds[m]))

        # save individual sample plot (unchanged from before)
        plt.figure(figsize=(6, 4))
        plt.plot(SHEAR_RATES, actual, marker='o', linewidth=2, label="Actual")
        for m in models:
            plt.plot(
                SHEAR_RATES, preds[m],
                marker='x', linestyle='--',
                label=f"{m} (RMSE={errors[m][-1]:.2f})"
            )
        plt.xscale('log')
        plt.xlabel("Shear Rate")
        plt.ylabel("Viscosity")
        plt.title(f"Sample {sample_ids[idx]} Predictions")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"sample_{sample_ids[idx]}.png"))
        plt.close()

    # print overall averages
    print("\n=== Average RMSE Across All Samples ===")
    for m in models:
        print(f"{m:10s}: {np.mean(errors[m]):.3f}")

    # --- improved summary plot ---
    plt.figure(figsize=(12, 5))

    x = np.arange(len(sample_ids))
    for m in models:
        # line + scatter for clarity
        plt.plot(x, errors[m], marker='o', linestyle='-', label=m)
        plt.scatter(x, errors[m], s=50)  # emphasize points
        # mean line
        mean_err = np.mean(errors[m])
        plt.hlines(
            mean_err, xmin=0, xmax=len(x)-1,
            linestyles='dashed',
            label=f"{m} mean={mean_err:.2f}"
        )

    # label samples on x-axis
    plt.xticks(ticks=x, labels=sample_ids, rotation=45, ha='right')
    plt.xlabel("Sample ID")
    plt.ylabel("RMSE")
    plt.title("Per‑Sample RMSE by Model")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(ncol=2, fontsize="small")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rmse_summary_improved.png"))
    plt.close()


def test_first_last_measurement_errors_improved():
    # load data
    df = pd.read_csv(CSV_PATH)
    sample_ids = df['ID'].astype(str).tolist()
    X = df.drop(columns=TARGET_COLS)
    y = df[TARGET_COLS]

    models = ["Linear", "Neural Net", "XGB"]
    first_errors = {m: [] for m in models}
    last_errors = {m: [] for m in models}

    # collect raw values for plotting and Excel
    actual_first = []
    actual_last = []
    first_preds = {m: [] for m in models}
    last_preds = {m: [] for m in models}

    # per‑sample first/last errors
    print("\n=== Per‑Sample First and Last Measurement Errors ===")
    for idx, row in X.iterrows():
        params = row.drop(labels=['ID']).to_dict()
        actual = y.loc[idx].values
        preds = {m: predict_viscosity(params, m) for m in models}

        actual_first.append(actual[0])
        actual_last.append(actual[-1])

        for m in models:
            err_f = abs(actual[0] - preds[m][0])
            err_l = abs(actual[-1] - preds[m][-1])
            first_errors[m].append(err_f)
            last_errors[m].append(err_l)
            first_preds[m].append(preds[m][0])
            last_preds[m].append(preds[m][-1])

        errs = "  ".join(
            f"{m}: first={first_errors[m][-1]:.2f}, last={last_errors[m][-1]:.2f}"
            for m in models
        )
        print(f"Sample {sample_ids[idx]}  ->  {errs}")

    # --- write results to Excel ---
    # build per-sample DataFrame
    per_sample_data = {
        "Sample ID": sample_ids,
        "Actual First (100)": actual_first,
        "Actual Last (15e6)": actual_last,
    }
    for m in models:
        per_sample_data[f"{m} Pred First"] = first_preds[m]
        per_sample_data[f"{m} Error First"] = first_errors[m]
        per_sample_data[f"{m} Pred Last"] = last_preds[m]
        per_sample_data[f"{m} Error Last"] = last_errors[m]

    per_sample_df = pd.DataFrame(per_sample_data)

    # build summary DataFrame
    summary_data = {
        "Model": models,
        "Avg Error First": [np.mean(first_errors[m]) for m in models],
        "Avg Error Last":  [np.mean(last_errors[m]) for m in models]
    }
    summary_df = pd.DataFrame(summary_data)

    # write both sheets
    excel_path = os.path.join(OUTPUT_DIR, "first_last_results.xlsx")
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        per_sample_df.to_excel(writer, sheet_name="Per_Sample", index=False)
        summary_df.to_excel(writer, sheet_name="Summary",    index=False)

    print(f"\nWrote detailed results to {excel_path}")

    # --- (rest of your plotting code unchanged) ---
    x = np.arange(len(sample_ids))
    cmap = plt.get_cmap("tab10")
    colors = {m: cmap(i) for i, m in enumerate(models)}

    plt.figure(figsize=(16, 6), dpi=120)
    plt.suptitle(
        "First & Last Shear‑Rate Measurements\nActual (●) vs. Model Predictions (×)\n"
        "Dashed lines show errors; arrows mark the largest-error sample per model",
        fontsize=14,
        y=1.02
    )

    # First-measurement subplot
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(x, actual_first, 'ko-', label="Actual", markersize=6)
    for m in models:
        ax1.scatter(x, first_preds[m], marker='x',
                    s=60, color=colors[m], label=m)
        for xi, (a, p) in enumerate(zip(actual_first, first_preds[m])):
            ax1.vlines(xi, a, p, linestyles='dashed',
                       color=colors[m], alpha=0.5)
        worst_i = np.argmax(np.abs(np.array(first_preds[m]) - actual_first))
        ax1.annotate(
            f"{m} worst: {sample_ids[worst_i]}",
            xy=(worst_i, first_preds[m][worst_i]),
            xytext=(worst_i, first_preds[m]
                    [worst_i] + 0.1 * max(actual_first)),
            arrowprops=dict(arrowstyle="->", color=colors[m]),
            color=colors[m],
            fontsize=9
        )
    ax1.set_title("First Measurement (Shear = 100)", pad=12)
    ax1.set_xlabel("Sample ID")
    ax1.set_ylabel("Viscosity")
    ax1.set_xticks(x)
    ax1.set_xticklabels(sample_ids, rotation=45, ha='right')
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    ax1.legend(fontsize="small")

    # Last-measurement subplot
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(x, actual_last, 'ko-', label="Actual", markersize=6)
    for m in models:
        ax2.scatter(x, last_preds[m], marker='x',
                    s=60, color=colors[m], label=m)
        for xi, (a, p) in enumerate(zip(actual_last, last_preds[m])):
            ax2.vlines(xi, a, p, linestyles='dashed',
                       color=colors[m], alpha=0.5)
        worst_i = np.argmax(np.abs(np.array(last_preds[m]) - actual_last))
        ax2.annotate(
            f"{m} worst: {sample_ids[worst_i]}",
            xy=(worst_i, last_preds[m][worst_i]),
            xytext=(worst_i, last_preds[m][worst_i] + 0.1 * max(actual_last)),
            arrowprops=dict(arrowstyle="->", color=colors[m]),
            color=colors[m],
            fontsize=9
        )
    ax2.set_title("Last Measurement (Shear = 15 000 000)", pad=12)
    ax2.set_xlabel("Sample ID")
    ax2.set_xticks(x)
    ax2.set_xticklabels(sample_ids, rotation=45, ha='right')
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    ax2.legend(fontsize="small")

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig(os.path.join(
        OUTPUT_DIR, "first_last_predictions_enhanced.png"))
    plt.close()

    # aggregate averages
    print("\n=== Average First‑Measurement Error Across All Samples ===")
    for m in models:
        print(f"{m:10s}: {np.mean(first_errors[m]):.3f}")

    print("\n=== Average Last‑Measurement Error Across All Samples ===")
    for m in models:
        print(f"{m:10s}: {np.mean(last_errors[m]):.3f}")


if __name__ == "__main__":
    test_per_sample_rmse_improved()
    test_first_last_measurement_errors_improved()
