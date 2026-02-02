import copy
import io
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

# --- Configuration ---
# Updated to match your latest path
CHECKPOINT_PATH = "models/experiments/20260202_115239/model_2.pt"
REFERENCE_DATA_PATH = "data/processed/formulation_data_augmented_no_trast.csv"
RANDOM_SEED = 42
LEARNING_RATE = 0.02
EPOCHS = 100

# Handle imports
try:
    from inference import ViscosityPredictor
except ImportError:
    try:
        from visq_ml.inference import ViscosityPredictor
    except ImportError:
        print("Error: Could not import ViscosityPredictor.")
        sys.exit(1)

# --- Data ---
trastuzumab_csv = """ID,Protein_type,Protein_class_type,kP,MW,PI_mean,PI_range,Protein_conc,Temperature,Buffer_type,Buffer_pH,Buffer_conc,Salt_type,Salt_conc,Stabilizer_type,Stabilizer_conc,Surfactant_type,Surfactant_conc,Excipient_type,Excipient_conc,C_Class,HCI,Viscosity_100,Viscosity_1000,Viscosity_10000,Viscosity_100000,Viscosity_15000000
F304,Trastuzumab,mAb_IgG1,3,149,8.9,0.3,147,25,Acetate,5,20,none,0,none,0,none,0,none,0,1,1,5.83,4.79,2.84,2.8,1.06
F305,Trastuzumab,mAb_IgG1,3,149,8.9,0.3,157,25,Histidine,6,15,none,0,none,0,none,0,none,0,1,1,3.75,3.75,3.67,3.35,1.9
F306,Trastuzumab,mAb_IgG1,3,149,8.9,0.3,100,25,Histidine,6,15,none,0,Sucrose,0.3,tween-80,0.05,none,0,1,1,2.72,2.72,2.64,2.64,2.34
F307,Trastuzumab,mAb_IgG1,3,149,8.9,0.3,50,25,Histidine,6,15,none,0,none,0,tween-80,0.1,none,0,1,1,1.52,1.52,1.52,1.36,1.2
F308,Trastuzumab,mAb_IgG1,3,149,8.9,0.3,75,25,Histidine,6,15,none,0,Sucrose,0.5,none,0,none,0,1,1,1.56,1.56,1.56,1.56,0.8
F309,Trastuzumab,mAb_IgG1,3,149,8.9,0.3,90,25,Acetate,5,20,none,0,Sucrose,0.3,tween-80,0.1,none,0,1,1,3.19,2.72,2.48,2.72,1.85
F310,Trastuzumab,mAb_IgG1,3,149,8.9,0.3,70,25,Acetate,5,20,none,0,Sucrose,0.5,none,0,none,0,1,1,3.12,2.64,2.64,2.16,1.8
F311,Trastuzumab,mAb_IgG1,3,149,8.9,0.3,100,25,Acetate,5,20,none,0,none,0,tween-80,0.05,none,0,1,1,2.8,2.08,1.6,1.6,1.6
F312,Trastuzumab,mAb_IgG1,3,149,8.9,0.3,60,25,Acetate,5,20,none,0,Sucrose,0.4,tween-80,0.05,none,0,1,1,2,2,1.68,1.44,1.25
F313,Trastuzumab,mAb_IgG1,3,149,8.9,0.3,125,25,Histidine,6,15,none,0,none,0,none,0,none,0,1,1,2.36,2.36,2.36,2.36,2.36
F314,Trastuzumab,mAb_IgG1,3,149,8.9,0.3,125,25,Histidine,6,25,none,0,none,0,none,0,none,0,1,1,3.19,2.52,1.92,1.68,1.68
F315,Trastuzumab,mAb_IgG1,3,149,8.9,0.3,125,25,Histidine,6,25,none,0,none,0,none,0,Arginine,60,1,1,2.88,2.88,2.52,2.52,2.5
F316,Trastuzumab,mAb_IgG1,3,149,8.9,0.3,125,25,Histidine,6,25,none,0,none,0,none,0,Arginine,160,1,1,3.35,2.72,2.08,1.73,1.73
F317,Trastuzumab,mAb_IgG1,3,149,8.9,0.3,125,25,Histidine,6,40,none,0,none,0,none,0,none,0,1,1,2.08,2.08,1.64,1.64,1.64
F319,Trastuzumab,mAb_IgG1,3,149,8.9,0.3,125,25,Histidine,6,25,none,0,Sucrose,0.2,none,0,Arginine,60,1,1,2.96,2.53,2.16,2,2
"""

TARGET_COLS = [
    "Viscosity_100",
    "Viscosity_1000",
    "Viscosity_10000",
    "Viscosity_100000",
    "Viscosity_15000000",
]


def run_loocv_diagnostic():
    df_full = pd.read_csv(io.StringIO(trastuzumab_csv))
    try:
        ref_df = pd.read_csv(REFERENCE_DATA_PATH)
        print(f"Reference data loaded: {len(ref_df)} rows")
    except Exception as e:
        print(f"Warning: Reference data not found ({e}).")
        ref_df = None

    test_range = [1, 3, 5, 8, 10, 12, 14]
    K_FOLDS = 10

    print(
        f"{'Train Size (N)':<15} | {'Mean Unseen MAPE':<18} | {'Mean Unseen RMSE':<18} | {'Std Dev MAPE':<15}"
    )
    print("-" * 75)

    for n_train in test_range:
        errors_mape = []
        errors_rmse = []
        for k in range(K_FOLDS):
            df_shuffled = df_full.sample(
                frac=1, random_state=RANDOM_SEED + k + n_train
            ).reset_index(drop=True)

            df_train = df_shuffled.iloc[:n_train].copy()
            y_train = df_train[TARGET_COLS].values
            df_test = df_shuffled.iloc[n_train:].copy()
            if len(df_test) == 0:
                continue
            df_test_single = df_test.iloc[[0]].copy()
            y_test_single = df_test_single[TARGET_COLS].values
            vp = ViscosityPredictor(CHECKPOINT_PATH)
            vp.hydrate()
            stdout_backup = sys.stdout
            sys.stdout = io.StringIO()
            try:
                vp.learn(
                    df_new=df_train,
                    y_new=y_train,
                    epochs=EPOCHS,
                    lr=LEARNING_RATE,
                    reference_df=ref_df,
                )
            except Exception:
                sys.stdout = stdout_backup
                continue
            sys.stdout = stdout_backup
            y_pred = vp.predict(df_test_single)

            mape = mean_absolute_percentage_error(
                y_test_single.flatten(), y_pred.flatten()
            )
            rmse = np.sqrt(
                mean_squared_error(y_test_single.flatten(), y_pred.flatten())
            )

            errors_mape.append(mape)
            errors_rmse.append(rmse)
        if errors_mape:
            avg_mape = np.mean(errors_mape)
            std_mape = np.std(errors_mape)
            avg_rmse = np.mean(errors_rmse)
            print(
                f"{n_train:<15} | {avg_mape:>18.2%} | {avg_rmse:>18.4f} | {std_mape:>14.2%}"
            )
        else:
            print(f"{n_train:<15} | {'N/A':>18} | {'N/A':>18} | {'N/A':>14}")

    vp_final = ViscosityPredictor(CHECKPOINT_PATH)
    vp_final.hydrate()

    stdout_backup = sys.stdout
    sys.stdout = io.StringIO()
    vp_final.learn(
        df_new=df_full,
        y_new=df_full[TARGET_COLS].values,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        reference_df=ref_df,
    )
    sys.stdout = stdout_backup

    final_preds = vp_final.predict(df_full)
    print(f"{'Shear Rate':<20} | {'R2 Score':<10} | {'RMSE':<10}")
    print("-" * 46)
    for i, col in enumerate(TARGET_COLS):
        r2 = r2_score(df_full[TARGET_COLS].values[:, i], final_preds[:, i])
        rmse = np.sqrt(
            np.mean((df_full[TARGET_COLS].values[:, i] - final_preds[:, i]) ** 2)
        )
        shear_label = col.replace("Viscosity_", "") + " s^-1"
        print(f"{shear_label:<20} | {r2:>8.4f}   | {rmse:>8.4f}")


if __name__ == "__main__":
    run_loocv_diagnostic()
