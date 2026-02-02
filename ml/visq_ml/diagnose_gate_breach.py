import copy
import io
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

# --- Configuration ---
CHECKPOINT_PATH = "models/experiments/20260202_152217/model_3.pt"
REFERENCE_DATA_PATH = "data/processed/formulation_data_augmented_no_trast.csv"
RANDOM_SEED = 42
LEARNING_RATE = 0.05
EPOCHS = 200

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


def load_data():
    df_full = pd.read_csv(io.StringIO(trastuzumab_csv))
    try:
        ref_df = pd.read_csv(REFERENCE_DATA_PATH)
        print(f"Reference data loaded: {len(ref_df)} rows")
    except Exception as e:
        print(f"Warning: Reference data not found ({e}).")
        ref_df = None
    return df_full, ref_df


def run_loocv_diagnostic(df_full, ref_df):
    """
    Standard Random Split CV.
    Tests Interpolation (Gap Filling) and Reference Drift.
    """
    if ref_df is not None:
        y_ref = ref_df[TARGET_COLS].values
        vp_base = ViscosityPredictor(CHECKPOINT_PATH).hydrate()
        preds_base = vp_base.predict(ref_df)
        baseline_ref_mape = mean_absolute_percentage_error(y_ref.flatten(), preds_base.flatten())
        print(f"Baseline Reference MAPE: {baseline_ref_mape:.2%}")
        del vp_base
    else:
        baseline_ref_mape = 0.0

    test_range = [1, 2, 3, 5, 8, 10, 12, 14]
    K_FOLDS = 5 

    header = (
        f"{'N':<3} | "
        f"{'Seen MAPE':<10} | "
        f"{'Unseen MAPE':<12} | "
        f"{'Unseen RMSE':<12} | "
        f"{'Ref MAPE':<10} | "
        f"{'Ref Drift':<10}"
    )
    print("\n" + "=" * len(header))
    print(" INTERPOLATION TEST (Random Split)")
    print(" Checks: Can we fill in gaps within the design space?")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for n_train in test_range:
        metrics = {"seen_mape": [], "unseen_mape": [], "unseen_rmse": [], "ref_mape": []}

        for k in range(K_FOLDS):
            df_shuffled = df_full.sample(frac=1, random_state=RANDOM_SEED + k + n_train).reset_index(drop=True)
            df_train = df_shuffled.iloc[:n_train].copy()
            y_train = df_train[TARGET_COLS].values
            df_test = df_shuffled.iloc[n_train:].copy()
            if len(df_test) == 0: continue
            y_test = df_test[TARGET_COLS].values

            vp = ViscosityPredictor(CHECKPOINT_PATH).hydrate()
            
            # Silent Train
            stdout_backup = sys.stdout
            sys.stdout = io.StringIO()
            try:
                vp.learn(df_train, y_train, epochs=EPOCHS, lr=LEARNING_RATE)
            except:
                pass
            sys.stdout = stdout_backup

            # Metrics
            pred_train = vp.predict(df_train)
            metrics["seen_mape"].append(mean_absolute_percentage_error(y_train.flatten(), pred_train.flatten()))

            pred_test = vp.predict(df_test)
            metrics["unseen_mape"].append(mean_absolute_percentage_error(y_test.flatten(), pred_test.flatten()))
            metrics["unseen_rmse"].append(np.sqrt(mean_squared_error(y_test.flatten(), pred_test.flatten())))

            if ref_df is not None:
                pred_ref = vp.predict(ref_df)
                metrics["ref_mape"].append(mean_absolute_percentage_error(y_ref.flatten(), pred_ref.flatten()))

        if metrics["unseen_mape"]:
            avg_seen = np.mean(metrics["seen_mape"])
            avg_unseen = np.mean(metrics["unseen_mape"])
            avg_rmse = np.mean(metrics["unseen_rmse"])
            if metrics["ref_mape"]:
                avg_ref = np.mean(metrics["ref_mape"])
                drift = avg_ref - baseline_ref_mape
            else:
                avg_ref = 0.0
                drift = 0.0

            print(
                f"{n_train:<3} | "
                f"{avg_seen:>10.2%} | "
                f"{avg_unseen:>12.2%} | "
                f"{avg_rmse:>12.4f} | "
                f"{avg_ref:>10.2%} | "
                f"{drift:>+10.2%}"
            )

def run_extrapolation_diagnostic(df_full):
    """
    Concentration Extrapolation Test.
    Train on LOW Concentration -> Test on HIGH Concentration.
    Tests: Did we learn the Physics (Slope) or just the Bias (Intercept)?
    """
    # Sort by concentration
    df_sorted = df_full.sort_values("Protein_conc").reset_index(drop=True)
    
    # Define test points (Requires enough points to form a train set)
    # Trastuzumab Concs: 50, 60, 70, 75, 90, 100, 100, 125...
    test_range = [3, 5, 8, 10] 

    header = (
        f"{'N':<3} | "
        f"{'Train Conc Range':<18} | "
        f"{'Test Conc Range':<18} | "
        f"{'Seen MAPE':<10} | "
        f"{'Extra MAPE':<10}"
    )
    print("\n" + "=" * len(header))
    print(" EXTRAPOLATION TEST (Physics Generalization)")
    print(" Checks: Can we predict High Concentration behavior from Low Concentration data?")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for n_train in test_range:
        df_train = df_sorted.iloc[:n_train].copy()
        df_test = df_sorted.iloc[n_train:].copy()
        
        y_train = df_train[TARGET_COLS].values
        y_test = df_test[TARGET_COLS].values
        
        # Info strings
        min_tr, max_tr = df_train["Protein_conc"].min(), df_train["Protein_conc"].max()
        min_te, max_te = df_test["Protein_conc"].min(), df_test["Protein_conc"].max()
        tr_range = f"{min_tr:.0f}-{max_tr:.0f} mg/mL"
        te_range = f"{min_te:.0f}-{max_te:.0f} mg/mL"

        vp = ViscosityPredictor(CHECKPOINT_PATH).hydrate()
        
        # Silent Train
        stdout_backup = sys.stdout
        sys.stdout = io.StringIO()
        try:
            vp.learn(df_train, y_train, epochs=EPOCHS, lr=LEARNING_RATE)
        except:
            pass
        sys.stdout = stdout_backup

        pred_train = vp.predict(df_train)
        seen_mape = mean_absolute_percentage_error(y_train.flatten(), pred_train.flatten())

        pred_test = vp.predict(df_test)
        extra_mape = mean_absolute_percentage_error(y_test.flatten(), pred_test.flatten())

        print(
            f"{n_train:<3} | "
            f"{tr_range:<18} | "
            f"{te_range:<18} | "
            f"{seen_mape:>10.2%} | "
            f"{extra_mape:>10.2%}"
        )


if __name__ == "__main__":
    df, ref_df = load_data()
    
    # 1. Run Standard Interpolation / Drift Check
    run_loocv_diagnostic(df, ref_df)
    
    # 2. Run New Physics Generalization Check
    run_extrapolation_diagnostic(df)