import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_hard_igg1_curves(csv_path="data/raw/formulation_data_02162026.csv"):
    df = pd.read_csv(csv_path)

    # Standardize to lowercase to match our pipeline logic
    if "Protein_type" in df.columns:
        df["Protein_type"] = df["Protein_type"].astype(str).str.lower()
    else:
        print("Error: 'Protein_type' column not found.")
        return

    target_groups = ["bevacizumab", "trastuzumab", "adalimumab"]
    colors = {"bevacizumab": "red", "trastuzumab": "blue", "adalimumab": "green"}

    # Map the exact shear rates to your column names
    shear_cols = {
        100.0: "Viscosity_100",
        1000.0: "Viscosity_1000",
        10000.0: "Viscosity_10000",
        100000.0: "Viscosity_100000",
        15000000.0: "Viscosity_15000000",
    }

    plt.figure(figsize=(10, 6))
    labels_added = set()

    for _, row in df.iterrows():
        group = row.get("Protein_type", "unknown")
        if group not in target_groups:
            continue

        x_vals, y_vals = [], []

        for shear_val, col_name in shear_cols.items():
            if col_name in row and pd.notna(row[col_name]) and row[col_name] > 0:
                x_vals.append(shear_val)
                y_vals.append(row[col_name])

        # Only plot if we have a curve
        if len(x_vals) > 1:
            label = group if group not in labels_added else None
            plt.plot(
                x_vals,
                y_vals,
                marker="o",
                alpha=0.4,  # Use transparency to see overlapping curves
                color=colors[group],
                label=label,
            )
            if label:
                labels_added.add(group)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Shear Rate (1/s)")
    plt.ylabel("Viscosity (cP)")
    plt.title("Viscosity Profiles: Bevacizumab vs. Trastuzumab & Adalimumab")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    # Ensure legend handles overlap gracefully
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.tight_layout()
    plt.show()


# Execute the function
plot_hard_igg1_curves()
