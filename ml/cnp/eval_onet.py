import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from train_o_net import CrossSampleCNP, load_and_preprocess

# Settings
MODEL_PATH = "models/experiments/o_net/best_model.pth"
DATA_PATH = "data/raw/formulation_data_02052026.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_and_plot():
    # 1. Load Data & Model
    print("Loading resources...")
    samples, static_dim = load_and_preprocess(DATA_PATH)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    config = checkpoint["config"]

    model = CrossSampleCNP(
        static_dim, config["hidden_dim"], config["latent_dim"], config["dropout"]
    ).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # 2. Group Data
    grouped = {}
    for s in samples:
        if s["group"] not in grouped:
            grouped[s["group"]] = []
        grouped[s["group"]].append(s)

    # 3. Predict & Store Errors
    results = []

    print("Running inference...")
    with torch.no_grad():
        for protein, items in grouped.items():
            if len(items) < 5:
                continue

            split = len(items) // 2
            context_items = items[:split]
            target_items = items[split:]

            # Build Context Tensor
            ctx_list = []
            for s in context_items:
                stat = s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1)
                ctx_list.append(torch.cat([s["points"], stat], dim=1))
            ctx_tensor = torch.cat(ctx_list, dim=0).unsqueeze(0).to(DEVICE)

            # Predict Targets
            preds = []
            actuals = []

            for t in target_items:
                # 1. Prepare Inputs
                q_shear = t["points"][:, [0]].unsqueeze(0).to(DEVICE)
                q_visc_true = t["points"][:, [1]].unsqueeze(0).to(DEVICE)

                n_points = t["points"].shape[0]
                q_stat = (
                    t["static"].unsqueeze(0).repeat(n_points, 1).unsqueeze(0).to(DEVICE)
                )

                # 2. Inference
                y_pred = model(ctx_tensor, q_shear, q_stat)

                # 3. Flatten predictions (Safe for plotting)
                batch_preds = y_pred.view(-1).cpu().numpy().tolist()
                batch_actuals = q_visc_true.view(-1).cpu().numpy().tolist()

                preds.extend(batch_preds)
                actuals.extend(batch_actuals)

            if len(preds) == 0:
                continue

            mse = np.mean((np.array(preds) - np.array(actuals)) ** 2)
            results.append(
                {"protein": protein, "mse": mse, "preds": preds, "actuals": actuals}
            )

    # 4. Sort by Error
    results.sort(key=lambda x: x["mse"])
    best_3 = results[:3]
    worst_3 = results[-3:]

    # 5. Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # FIX: Safely handle missing loss key in title
    val_loss_val = checkpoint.get("best_loss")
    if isinstance(val_loss_val, (int, float)):
        loss_str = f"{val_loss_val:.4f}"
    else:
        loss_str = "N/A"

    fig.suptitle(f"Model Performance (Best 3 vs Worst 3)\nVal Loss: {loss_str}")

    def plot_on_ax(ax, res, title_prefix):
        y_true = 10 ** np.array(res["actuals"])
        y_pred = 10 ** np.array(res["preds"])

        ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="w")

        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, "r--", alpha=0.75, zorder=0)

        ax.set_title(f"{title_prefix}: {res['protein']}\nMSE: {res['mse']:.4f}")
        ax.set_xlabel("True Viscosity (cP)")
        ax.set_ylabel("Predicted Viscosity (cP)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="-", alpha=0.2)

    for i, res in enumerate(best_3):
        if i < 3:
            plot_on_ax(axes[0, i], res, "BEST")

    for i, res in enumerate(worst_3):
        if i < 3:
            plot_on_ax(axes[1, i], res, "WORST")

    plt.tight_layout()
    plt.savefig("models/experiments/o_net/performance_grid.png")
    print("\nPlot saved to models/experiments/o_net/performance_grid.png")


if __name__ == "__main__":
    evaluate_and_plot()
