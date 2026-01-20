import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Mock class for demonstration (Replace with your actual import)
from src.inference import ViscosityPredictor

srs = ["100 1/s", "1,000 1/s", "10,000 1/s", "100,000 1/s", "15,000,000 1/s"]
if __name__ == "__main__":
    # 1. Setup and Prediction
    # ---------------------------------------------------------
    vp = ViscosityPredictor("models/experiments/20260120_102246/model_0.pt")
    df = pd.read_csv("output_mapping_experiment.csv")
    vp.hydrate()
    # Collect predictions efficiently
    # Assuming vp.predict(df) returns an iterable of arrays/scalars
    # If vp.predict can handle the whole DF at once, even better.
    y_matrix = vp.predict(df)
    # Handle NaNs
    nan_rows = np.isnan(y_matrix).any(axis=1)
    if nan_rows.any():
        print(f"Dropping {nan_rows.sum()} NaN rows.")
        y_matrix = y_matrix[~nan_rows]
        df = df[~nan_rows].reset_index(drop=True)

    output_cols = [f"{srs[i]}" for i in range(y_matrix.shape[1])]

    # ---------------------------------------------------------
    # 2. Topological Analysis (3D Upgrade)
    # ---------------------------------------------------------

    # A. 3D t-SNE
    print("Computing 3D t-SNE...")
    tsne = TSNE(n_components=3, perplexity=30, random_state=42)
    y_tsne = tsne.fit_transform(y_matrix)

    # B. Clustering (Regime Identification)
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(y_matrix)

    # Add results to dataframe
    df["Cluster_Label"] = [f"Regime {c}" for c in clusters]
    df["tSNE_1"] = y_tsne[:, 0]
    df["tSNE_2"] = y_tsne[:, 1]
    df["tSNE_3"] = y_tsne[:, 2]  # New 3rd Dimension

    # ---------------------------------------------------------
    # 3. Visualization
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(20, 9))

    # --- Panel A: The 3D Manifold ---
    # We use 'projection=3d' to create the 3D space
    ax = fig.add_subplot(1, 2, 1, projection="3d")

    # Generate colors for clusters
    unique_labels = sorted(df["Cluster_Label"].unique())
    colors = sns.color_palette("viridis", len(unique_labels))
    color_map = dict(zip(unique_labels, colors))

    # Plot each cluster to handle the legend correctly
    for label in unique_labels:
        subset = df[df["Cluster_Label"] == label]
        ax.scatter(
            subset["tSNE_1"],
            subset["tSNE_2"],
            subset["tSNE_3"],
            c=[color_map[label]],
            label=label,
            s=60,
            alpha=0.8,
        )

    ax.set_title("Topology")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_zlabel("Dim 3")
    ax.legend()

    # --- Panel B: The Regime Archetypes (2D Curves) ---
    ax2 = fig.add_subplot(1, 2, 2)

    # Prepare data for lineplot
    y_df = pd.DataFrame(y_matrix, columns=output_cols)
    y_df["Cluster_Label"] = df["Cluster_Label"]
    y_melted = y_df.melt(
        id_vars="Cluster_Label", var_name="Shear_Rate", value_name="Viscosity"
    )

    sns.lineplot(
        data=y_melted,
        x="Shear_Rate",
        y="Viscosity",
        hue="Cluster_Label",
        palette="viridis",
        linewidth=3,
        errorbar="sd",
        ax=ax2,
    )
    ax2.set_title("Regime Archetypes")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
