import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from pandas.plotting import parallel_coordinates
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from src.inference import ViscosityPredictor

# --- Configuration ---
SHEAR_RATES = [
    "100 1/s",
    "1,000 1/s",
    "10,000 1/s",
    "100,000 1/s",
    "15,000,000 1/s",
]
MODEL_ASSETS = [
    r"models/experiments/20260120_152300/model_0.pt",
    r"models/experiments/20260120_152300/model_1.pt",
    r"models/experiments/20260120_152300/model_2.pt",
    r"models/experiments/20260120_152300/model_3.pt",
    r"models/experiments/20260120_152300/model_4.pt",
]
INPUT_DATA = r"drawings/output_mapping_experiment.csv"


def get_predictions(model_path=MODEL_ASSETS[0], data_path=INPUT_DATA):
    """
    Loads data and generates viscosity predictions.
    """
    print(f"Loading model from {model_path}...")
    vp = ViscosityPredictor(model_path)
    df = pd.read_csv(data_path)
    vp.hydrate()

    print("Generating predictions...")
    y_matrix = vp.predict(df)

    # Handle NaNs
    nan_rows = np.isnan(y_matrix).any(axis=1)
    if nan_rows.any():
        print(f"Dropping {nan_rows.sum()} NaN rows.")
        y_matrix = y_matrix[~nan_rows]
        df = df[~nan_rows].reset_index(drop=True)

    output_cols = [f"{SHEAR_RATES[i]}" for i in range(y_matrix.shape[1])]
    return y_matrix, df, output_cols


def get_optimal_clusters(data, max_k=10, random_state=42):
    """
    Determines the optimal number of clusters using Silhouette Score.
    """
    print(f"Determining optimal clusters (checking k=2 to {max_k})...")
    best_score = -1
    best_k = 2

    # Iterate through potential k values
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(data)
        score = silhouette_score(data, labels)

        print(f"  k={k}: Silhouette Score = {score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k

    print(f"-> Optimal number of clusters determined: {best_k}")
    return best_k


def perform_clustering(data, n_clusters=None):
    """
    Performs KMeans clustering. If n_clusters is None, auto-detects it.
    Returns the cluster labels.
    """
    if n_clusters is None:
        n_clusters = get_optimal_clusters(data)

    print(f"Clustering with k={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(data)

    # Return formatted labels (e.g., "Regime 0", "Regime 1")
    return [f"Regime {c}" for c in clusters]


def plot_2d_tsne(matrix, labels, perplexity=30):
    """
    Computes 2D t-SNE and plots the results.
    """
    print("Computing 2D t-SNE...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    y_tsne = tsne.fit_transform(matrix)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=y_tsne[:, 0], y=y_tsne[:, 1], hue=labels, palette="viridis", s=80, alpha=0.8
    )
    plt.title("2D t-SNE Topology")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_3d_tsne(matrix, labels, perplexity=30):
    """
    Computes 3D t-SNE and plots the results.
    """
    print("Computing 3D t-SNE...")
    tsne = TSNE(n_components=3, perplexity=perplexity, random_state=42)
    y_tsne = tsne.fit_transform(matrix)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Helper to map labels to colors
    unique_labels = sorted(list(set(labels)))
    colors = sns.color_palette("viridis", len(unique_labels))
    color_map = dict(zip(unique_labels, colors))

    for label in unique_labels:
        # Boolean mask for current label
        indices = [i for i, x in enumerate(labels) if x == label]

        ax.scatter(
            y_tsne[indices, 0],
            y_tsne[indices, 1],
            y_tsne[indices, 2],
            c=[color_map[label]],
            label=label,
            s=60,
            alpha=0.8,
        )

    ax.set_title("3D t-SNE Topology")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_zlabel("Dim 3")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_regime_archetypes(matrix, labels, output_cols):
    """
    Plots the viscosity profiles (archetypes) for each cluster.
    """
    print("Plotting Regime Archetypes...")

    # Prepare dataframe for Seaborn
    y_df = pd.DataFrame(matrix, columns=output_cols)
    y_df["Cluster_Label"] = labels

    y_melted = y_df.melt(
        id_vars="Cluster_Label", var_name="Shear_Rate", value_name="Viscosity"
    )

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=y_melted,
        x="Shear_Rate",
        y="Viscosity",
        hue="Cluster_Label",
        palette="viridis",
        linewidth=3,
        errorbar="sd",  # Standard deviation shading
    )
    plt.title("Regime Archetypes (Viscosity Profiles)")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_2d_pca(matrix, labels):
    """
    Computes 2D PCA and plots the results with explained variance.
    """
    print("Computing 2D PCA...")
    pca = PCA(n_components=2)
    y_pca = pca.fit_transform(matrix)

    # Get explained variance for axis labels
    var_exp = pca.explained_variance_ratio_

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=y_pca[:, 0], y=y_pca[:, 1], hue=labels, palette="viridis", s=80, alpha=0.8
    )
    plt.title(f"2D PCA Topology (Total Var Explained: {sum(var_exp):.1%})")
    plt.xlabel(f"PC 1 ({var_exp[0]:.1%})")
    plt.ylabel(f"PC 2 ({var_exp[1]:.1%})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_3d_pca(matrix, labels):
    """
    Computes 3D PCA and plots the results with explained variance.
    """
    print("Computing 3D PCA...")
    pca = PCA(n_components=3)
    y_pca = pca.fit_transform(matrix)

    var_exp = pca.explained_variance_ratio_

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Helper to map labels to colors (consistent with t-SNE)
    unique_labels = sorted(list(set(labels)))
    colors = sns.color_palette("viridis", len(unique_labels))
    color_map = dict(zip(unique_labels, colors))

    for label in unique_labels:
        indices = [i for i, x in enumerate(labels) if x == label]
        ax.scatter(
            y_pca[indices, 0],
            y_pca[indices, 1],
            y_pca[indices, 2],
            c=[color_map[label]],
            label=label,
            s=60,
            alpha=0.8,
        )

    ax.set_title(f"3D PCA Topology (Total Var Explained: {sum(var_exp):.1%})")
    ax.set_xlabel(f"PC 1 ({var_exp[0]:.1%})")
    ax.set_ylabel(f"PC 2 ({var_exp[1]:.1%})")
    ax.set_zlabel(f"PC 3 ({var_exp[2]:.1%})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_umap_2d(matrix, labels, n_neighbors=15, min_dist=0.1):
    """
    Computes 2D UMAP and plots the results.
    Preserves global structure better than t-SNE.
    """
    print("Computing 2D UMAP...")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=42
    )
    y_umap = reducer.fit_transform(matrix)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=y_umap[:, 0], y=y_umap[:, 1], hue=labels, palette="viridis", s=60, alpha=0.8
    )
    plt.title("2D UMAP Topology (Global Structure Preserved)")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_parallel_coordinates_custom(matrix, labels, output_cols):
    """
    Visualizes the raw flow of data across dimensions using Parallel Coordinates.
    Good for seeing the density and 'tightness' of clusters.
    """
    print("Plotting Parallel Coordinates...")

    # Prepare DataFrame
    df_vis = pd.DataFrame(matrix, columns=output_cols)
    df_vis["Cluster"] = labels

    plt.figure(figsize=(15, 6))

    # We use pandas built-in parallel_coordinates, but wrap it to control styling
    # Note: For very large datasets, sample the df_vis before plotting to improve performance
    if len(df_vis) > 2000:
        print(f"  Note: Sampling 2000 points from {len(df_vis)} for readability.")
        df_vis = df_vis.sample(2000, random_state=42)

    parallel_coordinates(
        df_vis,
        "Cluster",
        colormap="viridis",
        alpha=0.3,  # Transparency is key here to see density
        linewidth=1,
    )

    plt.title("Parallel Coordinates (Raw Data Flow)")
    plt.ylabel("Viscosity")
    plt.xlabel("Shear Rate Dimensions")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    y_matrix, df, output_cols = get_predictions()
    cluster_labels = perform_clustering(y_matrix, n_clusters=7)
    plot_2d_tsne(y_matrix, cluster_labels)
    plot_3d_tsne(y_matrix, cluster_labels)
    plot_2d_pca(y_matrix, cluster_labels)
    plot_3d_pca(y_matrix, cluster_labels)
    plot_umap_2d(y_matrix, cluster_labels)
    plot_parallel_coordinates_custom(y_matrix, cluster_labels, output_cols)
    plot_regime_archetypes(y_matrix, cluster_labels, output_cols)
