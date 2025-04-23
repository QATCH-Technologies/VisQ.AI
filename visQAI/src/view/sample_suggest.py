import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from typing import Dict, List, Any
from mpl_toolkits.mplot3d import Axes3D
Sample = Dict[str, Any]


def k_nearest_visc_neighbors(
    sample: Sample,
    train_df: pd.DataFrame,
    viscosity_cols: List[str],
    k: int = 50
) -> pd.DataFrame:
    """
    Return the k rows whose 5-point viscosity profiles are closest
    to sample['visc_profile'], ignoring all other features.
    """
    V = train_df[viscosity_cols].values
    target = np.array(sample['visc_profile']).reshape(1, -1)
    dists = np.linalg.norm(V - target, axis=1)
    idx = np.argsort(dists)[:k]
    return train_df.iloc[idx].copy()


def suggest_via_kmeans_params(
    neighbors: pd.DataFrame,
    n_suggestions: int = 5,
    random_state: int = 0
) -> List[Dict[str, Any]]:
    """
    Cluster the neighbor set in the 5 tunable parameters:
      - Protein (numeric)
      - Buffer (categorical)
      - Sugar (M) (numeric)
      - Surfactant (categorical)
      - TWEEN (numeric)
    Return one medoid sample per cluster.
    """
    # 1) Define feature columns
    num_cols = ['Protein', 'Sugar (M)', 'TWEEN']
    cat_cols = ['Buffer', 'Surfactant']

    # 2) Build a ColumnTransformer to scale and encode
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ], remainder='drop')

    # 3) Transform the neighbor dataframe into a unified array
    X = preprocessor.fit_transform(neighbors)

    # 4) Perform k-means clustering on the transformed space
    km = KMeans(n_clusters=n_suggestions, random_state=random_state)
    km.fit(X)
    centers = km.cluster_centers_

    # 5) Find the nearest actual neighbor (medoid) for each cluster center
    suggestions = []
    for center in centers:
        dists = np.linalg.norm(X - center, axis=1)
        idx = np.argmin(dists)
        row = neighbors.iloc[idx]
        suggestions.append({
            'Protein':    row['Protein'],
            'Buffer':     row['Buffer'],
            'Sugar (M)':  row['Sugar (M)'],
            'Surfactant': row['Surfactant'],
            'TWEEN':      row['TWEEN']
        })

    return suggestions


def plot_suggestions_space(
    train_df: pd.DataFrame,
    neighbors: pd.DataFrame,
    suggestions: List[Dict[str, Any]],
    sample: Sample
):
    # 1) Build transformer: scale numeric, encode categorical
    num_cols = ['Protein', 'Sugar (M)', 'TWEEN']
    cat_cols = ['Buffer', 'Surfactant']
    pre = ColumnTransformer([
        ('num',   StandardScaler(),              num_cols),
        ('cat',   OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
    ], remainder='drop')

    # 2) Fit on the full training set & transform
    X_train = pre.fit_transform(train_df)

    # 3) Fit PCA to 3 components
    pca = PCA(n_components=3, random_state=0)
    X_train3 = pca.fit_transform(X_train)

    # 4) Prepare and project neighbors
    X_neigh3 = pca.transform(pre.transform(neighbors))

    # 5) Prepare and project suggestions
    sugg_df = pd.DataFrame(suggestions)
    X_sugg3 = pca.transform(pre.transform(sugg_df))

    # 6) Prepare and project the single initial sample
    sample_df = pd.DataFrame([{
        'Protein':    sample['Protein'],
        'Sugar (M)':  sample['Sugar (M)'],
        'TWEEN':      sample['TWEEN'],
        'Buffer':     sample['Buffer'],
        'Surfactant': sample['Surfactant']
    }])
    X_samp3 = pca.transform(pre.transform(sample_df))

    # 7) Plot all four sets in 3D
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(*X_train3.T,    alpha=0.2, label='Known space')
    ax.scatter(*X_neigh3.T,    alpha=0.7, label='Neighbors')
    ax.scatter(*X_sugg3.T,     marker='*', s=120, label='Suggestions')
    ax.scatter(*X_samp3.T,     marker='X', s=120, label='Initial sample')

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('3D PCA of Formulation Space & Suggestions')
    ax.legend()
    plt.tight_layout()
    plt.show()


def main():
    # 1) Load your dataset (update path as needed)
    df = pd.read_csv('content/formulation_data_04222025_2.csv')

    # 2) Define your viscosity columns
    viscosity_cols = [
        'Viscosity100', 'Viscosity1000', 'Viscosity10000',
        'Viscosity100000', 'Viscosity15000000'
    ]

    # 3) Prompt the user for initial sample
    sample = {
        'Protein':      float(input("Protein concentration: ")),
        'Buffer':       input("Buffer type: "),
        'Surfactant':   input("Surfactant type: "),
        'Sugar (M)':    float(input("Sugar concentration (M): ")),
        'TWEEN':        float(input("TWEEN concentration: ")),
        'visc_profile': np.array([
            float(x) for x in input(
                "Enter 5-point viscosity profile (comma-separated): "
            ).split(',')
        ])
    }

    # 4) Find neighbors in viscosity space
    neighbors = k_nearest_visc_neighbors(
        sample, df, viscosity_cols, k=50
    )

    # 5) Suggest new experiments via k-means
    suggestions = suggest_via_kmeans_params(
        neighbors, n_suggestions=4
    )

    # 6) Print suggestions
    print("\nSuggested new experiments:")
    for i, s in enumerate(suggestions, 1):
        print(
            f"Option {i}: Protein={s['Protein']}, "
            f"Buffer={s['Buffer']}, Sugar(M)={s['Sugar (M)']}, "
            f"Surfactant={s['Surfactant']}, TWEEN={s['TWEEN']}"
        )

    # 7) Plot the known space and the suggestions
    plot_suggestions_space(df, neighbors, suggestions, sample)


if __name__ == "__main__":
    main()
