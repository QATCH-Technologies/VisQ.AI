import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import griddata
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
    y_matrix = vp.predict(df, return_log_space=False)
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
    # 3. Visualization (Animated Smooth Surface)
    # ---------------------------------------------------------
    import matplotlib.animation as animation
    from matplotlib import cm, colors
    from scipy.interpolate import griddata

    print("Preparing 3D Surface data...")

    # A. Data Preparation (Smooth Interpolation)
    # -----------------------------------------------------
    x, y, z = df["tSNE_1"], df["tSNE_2"], df["tSNE_3"]

    # Color mapping: Log Mean Viscosity
    color_metric = np.log1p(y_matrix.mean(axis=1))

    # Create a high-res grid (100x100) for silky smooth surface
    resolution = 100
    xi = np.linspace(x.min(), x.max(), resolution)
    yi = np.linspace(y.min(), y.max(), resolution)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate Z (height) and Color values onto the grid using cubic spline
    zi = griddata((x, y), z, (xi, yi), method="cubic")
    ci = griddata((x, y), color_metric, (xi, yi), method="cubic")

    # B. Plotting the Static Frame
    # -----------------------------------------------------
    # Use a large square figure for high-res video output
    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    # Handle colors manually to paint the face of the surface
    norm = colors.Normalize(vmin=color_metric.min(), vmax=color_metric.max())
    surface_colors = cm.plasma(norm(ci))

    # Plot the surface
    # rstride=1/cstride=1 ensures we render every grid pixel
    surf = ax.plot_surface(
        xi,
        yi,
        zi,
        facecolors=surface_colors,
        rstride=1,
        cstride=1,
        shade=True,
        antialiased=True,
        linewidth=0,  # No wireframe lines for a solid look
    )

    # Clean up the plot (No axis panes)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)  # Turn off grid for a "floating object" look

    # Optional: Turn off axis ticks completely for a pure render
    # ax.set_axis_off()

    ax.set_title("Viscosity Manifold Topology", fontsize=15)

    # Set initial view
    init_elev = 30
    ax.view_init(elev=init_elev, azim=0)

    # Add Colorbar
    m = cm.ScalarMappable(cmap=cm.plasma, norm=norm)
    m.set_array([])
    cbar = fig.colorbar(m, ax=ax, shrink=0.5, aspect=20, pad=0.05)
    cbar.set_label("Log Mean Viscosity", rotation=270, labelpad=15)

    # C. Animation Loop
    # -----------------------------------------------------
    def update(frame):
        # Rotate the azimuth angle by 'frame' degrees
        # We keep elevation constant at 30
        ax.view_init(elev=init_elev, azim=frame)
        return (fig,)

    # Create the animation object
    # Frames: 0 to 360 degrees, stepping by 2 degrees (180 frames total)
    print("Rendering animation frames...")
    ani = animation.FuncAnimation(
        fig, update, frames=np.arange(0, 360, 2), interval=50  # 50ms per frame = 20 fps
    )

    # Save to GIF
    # 'pillow' is the standard writer for GIFs in matplotlib
    output_file = "manifold_spin.gif"
    ani.save(output_file, writer="pillow", fps=20)

    print(f"Done! Animation saved to {output_file}")

    # Note: plt.show() might not play the animation correctly in all IDEs
    # after saving, so we usually just save it.
