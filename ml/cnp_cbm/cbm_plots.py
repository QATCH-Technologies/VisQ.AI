import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_cbm_partial_dependence(
    model, device, concept_names, target_concept="hydrophobicity", baseline_val=0.0
):
    """
    Plots the Partial Dependence of the decoder's log-viscosity output
    with respect to a single physical concept, sweeping across different shear rates.
    """
    model.eval()

    # Identify the index of the concept we want to sweep
    if target_concept not in concept_names:
        print(f"Concept {target_concept} not found!")
        return
    c_idx = concept_names.index(target_concept)

    # Define shear rates to test (100 to 10M s^-1)
    shear_rates = torch.tensor([1e2, 1e3, 1e4, 1e5, 1e6, 1e7], dtype=torch.float32, device=device)
    log_shear = torch.log10(shear_rates).unsqueeze(1)  # Shape: [6, 1]

    # Sweep the target concept from -1.0 to 1.0 (typical range for tanh/sigmoid normalized)
    concept_sweep = torch.linspace(-1.0, 1.0, steps=50, device=device)

    results = []

    with torch.no_grad():
        for shear_val in log_shear:
            # Create a batch of concepts all set to baseline (e.g., 0.0)
            c_batch = torch.full(
                (len(concept_sweep), model.n_concepts), baseline_val, device=device
            )
            # Override the target concept with our sweep values
            c_batch[:, c_idx] = concept_sweep

            # Create query matrix for the decoder: [shear | static_context (concepts)]
            # Note: Depending on your exact decoder signature, you pass the shear and concepts.
            # Assuming CNP decoder takes [query_x, representation] where query_x is shear and representation is c
            shear_batch = shear_val.repeat(len(concept_sweep), 1)

            # Forward pass through decoder ONLY
            # (Adjust this line if your decoder expects a different input shape)
            pred_log_visc = model.decoder(shear_batch, c_batch)

            for i in range(len(concept_sweep)):
                results.append(
                    {
                        "Concept_Value": concept_sweep[i].item(),
                        "Shear_Rate": (10 ** shear_val.item()),
                        "Pred_Log_Viscosity": pred_log_visc[i].item(),
                    }
                )

    df_pdp = pd.DataFrame(results)

    # Plotting
    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=df_pdp,
        x="Concept_Value",
        y="Pred_Log_Viscosity",
        hue="Shear_Rate",
        palette="viridis",
        linewidth=2,
    )
    plt.title(f"Decoder Partial Dependence: {target_concept}")
    plt.xlabel(f"{target_concept} Activation")
    plt.ylabel("Predicted Log10(Viscosity)")
    plt.legend(title="Shear Rate ($s^{-1}$)", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"./models/experiments/cbm_cnp_v3/pdp_{target_concept}.png", dpi=300)
    plt.show()


def plot_concept_vs_proxy_scatter(model, dataloader, device, concept_names, proxy_mapping):
    """
    Extracts the learned concept activations and plots them against the ground-truth proxy features
    to visualize the non-linear mappings the encoder learned.

    proxy_mapping: dict mapping concept_name -> proxy_feature_index_in_batch
    """
    model.eval()

    all_activations = []
    all_proxies = []

    with torch.no_grad():
        for batch in dataloader:
            # Unpack your batch (adjust according to your actual dataloader structure)
            context_x, context_y, query_x, query_y, static_feats, group_idx = [
                b.to(device) for b in batch
            ]

            # Get concept activations from the encoder
            # (Assuming model.encode returns the deterministic representation r, which we project)
            r = model.encoder(context_x, context_y, static_feats)
            raw_concepts = model.concept_proj(r)

            # Apply the activation functions (tanh/sigmoid) as defined in your CONCEPT_DEFS
            # For simplicity here, we'll just grab the raw outputs, but you can apply torch.tanh/torch.sigmoid
            # based on how your forward() pass is structured.
            activations = torch.tanh(raw_concepts)  # Adjust if some are sigmoid!

            all_activations.append(activations.cpu().numpy())
            all_proxies.append(static_feats.cpu().numpy())

    all_activations = np.concatenate(all_activations, axis=0)
    all_proxies = np.concatenate(all_proxies, axis=0)

    # Create subplots for each mapped concept
    n_plots = len(proxy_mapping)
    cols = 3
    rows = (n_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    for i, (concept_name, proxy_idx) in enumerate(proxy_mapping.items()):
        if concept_name not in concept_names:
            continue

        c_idx = concept_names.index(concept_name)

        c_vals = all_activations[:, c_idx]
        p_vals = all_proxies[:, proxy_idx]

        # Calculate Pearson R for the title
        r_val = np.corrcoef(c_vals, p_vals)[0, 1]

        ax = axes[i]
        sns.scatterplot(x=p_vals, y=c_vals, alpha=0.5, ax=ax, color="teal")
        ax.set_title(f"{concept_name}\n(r = {r_val:.3f})")
        ax.set_xlabel("Engineered Proxy Value (Normalized)")
        ax.set_ylabel("Learned Concept Activation")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig("./models/experiments/cbm_cnp_v3/concept_proxy_scatter.png", dpi=300)
    plt.show()


# --- 1. Run the PDP for a specific concept ---
# Let's look at how hydrophobicity and crowding drive the non-Newtonian curves
print("Generating Partial Dependence Plots...")
plot_cbm_partial_dependence(
    model=model, device=device, concept_names=concept_names, target_concept="hydrophobicity"
)

plot_cbm_partial_dependence(
    model=model, device=device, concept_names=concept_names, target_concept="crowding"
)

# --- 2. Run the Scatterplots ---
print("Generating Concept vs. Proxy Scatterplots...")

# You need to map the concept name to the index of its proxy in your static_feats tensor.
# You can usually find these indices by looking at your feature_names list.
proxy_mapping = {
    "self_interaction": feature_names.index("num__conc_x_kP"),
    "hydrophobicity": feature_names.index("num__conc_x_HCI"),
    "charge_environment": feature_names.index("num__charge_x_ionic"),
    "ionic_screening": feature_names.index("num__Ionic_Strength_Proxy"),
    "crowding": feature_names.index("num__Protein_conc"),
    "nonlinear_conc": feature_names.index("num__conc_sq"),
    "cosolute_interaction": feature_names.index("num__Stabilizer_mg_mL"),
    "cosolute_protection": feature_names.index("num__Crowding_Index"),
}

# Pass the full training dataloader so we get a rich scatter plot
plot_concept_vs_proxy_scatter(
    model=model,
    dataloader=train_loader,  # or val_loader
    device=device,
    concept_names=concept_names,
    proxy_mapping=proxy_mapping,
)
