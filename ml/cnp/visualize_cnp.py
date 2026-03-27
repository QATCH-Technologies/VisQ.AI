"""
visualize_cnp.py
================
Generates architecture visualizations of CrossSampleCNP using three methods:

  1. torchviz  — computation graph from a real forward pass
  2. netron    — interactive layer browser (opens in browser, or exports ONNX)
  3. matplotlib + networkx — clean publication-style diagram

Usage:
  python visualize_cnp.py --all
  python visualize_cnp.py --torchviz
  python visualize_cnp.py --netron
  python visualize_cnp.py --networkx
  python visualize_cnp.py --model /path/to/best_model.pth   # use real weights

Outputs (written to ./viz_outputs/):
  cnp_torchviz.pdf      torchviz computation graph
  cnp_model.onnx        ONNX export for netron (or pass --netron to auto-open)
  cnp_networkx.png      matplotlib+networkx diagram
"""

import argparse
import os
import sys
import warnings

# ── Windows/Anaconda: suppress duplicate OpenMP runtime error ─────────────────
# Multiple conda packages (pytorch, numpy, scipy) can each bundle libiomp5md.dll.
# Setting this env var before any import prevents the OMP Error #15 crash.
# Must be set before numpy/torch are imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Suppress the TorchScript ONNX deprecation warning (informational only)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch.onnx")

import torch
import torch.nn as nn

# ─────────────────────────────────────────────
# Model definition (mirrors train_cnp_3.py)
# ─────────────────────────────────────────────


class AttentionPool(nn.Module):
    def __init__(self, latent_dim, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(latent_dim, n_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, latent_dim))
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        q = self.query.expand(x.size(0), -1, -1)
        out, _ = self.attn(q, x, x)
        return self.norm(out.squeeze(1))


class CrossSampleCNP(nn.Module):
    def __init__(self, static_dim, hidden_dim=128, latent_dim=128, dropout=0.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2 + static_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.pooler = AttentionPool(latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(1 + static_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, context_tensor, query_shear, query_static):
        encoded = self.encoder(context_tensor)
        r = self.pooler(encoded)
        n_queries = query_shear.size(1)
        r_expanded = r.unsqueeze(1).repeat(1, n_queries, 1)
        decoder_input = torch.cat([query_shear, query_static, r_expanded], dim=-1)
        return self.decoder(decoder_input)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

OUT_DIR = "./viz_outputs"
os.makedirs(OUT_DIR, exist_ok=True)


def load_model(model_path=None, static_dim=32, hidden_dim=128, latent_dim=128):
    """Load from checkpoint if given, else build a fresh model with dummy weights.

    Handles multiple checkpoint formats:
      - train_cnp_3.py  ->  keys: 'state_dict', 'config', 'static_dim'
      - older scripts   ->  keys: 'model_state_dict', 'config', 'static_dim'
      - bare state dict ->  top-level keys are parameter names
    """
    if model_path and os.path.exists(model_path):
        print(f"[model] Loading checkpoint: {model_path}")
        ckpt = torch.load(model_path, map_location="cpu")

        # Show what's actually in the file so key mismatches are easy to debug
        if isinstance(ckpt, dict):
            print(f"[model] Checkpoint keys: {list(ckpt.keys())}")
        else:
            print(f"[model] Checkpoint is a raw state_dict (OrderedDict / dict of tensors)")

        # ── Resolve state dict ──────────────────────────────────────────────
        STATE_DICT_KEYS = ("state_dict", "model_state_dict", "model")
        if isinstance(ckpt, dict) and not any(k in ckpt for k in STATE_DICT_KEYS):
            # Assume the checkpoint IS the state dict (bare save)
            state_dict = ckpt
        else:
            found = next((k for k in STATE_DICT_KEYS if k in ckpt), None)
            if found is None:
                raise KeyError(
                    f"Cannot find a state dict in checkpoint.\n"
                    f"Available keys: {list(ckpt.keys())}\n"
                    f"Expected one of: {STATE_DICT_KEYS}"
                )
            state_dict = ckpt[found]

        # ── Resolve hyperparameters ─────────────────────────────────────────
        if isinstance(ckpt, dict):
            static_dim = ckpt.get("static_dim", static_dim)
            cfg = ckpt.get("config", {}) or {}
            hidden_dim = cfg.get("hidden_dim", hidden_dim)
            latent_dim = cfg.get("latent_dim", latent_dim)

        model = CrossSampleCNP(static_dim, hidden_dim, latent_dim)
        model.load_state_dict(state_dict)
        print(f"[model] Loaded  static_dim={static_dim}  hidden={hidden_dim}  latent={latent_dim}")
    else:
        if model_path:
            print(
                f"[model] WARNING: {model_path} not found — using untrained model (static_dim={static_dim})"
            )
        else:
            print(f"[model] No checkpoint given — using untrained model (static_dim={static_dim})")
        model = CrossSampleCNP(static_dim, hidden_dim, latent_dim)

    model.eval()
    return model, static_dim, latent_dim


def make_dummy_inputs(static_dim, batch=1, n_ctx=8, n_q=5):
    ctx = torch.randn(batch, n_ctx, 2 + static_dim)
    shear = torch.randn(batch, n_q, 1)
    static = torch.randn(batch, n_q, static_dim)
    return ctx, shear, static


# ─────────────────────────────────────────────
# 1. torchviz
# ─────────────────────────────────────────────


def run_torchviz(model, static_dim):
    try:
        from torchviz import make_dot
    except ImportError:
        print("[torchviz] Not installed — run:  pip install torchviz")
        return

    print("[torchviz] Running forward pass...")
    ctx, shear, static = make_dummy_inputs(static_dim)
    out = model(ctx, shear, static)

    dot = make_dot(
        out,
        params=dict(model.named_parameters()),
        show_attrs=True,
        show_saved=False,
    )
    dot.attr(rankdir="TB", bgcolor="white", fontcolor="#111111", fontname="Helvetica")
    dot.attr(
        "node",
        style="filled,rounded",
        shape="box",
        fillcolor="#f0f4ff",
        fontcolor="#111111",
        fontname="Helvetica",
        fontsize="10",
        color="#4a6fa5",
        penwidth="1.2",
    )
    dot.attr("edge", color="#555555", arrowsize="0.7")

    out_path = os.path.join(OUT_DIR, "cnp_torchviz")

    # Always save the raw .dot source — works even without graphviz installed
    dot_src_path = out_path + ".dot"
    dot.save(dot_src_path)
    print(f"[torchviz] .dot source saved -> {dot_src_path}")

    # Try to render to PDF — requires graphviz system executable ('dot' on PATH)
    try:
        dot.render(out_path, format="pdf", cleanup=True)
        print(f"[torchviz] PDF saved -> {out_path}.pdf")
        print("           Note: torchviz shows every tensor op — great for grad debugging,")
        print("           but typically too dense for presentations.")
    except Exception as e:
        print(f"[torchviz] WARNING: Could not render PDF — {e}")
        print()
        print("           The Graphviz system executable ('dot') is not on your PATH.")
        print("           Fix options:")
        print("             conda:   conda install -c conda-forge graphviz")
        print("             winget:  winget install graphviz.graphviz  (restart terminal after)")
        print("             manual:  https://graphviz.org/download/ — check 'Add to PATH'")
        print()
        print(f"           Once installed, render manually:")
        print(f"             dot -Tpdf {dot_src_path} -o {out_path}.pdf")
        print(f"           Or open {dot_src_path} at https://dreampuf.github.io/GraphvizOnline")


# ─────────────────────────────────────────────
# 2. netron  (ONNX export + optional browser open)
# ─────────────────────────────────────────────


def run_netron(model, static_dim, open_browser=True):
    onnx_path = os.path.join(OUT_DIR, "cnp_model.onnx")
    ctx, shear, static_in = make_dummy_inputs(static_dim)

    print("[netron] Exporting to ONNX...")
    torch.onnx.export(
        model,
        (ctx, shear, static_in),
        onnx_path,
        input_names=["context_tensor", "query_shear", "query_static"],
        output_names=["log_viscosity"],
        dynamic_axes={
            "context_tensor": {0: "batch", 1: "n_context"},
            "query_shear": {0: "batch", 1: "n_queries"},
            "query_static": {0: "batch", 1: "n_queries"},
            "log_viscosity": {0: "batch", 1: "n_queries"},
        },
        opset_version=14,
    )
    print(f"[netron] ONNX saved -> {onnx_path}")

    if open_browser:
        try:
            import netron

            print("[netron] Opening in browser at http://localhost:8080 ...")
            print("         Press Ctrl+C to stop the server.")
            netron.start(onnx_path)  # blocks until Ctrl+C
        except ImportError:
            print("[netron] netron not installed — run:  pip install netron")
            print(f"         Then:  netron {onnx_path}")
            print("         Or drag-drop the .onnx file to https://netron.app")
    else:
        print(f"[netron] To view:  netron {onnx_path}")
        print("         Or drag-drop to https://netron.app")


# ─────────────────────────────────────────────
# 3. matplotlib + networkx
# ─────────────────────────────────────────────


def run_networkx():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        print("[networkx] Missing deps — run:  pip install matplotlib networkx")
        return

    print("[networkx] Building graph...")

    # ── Node definitions ────────────────────────────────────────────────────
    nodes = [
        # id                  display label               group       col row
        ("ctx_in", "Context Input\n(c, logη, static)", "input", 0, 0),
        ("enc", "Encoder MLP\n3xLinear+ReLU", "encoder", 0, 1),
        ("pool", "MultiheadAttention\nn_heads=4", "pooler", 0, 2),
        ("ln", "LayerNorm\n-> r", "pooler", 0, 3),
        ("qry_in", "Query Input\n(shear, static)", "input", 2, 1),
        ("cat", "Concatenate\nshear ‖ static ‖ r", "decoder", 1, 4),
        ("dec", "Decoder MLP\n3xLinear+ReLU", "decoder", 1, 5),
        ("out", "log η̂\n(viscosity)", "output", 1, 6),
    ]

    edges = [
        ("ctx_in", "enc"),
        ("enc", "pool"),
        ("pool", "ln"),
        ("ln", "cat"),
        ("qry_in", "cat"),
        ("cat", "dec"),
        ("dec", "out"),
    ]

    group_colors = {
        "input": "#1e2433",
        "encoder": "#1a3a5c",
        "pooler": "#1a3d2e",
        "decoder": "#3a1f4a",
        "output": "#3a2a10",
    }
    group_edge_colors = {
        "input": "#64748b",
        "encoder": "#3b82f6",
        "pooler": "#10b981",
        "decoder": "#a855f7",
        "output": "#f59e0b",
    }

    COL_SPACING = 3.5
    ROW_SPACING = 2.2

    pos = {}
    for nid, _, _, col, row in nodes:
        pos[nid] = (col * COL_SPACING, -row * ROW_SPACING)

    G = nx.DiGraph()
    for nid, label, group, col, row in nodes:
        G.add_node(nid, label=label, group=group)
    G.add_edges_from(edges)

    fig, ax = plt.subplots(figsize=(13, 11), facecolor="#0a0e1a")
    ax.set_facecolor("#0a0e1a")

    # Draw per-group so we can colour nodes and borders separately
    for nid, _, group, _, _ in nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[nid],
            ax=ax,
            node_size=4200,
            node_color=group_colors[group],
            edgecolors=group_edge_colors[group],
            linewidths=2.0,
            node_shape="s",  # square
        )

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edge_color="#60a5fa",
        arrows=True,
        arrowsize=22,
        arrowstyle="-|>",
        width=2.0,
        alpha=0.7,
        connectionstyle="arc3,rad=0.05",
        min_source_margin=38,
        min_target_margin=38,
    )

    labels = {nid: label for nid, label, _, _, _ in nodes}
    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        ax=ax,
        font_color="#e2e8f0",
        font_size=8.5,
        font_family="monospace",
        font_weight="bold",
    )

    # Legend
    legend_handles = [
        mpatches.Patch(
            facecolor=group_colors[g],
            edgecolor=group_edge_colors[g],
            linewidth=2,
            label=g.capitalize(),
        )
        for g in ["input", "encoder", "pooler", "decoder", "output"]
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        facecolor="#111827",
        edgecolor="#334155",
        labelcolor="#e2e8f0",
        fontsize=9,
        framealpha=0.9,
    )

    ax.set_title(
        "VisQ.AI Model Architecture",
        color="#e2e8f0",
        fontsize=13,
        fontweight="bold",
        fontfamily="monospace",
        pad=16,
    )
    ax.axis("off")
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, "cnp_networkx.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="#0a0e1a")
    plt.close()
    print(f"[networkx] Saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="CNP architecture visualizations")
    parser.add_argument(
        "--model",
        default="models/experiments/o_net_v3_debug_aug/best_model.pth",
        help="Path to best_model.pth checkpoint (optional)",
    )
    parser.add_argument(
        "--static-dim",
        type=int,
        default=32,
        help="static_dim if no checkpoint given (default: 32)",
    )
    parser.add_argument("--all", action="store_true", help="Run all methods")
    parser.add_argument("--torchviz", action="store_true")
    parser.add_argument("--netron", action="store_true")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Export ONNX but don't open browser (for --netron)",
    )
    parser.add_argument("--networkx", action="store_true")
    args = parser.parse_args()

    # Default to --all if nothing specified
    if not any([args.all, args.torchviz, args.netron, args.networkx]):
        args.all = True

    model, static_dim, latent_dim = load_model(args.model, args.static_dim)

    print(f"\nOutputs will be written to: {os.path.abspath(OUT_DIR)}\n")

    if args.all or args.torchviz:
        print("─" * 50)
        run_torchviz(model, static_dim)

    if args.all or args.netron:
        print("─" * 50)
        open_browser = not args.no_browser
        run_netron(model, static_dim, open_browser=open_browser)

    if args.all or args.networkx:
        print("─" * 50)
        run_networkx()

    print("\n─" * 50)
    print(f"Done. Check {os.path.abspath(OUT_DIR)}/")


if __name__ == "__main__":
    main()
