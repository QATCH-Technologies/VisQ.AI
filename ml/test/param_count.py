"""
count_model_params.py
=====================
Loads a CrossSampleCNP .pt checkpoint and reports the total and per-layer
parameter counts without needing the original training script.

Usage
-----
    python count_model_params.py path/to/model.pt [--static-dim N]

Arguments
---------
    model_path   : Path to the .pt checkpoint saved by train_cnp_3.py.
    --static-dim : (Optional) Override static_dim if it cannot be inferred
                   from the checkpoint.  Defaults to auto-detection.

The script handles two common save formats:
  1. Full model object  : torch.save(model, path)
  2. State-dict bundle  : torch.save({"model_state": ..., "static_dim": ...}, path)
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────────────────────────
# Model definition (must match train_cnp_3.py exactly)
# ──────────────────────────────────────────────────────────────────────────────


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


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def count_params(model: nn.Module) -> dict:
    """Return total, trainable, and per-named-module parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    per_module = {}
    for name, module in model.named_modules():
        if name == "":
            continue
        n = sum(p.numel() for p in module.parameters(recurse=False))
        if n > 0:
            per_module[name] = n
    return {"total": total, "trainable": trainable, "per_module": per_module}


def infer_static_dim_from_state(state: dict) -> int | None:
    """Try to derive static_dim from the first encoder weight shape."""
    key = "encoder.0.weight"
    if key in state:
        # shape is [hidden_dim, 2 + static_dim]
        in_features = state[key].shape[1]
        return in_features - 2
    return None


def infer_dims_from_state(state: dict):
    """Return (static_dim, hidden_dim, latent_dim) if inferrable."""
    enc_key = "encoder.0.weight"  # [hidden, 2+static]
    lat_key = "encoder.5.weight"  # [latent, hidden]  (index 5 in Sequential)
    dec_key = "decoder.0.weight"  # [hidden, 1+static+latent]

    if enc_key not in state or lat_key not in state or dec_key not in state:
        # Try alternate key names (Sequential uses numeric indices)
        # encoder: 0=Linear, 1=ReLU, 2=Dropout, 3=Linear, 4=ReLU, 5=Linear
        lat_key = "encoder.5.weight"

    hidden_dim = state[enc_key].shape[0] if enc_key in state else None
    latent_dim = state[lat_key].shape[0] if lat_key in state else None
    static_dim = (state[enc_key].shape[1] - 2) if enc_key in state else None

    return static_dim, hidden_dim, latent_dim


def load_checkpoint(path: str, static_dim_override: int | None = None):
    """Load checkpoint and return a reconstructed model."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # ── Case 1: already a fully-instantiated nn.Module ──
    if isinstance(ckpt, nn.Module):
        print(f"  Checkpoint format : full model object")
        return ckpt

    # ── Case 2: dict — find the state dict inside ──
    state = None
    meta = {}
    if isinstance(ckpt, dict):
        for candidate in ("model_state", "state_dict", "model"):
            if candidate in ckpt:
                state = ckpt[candidate]
                meta = {k: v for k, v in ckpt.items() if k != candidate}
                print(f"  Checkpoint format : dict with key '{candidate}'")
                break
        if state is None:
            # Assume the dict itself is the state dict
            state = ckpt
            print(f"  Checkpoint format : raw state dict")

    if state is None:
        raise ValueError("Cannot locate a state dict in the checkpoint.")

    # ── Infer architecture dims ──
    sd, hd, ld = infer_dims_from_state(state)
    static_dim = static_dim_override or meta.get("static_dim") or sd
    hidden_dim = meta.get("hidden_dim") or hd or 128
    latent_dim = meta.get("latent_dim") or ld or 128
    dropout = meta.get("dropout", 0.0)

    if static_dim is None:
        raise ValueError(
            "Cannot infer static_dim from checkpoint. "
            "Pass --static-dim N explicitly."
        )

    print(
        f"  Inferred dims     : static_dim={static_dim}, "
        f"hidden_dim={hidden_dim}, latent_dim={latent_dim}"
    )

    model = CrossSampleCNP(
        static_dim=static_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        dropout=dropout,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  Warning: missing keys  : {missing}")
    if unexpected:
        print(f"  Warning: unexpected keys: {unexpected}")

    return model


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Count parameters in a CrossSampleCNP .pt file."
    )
    model_path = "models/experiments/o_net_v3_debug_aug/best_model.pth"
    parser.add_argument(
        "--static-dim",
        type=int,
        default=None,
        help="Override static_dim if auto-detection fails.",
    )
    args = parser.parse_args()

    path = Path(model_path)
    if not path.exists():
        print(f"Error: file not found — {path}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  CrossSampleCNP Parameter Counter")
    print(f"{'='*60}")
    print(f"  File: {path}")
    print(f"  Size: {path.stat().st_size / 1024:.1f} KB")

    model = load_checkpoint(str(path), args.static_dim)
    info = count_params(model)

    print(f"\n{'─'*60}")
    print(f"  {'Total parameters':<35} {info['total']:>12,}")
    print(f"  {'Trainable parameters':<35} {info['trainable']:>12,}")
    print(f"{'─'*60}")
    print(f"  Per-submodule breakdown:")
    print(f"  {'Module':<38} {'Params':>10}")
    print(f"  {'─'*38} {'─'*10}")
    for name, n in info["per_module"].items():
        print(f"  {name:<38} {n:>10,}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
