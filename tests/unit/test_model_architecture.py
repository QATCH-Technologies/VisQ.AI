"""
Regression gate for the AttentionPool/CrossSampleCNP merge (previously two
independent, hand-synced copies in inference_o_net.py and
train_o_net_v4_rung1.py -- diffed byte-identical before merging into
visqai.models.cnp). torch.save's state_dict is keyed by attribute path, not
import path, so the merge should not break loading any existing checkpoint;
this test proves that against real checkpoints on disk rather than assuming it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from visqai.models.cnp import AttentionPool, CrossSampleCNP

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIRS = [REPO_ROOT / "models" / "experiments", REPO_ROOT / "models" / "production"]


def _discover_checkpoints() -> list[Path]:
    found = []
    for d in CHECKPOINT_DIRS:
        if d.exists():
            found.extend(d.rglob("best_model.pth"))
            found.extend(d.rglob("*.pt"))
    return found


CHECKPOINTS = _discover_checkpoints()


@pytest.mark.skipif(not CHECKPOINTS, reason="no checkpoints found under models/ to validate against")
@pytest.mark.parametrize("ckpt_path", CHECKPOINTS, ids=[str(p.relative_to(REPO_ROOT)) for p in CHECKPOINTS])
def test_merged_cnp_loads_existing_checkpoint_strict(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint or "static_dim" not in checkpoint:
        pytest.skip(f"{ckpt_path} is not a CrossSampleCNP-style checkpoint (missing state_dict/static_dim)")

    config = checkpoint["config"]
    model = CrossSampleCNP(
        static_dim=checkpoint["static_dim"],
        hidden_dim=config["hidden_dim"],
        latent_dim=config["latent_dim"],
        dropout=config["dropout"],
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    static_dim = checkpoint["static_dim"]
    batch, n_ctx, n_q = 2, 5, 3
    ctx = torch.randn(batch, n_ctx, 2 + static_dim)
    q_shear = torch.randn(batch, n_q, 1)
    q_static = torch.randn(batch, n_q, static_dim)

    with torch.no_grad():
        out = model(ctx, q_shear, q_static)
    assert out.shape == (batch, n_q, 1)

    with torch.no_grad():
        memory = model.encode_memory(ctx)
        out2 = model.decode_from_memory(memory, q_shear, q_static)
    assert torch.allclose(out, out2)


def test_attention_pool_output_matches_latent_dim():
    pool = AttentionPool(latent_dim=16, n_heads=4)
    x = torch.randn(3, 5, 16)
    out = pool(x)
    assert out.shape == (3, 16)
