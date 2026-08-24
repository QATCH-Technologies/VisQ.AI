"""
heldout_panels.py
==================
Registry + load guard for the Fv/whole-antibody charge plan's Task 2:
quarantine datasets that have already been (or easily could be) inspected
during feature/EDA work, so any metric later computed on them is honestly
labelled as an in-sample estimate of that inspection rather than a held-out
validation number.

The zero-shot molecule-discrimination panel (data/processed/zero-shot-data.csv)
is the concrete case this exists for: it was the subject of the EDA that
produced the r=-0.84 Fv-charge correlation finding (later shown, in Task 1's
diagnostics, to be a 3-block contrast rather than a continuous relationship --
see artifacts/fv_charge_diagnostics_report.md). Any feature adopted on the
strength of correlations measured on this panel makes a later "zero-shot"
metric on the same panel an in-sample estimate, not a validation.

Usage:
    from visqai.eval.heldout_panels import load_heldout_panel

    df, meta = load_heldout_panel("zero_shot_panel", purpose="final_eval")
    # meta["contaminated"] is True until a fresh, never-inspected panel
    # replaces this one in the registry. Propagate it into any results row/
    # table computed from `df` -- carry the tag forward, never drop it.

The `purpose="final_eval"` requirement is a deliberate friction point, not a
technical access control: it forces reaching for this panel to be a
conscious, named decision at the callsite. Every call -- granted or denied --
is appended to artifacts/heldout_access.log with a UTC timestamp and the
calling module, so there is a durable audit trail of who touched a
quarantined panel and when.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
ACCESS_LOG = REPO_ROOT / "artifacts" / "heldout_access.log"


@dataclass(frozen=True)
class HeldoutPanel:
    name: str
    path: Path
    contaminated: bool
    note: str


HELDOUT_PANELS: dict[str, HeldoutPanel] = {
    "zero_shot_panel": HeldoutPanel(
        name="zero_shot_panel",
        path=REPO_ROOT / "data" / "processed" / "zero-shot-data.csv",
        contaminated=True,
        note=(
            "46-row molecule-discrimination panel: two engineered variant "
            "families (AB-*/R1-* = R1_family, R2-* = R2_family) plus 8 clinical "
            "mAbs. Includes Tremelimumab (Protein_class_type=mAb_IgG2), an "
            "ISOTYPE CLASS absent from training (train classes: IgG1, IgG4, "
            "Bispecific, Fc-Fusion, Polyclonal, Other) -- flag as unseen-class, "
            "not merely unseen-protein, when reporting results that include it. "
            "Already the subject of extensive EDA (the r=-0.84 pooled Fv-charge "
            "correlation finding, and Task 1's checks 4-5 reproducing it as a "
            "3-block contrast) before this registry existed -- CONTAMINATED. "
            "Any metric computed from this panel is an in-sample estimate of "
            "that EDA, not a held-out validation number, until a fresh, "
            "never-inspected panel exists to replace it."
        ),
    ),
}


class HeldoutPanelAccessError(RuntimeError):
    """Raised when code tries to load a registered held-out panel without an
    explicit purpose="final_eval"."""


def _log_access(name: str, purpose: str | None, granted: bool, caller: str) -> None:
    ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    line = f"{ts}\t{name}\tpurpose={purpose!r}\tgranted={granted}\tcaller={caller}\n"
    with open(ACCESS_LOG, "a", encoding="utf-8") as f:
        f.write(line)


def _calling_module() -> str:
    """Best-effort 'module:function:line' of the first frame outside this
    module, for the access-log audit trail."""
    for frame in inspect.stack()[2:]:
        mod = frame.frame.f_globals.get("__name__", "?")
        if mod != __name__:
            return f"{mod}:{frame.function}:{frame.lineno}"
    return "?"


def load_heldout_panel(name: str, purpose: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Load a registered held-out panel by name.

    Raises HeldoutPanelAccessError unless `purpose="final_eval"` is passed
    explicitly. Every call is logged to artifacts/heldout_access.log
    regardless of outcome.

    Returns (df, meta) where meta always carries `contaminated` -- callers
    must tag it onto any metric/results row derived from `df` and must not
    silently drop it.
    """
    caller = _calling_module()

    if name not in HELDOUT_PANELS:
        _log_access(name, purpose, False, caller)
        raise HeldoutPanelAccessError(f"Unknown held-out panel {name!r}. Registered: {sorted(HELDOUT_PANELS)}")

    panel = HELDOUT_PANELS[name]

    if purpose != "final_eval":
        _log_access(name, purpose, False, caller)
        raise HeldoutPanelAccessError(
            f"Refusing to load held-out panel {name!r} without purpose='final_eval' "
            f"(got purpose={purpose!r}). This panel is quarantined: {panel.note}"
        )

    _log_access(name, purpose, True, caller)
    df = pd.read_csv(panel.path)
    meta = {
        "name": panel.name,
        "contaminated": panel.contaminated,
        "note": panel.note,
        "path": str(panel.path),
    }
    if panel.contaminated:
        logger.warning(
            "Loaded held-out panel %r for purpose=%r -- this panel is CONTAMINATED. "
            "Tag contaminated=True on every metric computed from it and do not "
            "report it as validation evidence. %s",
            name,
            purpose,
            panel.note,
        )
    return df, meta
