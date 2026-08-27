from pathlib import Path

import pandas as pd
import pytest

from visqai.eval.zero_shot_eval import (
    HELDOUT_PANELS,
    HeldoutPanelAccessError,
    load_heldout_panel,
)


def test_load_without_purpose_raises():
    with pytest.raises(HeldoutPanelAccessError):
        load_heldout_panel("zero_shot_panel")


def test_load_with_wrong_purpose_raises():
    with pytest.raises(HeldoutPanelAccessError):
        load_heldout_panel("zero_shot_panel", purpose="exploratory_eda")


def test_load_unknown_panel_raises_even_with_correct_purpose():
    with pytest.raises(HeldoutPanelAccessError):
        load_heldout_panel("not_a_registered_panel", purpose="final_eval")


@pytest.mark.skipif(
    not HELDOUT_PANELS["zero_shot_panel"].path.exists(),
    reason="zero-shot panel CSV not present in this checkout",
)
def test_load_with_final_eval_purpose_succeeds_and_tags_contaminated():
    df, meta = load_heldout_panel("zero_shot_panel", purpose="final_eval")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert meta["contaminated"] is True
    assert meta["name"] == "zero_shot_panel"


@pytest.mark.skipif(
    not HELDOUT_PANELS["zero_shot_panel"].path.exists(),
    reason="zero-shot panel CSV not present in this checkout",
)
def test_every_access_attempt_is_logged(tmp_path, monkeypatch):
    import visqai.eval.zero_shot_eval as hp

    fake_log = tmp_path / "heldout_access.log"
    monkeypatch.setattr(hp, "ACCESS_LOG", fake_log)

    with pytest.raises(HeldoutPanelAccessError):
        load_heldout_panel("zero_shot_panel")  # denied
    load_heldout_panel("zero_shot_panel", purpose="final_eval")  # granted

    assert fake_log.exists()
    lines = fake_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert "granted=False" in lines[0]
    assert "granted=True" in lines[1]
    assert "zero_shot_panel" in lines[0] and "zero_shot_panel" in lines[1]
    assert "test_heldout_panels" in lines[0]  # calling module recorded
