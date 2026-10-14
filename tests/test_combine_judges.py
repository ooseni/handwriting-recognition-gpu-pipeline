"""combine_judges: the sum over judges scored on different machines, as one submission."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from road import paths

ROOT = Path(__file__).resolve().parents[1]
needs_data = pytest.mark.skipif(not (paths.data_dir() / "Test.csv").exists(),
                                reason="dataset not present")


def load(name):
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.remove(str(ROOT / "scripts"))


@needs_data
def test_the_summed_judges_pick_per_line_and_write_a_valid_submission(tmp_path, monkeypatch):
    combine = load("combine_judges")
    monkeypatch.setenv("ROAD_RUNS", str(tmp_path / "runs"))
    monkeypatch.setattr(paths, "submissions_dir", lambda: tmp_path)
    test_ids = list(pd.read_csv(paths.data_dir() / "Test.csv", dtype=str).ID)
    rows = [(i, t) for i in test_ids for t in ("good", "bad")]
    for name, judge, good, bad in (("a", "j1", -0.1, -0.2), ("b", "j2", -0.3, -0.1)):
        d = tmp_path / "runs" / name
        d.mkdir(parents=True)
        pd.DataFrame({"ID": [r[0] for r in rows], "Target": [r[1] for r in rows],
                      judge: [good if t == "good" else bad for _, t in rows]}).to_csv(
            d / "scores.csv", index=False)
    # j1 prefers "good" by 0.1, j2 prefers "bad" by 0.2: the sum picks "bad"
    assert combine.main(["a", "b", "--out", "ens_test"]) == 0
    sub = pd.read_csv(tmp_path / "ens_test.csv", dtype=str, keep_default_na=False)
    assert len(sub) == len(test_ids) and set(sub.Target) == {"bad"}
    assert combine.main(["a", "b", "--out", "ens_j1", "--judges", "j1"]) == 0
    assert set(pd.read_csv(tmp_path / "ens_j1.csv", dtype=str).Target) == {"good"}
    assert (tmp_path / "runs" / "ens_test" / "ens_test.csv").exists()
