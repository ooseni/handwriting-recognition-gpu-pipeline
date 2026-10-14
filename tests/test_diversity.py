"""Agreement and solo rate: the two numbers that decide whether a run is worth adding."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diversity.py"
spec = importlib.util.spec_from_file_location("diversity", SCRIPT)
diversity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diversity)

IDS = ["l1", "l2", "l3", "l4"]


def series(*texts):
    return pd.Series(list(texts), index=IDS)


def test_agreement_counts_identical_lines_and_distance():
    a = series("the said", "heires", "ye", "premises")
    b = series("the said", "heirs", "yt", "premises")
    same, dist = diversity.agreement(a, b, IDS)
    assert same == 0.5                       # l1 and l4 match
    assert dist == (0 + 1 + 1 + 0) / 4       # heires->heirs drops one, ye->yt swaps one


def test_solo_rate_counts_only_lines_nobody_else_proposes():
    preds = {
        "a": series("x", "same", "same", "p"),
        "b": series("y", "same", "same", "p"),
        "c": series("z", "same", "other", "p"),
    }
    # l1: all three differ, so each is solo. l2: all identical, nobody solo.
    # l3: c alone differs. l4: all identical.
    assert diversity.solo_rate("a", preds, IDS) == 0.25
    assert diversity.solo_rate("b", preds, IDS) == 0.25
    assert diversity.solo_rate("c", preds, IDS) == 0.5


def test_a_duplicate_run_is_never_solo():
    """The case the script exists to catch: a run that adds nothing to the pool."""
    preds = {"a": series("w", "x", "y", "z"),
             "copy": series("w", "x", "y", "z"),
             "other": series("w", "different", "y", "z")}
    assert diversity.solo_rate("a", preds, IDS) == 0.0
    assert diversity.solo_rate("copy", preds, IDS) == 0.0
    assert diversity.solo_rate("other", preds, IDS) == 0.25
