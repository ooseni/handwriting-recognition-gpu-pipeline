"""The paired comparison used to judge a change on held-out lines."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_runs.py"
spec = importlib.util.spec_from_file_location("compare_runs", SCRIPT)
compare_runs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare_runs)


def errors(word_edits, char_edits):
    n = len(word_edits)
    return pd.DataFrame({"word_edits": word_edits, "words": [10.0] * n,
                         "char_edits": char_edits, "chars": [50.0] * n})


def test_a_run_against_itself_differs_by_nothing():
    e = errors([1.0, 2.0, 0.0, 3.0], [4.0, 6.0, 0.0, 9.0])
    diff, sd, share = compare_runs.paired(e, e, draws=50)
    assert (diff, sd, share) == (0.0, 0.0, 0.0)


def test_a_uniformly_better_run_wins_every_resample():
    worse = errors([2.0] * 20, [8.0] * 20)
    better = errors([1.0] * 20, [4.0] * 20)
    diff, _, share = compare_runs.paired(worse, better, draws=50)
    assert diff > 0 and share == 1.0
