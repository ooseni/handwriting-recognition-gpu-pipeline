"""The selection rules that re-pick a winner from an ensemble's saved scores.

These run on scores road.ensemble already computed, so a rule is a free submission --
which is only worth having if the rule does what it says.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rechoose.py"
spec = importlib.util.spec_from_file_location("rechoose", SCRIPT)
rechoose = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rechoose)

JUDGES = ["a", "b", "c"]


def frame(rows):
    return pd.DataFrame(rows, columns=["ID", "Target", *JUDGES])


def won(picked, line):
    return picked.set_index("ID").Target[line]


def test_sum_adds_the_judges_up():
    # "x" wins on total (-0.9) though "y" is the single best score any judge gives.
    picked = rechoose.pick(frame([
        ("l1", "x", -0.30, -0.30, -0.30),
        ("l1", "y", -0.10, -0.60, -0.70),
    ]), JUDGES, "sum")
    assert won(picked, "l1") == "x"


def test_max_takes_the_single_most_confident_judge():
    picked = rechoose.pick(frame([
        ("l1", "x", -0.30, -0.30, -0.30),
        ("l1", "y", -0.10, -0.60, -0.70),
    ]), JUDGES, "max")
    assert won(picked, "l1") == "y"


def test_vote_lets_the_majority_outweigh_one_confident_judge():
    """The case voting exists for: one judge is wildly confident and wrong, and the
    other two quietly disagree. A sum follows the loud judge; a vote does not."""
    rows = [
        ("l1", "loud", -0.01, -0.90, -0.90),   # judge a alone, and hugely: sum -1.81
        ("l1", "quiet", -1.50, -0.40, -0.40),  # judges b and c, narrowly:  sum -2.30
    ]
    assert won(rechoose.pick(frame(rows), JUDGES, "sum"), "l1") == "loud"
    assert won(rechoose.pick(frame(rows), JUDGES, "vote"), "l1") == "quiet"


def test_vote_falls_back_to_sum_when_every_judge_disagrees():
    rows = [
        ("l1", "x", -0.10, -0.90, -0.90),
        ("l1", "y", -0.90, -0.10, -0.95),
        ("l1", "z", -0.95, -0.95, -0.10),
    ]
    picked = rechoose.pick(frame(rows), JUDGES, "vote")
    assert won(picked, "l1") == "x"          # one vote each; the best total breaks it


def test_rank_ignores_magnitudes_but_keeps_order():
    """b and c prefer 'y' by a hair while a prefers 'x' by a mile. Summing follows the
    magnitude, ranking follows the two-to-one ordering."""
    rows = [
        ("l1", "x", -0.10, -0.52, -0.52),
        ("l1", "y", -2.00, -0.50, -0.50),
    ]
    assert won(rechoose.pick(frame(rows), JUDGES, "sum"), "l1") == "x"
    assert won(rechoose.pick(frame(rows), JUDGES, "rank"), "l1") == "y"


def test_every_line_yields_exactly_one_winner():
    rows = [("l1", "x", -0.1, -0.2, -0.3), ("l1", "y", -0.4, -0.5, -0.6),
            ("l2", "p", -0.7, -0.8, -0.9), ("l2", "q", -0.2, -0.2, -0.2),
            ("l3", "solo", -0.5, -0.5, -0.5)]
    for rule in rechoose.RULES:
        picked = rechoose.pick(frame(rows), JUDGES, rule)
        assert list(picked.ID) == ["l1", "l2", "l3"], rule
        assert won(picked, "l3") == "solo", rule   # an uncontested line is untouched


def test_an_unknown_rule_is_refused():
    with pytest.raises(SystemExit, match="--rule must be"):
        rechoose.pick(frame([("l1", "x", -0.1, -0.1, -0.1)]), JUDGES, "nonsense")
