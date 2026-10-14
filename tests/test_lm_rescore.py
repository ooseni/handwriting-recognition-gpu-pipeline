"""The character LM used to rescore ensemble candidates."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "lm_rescore.py"
spec = importlib.util.spec_from_file_location("lm_rescore", SCRIPT)
lm_rescore = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lm_rescore)

TEXTS = ["the said premises", "in the prsence of", "heires and assignes",
         "the said prsence", "to have and to hold the said premises"]


def test_probabilities_sum_to_one_over_the_alphabet():
    lm = lm_rescore.CharLM(TEXTS, n=4)
    alphabet = sorted({c for t in TEXTS for c in t}) + [lm_rescore.EOS]
    total = sum(lm.prob("\x02\x02th", c) for c in alphabet)
    # Witten-Bell keeps a sliver of mass on the uniform floor, so it is close, not exact
    assert 0.95 < total <= 1.0 + 1e-9


def test_it_prefers_the_spelling_the_transcriptions_use():
    """The reason to add it: the transcriptions spell 'prsence', and the model it is
    correcting writes 'presence'."""
    lm = lm_rescore.CharLM(TEXTS, n=5)
    assert lm.score("in the prsence of") > lm.score("in the presence of")


def test_weight_zero_is_the_plain_ensemble():
    scores = pd.DataFrame({"ID": ["l1", "l1"], "Target": ["loved by judges", "loved by lm"],
                           "a": ["-0.1", "-0.5"], "lm": [-3.0, -0.1]})
    assert lm_rescore.choose(scores, ["a"], 0.0)["l1"] == "loved by judges"
    assert lm_rescore.choose(scores, ["a"], 1.0)["l1"] == "loved by lm"
    assert math.isfinite(lm_rescore.CharLM(TEXTS).score("an unseen word"))
