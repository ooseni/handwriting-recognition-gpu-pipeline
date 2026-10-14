"""Metric: hand-checked cases, and agreement with jiwer at both ends of the weighting."""

from __future__ import annotations

import random

import jiwer
import pandas as pd
import pytest

from road import metric


def test_perfect_prediction_scores_zero():
    assert metric.score(["To all Christian People"], ["To all Christian People"]) == 0.0


def test_hand_computed_line():
    err = metric.line_errors(["the cat sat"], ["the cat sit"])
    assert err.iloc[0].tolist() == [1, 3, 1, 11]
    assert metric.score(["the cat sat"], ["the cat sit"]) == pytest.approx(0.5 / 3 + 0.5 / 11)


@pytest.mark.parametrize("hyp", ["", "   ", None, float("nan")])
def test_missing_prediction_is_fully_wrong(hyp):
    res = metric.aggregate(metric.line_errors(["heires and assignes"], [hyp]))
    assert res["wer"] == 1.0 and res["cer"] == 1.0


def test_outer_whitespace_ignored_inner_spaces_cost_cer_only():
    ref = "Sealed and delivered"
    assert metric.score([ref], ["  Sealed and delivered "]) == 0.0
    res = metric.aggregate(metric.line_errors([ref], ["Sealed  and delivered"]))
    assert res["wer"] == 0.0 and res["cer"] == pytest.approx(1 / len(ref))


def test_case_and_conventions_are_scored_literally():
    assert metric.line_errors(["y^e said W^m &"], ["ye said Wm and"]).word_edits[0] == 3
    assert metric.line_errors(["The"], ["the"]).char_edits[0] == 1


def test_insertions_can_push_a_line_above_one():
    res = metric.aggregate(metric.line_errors(["protest"], ["protest be it so"]))
    assert res["wer"] == 3.0


def test_empty_reference_is_rejected():
    with pytest.raises(ValueError, match="empty reference"):
        metric.line_errors(["  "], ["x"])


def test_default_weighting_is_sqrt_length():
    # 1-word line fully wrong (rate 1, weight 1); 4-word line right (rate 0, weight 2).
    err = metric.line_errors(["Witness", "in the presence of"], ["Witnes", "in the presence of"])
    assert metric.aggregate(err)["wer"] == pytest.approx(1 / 3)


def _noisy_pairs(n: int = 400, seed: int = 0) -> tuple[list[str], list[str]]:
    """Lines built from the corpus's own conventions, corrupted at character level
    so word and character errors both occur, along with stray and doubled spaces."""
    rng = random.Random(seed)
    vocab = ["To", "all", "Christian", "People", "y^e", "W^m", "&", "heires",
             "assignes", "1640", "protest-", "~", "Sealed", "presence"]
    alphabet = "abcdefghijklmnopqrstuvwxyzABC^~&-. "
    refs, hyps = [], []
    for _ in range(n):
        ref = " ".join(rng.choice(vocab) for _ in range(rng.randint(1, 14)))
        hyp = []
        for ch in ref:
            r = rng.random()
            if r < 0.05:
                continue  # deletion
            hyp.append(rng.choice(alphabet) if r < 0.10 else ch)  # substitution
            if rng.random() < 0.03:
                hyp.append(rng.choice(alphabet))  # insertion
        refs.append(ref)
        hyps.append("".join(hyp))
    return refs, hyps


def test_weight_factor_one_is_jiwer_corpus_rate():
    refs, hyps = _noisy_pairs()
    res = metric.aggregate(metric.line_errors(refs, hyps), weight_factor=1.0)
    assert res["wer"] == pytest.approx(jiwer.wer(refs, hyps))
    assert res["cer"] == pytest.approx(jiwer.cer(refs, hyps))


def test_public_estimate_is_the_leaderboards_own_formula():
    err = metric.line_errors(["the cat sat", "in the presence of"],
                             ["the cat sit", "in the presence of"])
    w = lambda n, e: (n ** metric.WEIGHT_FACTOR * e).sum() / (n ** metric.WEIGHT_FACTOR).sum()
    want = (0.5 * (1 - w(err.words, err.word_edits) / metric.LB_WORD_REF)
            + 0.5 * (1 - w(err.chars, err.char_edits) / metric.LB_CHAR_REF))
    assert metric.aggregate(err)["public"] == pytest.approx(want)
    # one word edit costs about as much leaderboard score as 4.7 character edits
    assert metric.LB_CHAR_REF / metric.LB_WORD_REF == pytest.approx(4.6, abs=0.3)


def test_weight_factor_zero_is_mean_of_jiwer_line_rates():
    refs, hyps = _noisy_pairs()
    res = metric.aggregate(metric.line_errors(refs, hyps), weight_factor=0.0)
    assert res["wer"] == pytest.approx(sum(map(jiwer.wer, refs, hyps)) / len(refs))
    assert res["cer"] == pytest.approx(sum(map(jiwer.cer, refs, hyps)) / len(refs))


@pytest.mark.parametrize("word_edits, char_edits, shown", [
    (11.5289665, 53.81177457, 0.030428445),    # the leaderboard's own Benchmark row
    (1.657704104, 2.697957004, 0.906402113),   # our qwen8b_all submission
])
def test_public_reproduces_real_leaderboard_rows(word_edits, char_edits, shown):
    """Two rows whose three numbers Zindi shows us, so the references are known rather
    than fitted: 11.99/55.03 missed the Benchmark row by 1.3e-4, 12/55 by 2e-10."""
    got = (0.5 * (1 - word_edits / metric.LB_WORD_REF)
           + 0.5 * (1 - char_edits / metric.LB_CHAR_REF))
    assert got == pytest.approx(shown, abs=1e-8)


def test_public_estimate_weights_long_lines_more():
    """The published columns cannot be plain means of integer edit counts -- no line
    count makes them integral at nine decimal places -- and the host says longer
    references weigh more. The weight barely matters, but a plain mean is wrong."""
    errors = pd.DataFrame({"word_edits": [0.0, 4.0], "words": [2.0, 20.0],
                           "char_edits": [0.0, 20.0], "chars": [10.0, 100.0]})
    plain = metric.public_estimate(errors, weight_factor=0.0)
    weighted = metric.public_estimate(errors, weight_factor=1.0)
    assert weighted < plain          # the long, wrong line counts for more
    assert plain == pytest.approx(0.5 * (1 - 2.0 / metric.LB_WORD_REF)
                                  + 0.5 * (1 - 10.0 / metric.LB_CHAR_REF))


def test_published_columns_are_not_plain_means():
    """The evidence for the line above, kept as a test so it cannot be forgotten: our
    own submissions' nine-decimal columns are integral for no plausible line count."""
    published = (1.499572556, 1.498147085, 2.462597361)
    integral = lambda v, n: abs(v * n - round(v * n)) < n * 5e-10  # noqa: E731
    shared = [n for n in range(1, 20001) if all(integral(v, n) for v in published)]
    assert shared == []          # single values coincide by chance; all three never do
