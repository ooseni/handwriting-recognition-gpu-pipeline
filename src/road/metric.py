"""Competition metric: 0.5 * weighted WER + 0.5 * weighted CER. Lower is better.

Known, from the competition page and starter pack:
  * the two error rates are combined 50/50;
  * "longer reference transcriptions are weighted more heavily";
  * missing, empty or invalid predictions are "penalised as incorrect";
  * the starter pack calls compute_weighted_{wer,cer}(..., weight_factor=0.5) from
    an evaluations/ module it does not ship.

Inferred: each line's error rate is weighted by len(ref) ** weight_factor, with
len counted in words for WER and characters for CER. weight_factor=0 is the plain
mean over lines and 1 the corpus rate (total edits / total reference length);
0.5 sits between. The CV report prints both endpoints, so a conclusion that
depends on the guess is visible.

Normalisation mirrors jiwer's defaults, the likeliest reference implementation:
strip, and for WER split on whitespace. Case and punctuation count -- the ^ ~ &
conventions are scored literally. Per-line rates are not capped at 1.

The leaderboard shows each metric as **mean edits per line** -- confirmed against its
own per-metric columns -- divides each by a reference and averages the two halves:

    public = 0.5 * (1 - word_edits_per_line / LB_WORD_REF)
           + 0.5 * (1 - char_edits_per_line / LB_CHAR_REF)

The references are exactly 12 and 55, not the 11.99 and 55.03 first fitted from our
own submissions. Both reproduce the leaderboard's Benchmark row (11.5289665 word edits,
53.81177457 char edits -> 0.030428445) and our qwen8b_all row to nine decimal places.

aggregate() reports that as `public`, higher is better. In rate terms it comes to
~0.47*WER + 0.56*CER, near the 50/50 the page states -- and one word edit costs as much
leaderboard score as 55/12 = 4.58 character edits.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

WER_WEIGHT = 0.5
CER_WEIGHT = 0.5
WEIGHT_FACTOR = 0.5
# Fitted on the benchmark row (11.53 / 53.81 -> public 0.0304) and our own submission
# (1.906 / 3.592 -> 0.8879); these are Zindi's references, not our scores.
LB_WORD_REF, LB_CHAR_REF = 12.0, 55.0  # exact; see the note above


def _text(s: object) -> str:
    """Non-strings (None, NaN from a CSV) count as an empty prediction."""
    return s.strip() if isinstance(s, str) else ""


def line_errors(refs: Iterable[str], hyps: Iterable[object]) -> pd.DataFrame:
    """Edit counts per line: word_edits, words, char_edits, chars.

    Lengths are the reference's. These four columns are enough to score any
    subset of lines under any weight_factor without recomputing distances.
    """
    rows = []
    for ref, hyp in zip(refs, hyps, strict=True):
        r, h = _text(ref), _text(hyp)
        if not r:
            raise ValueError(f"empty reference {ref!r}: error rate undefined")
        rw, hw = r.split(), h.split()
        rows.append((Levenshtein.distance(rw, hw), len(rw),
                     Levenshtein.distance(r, h), len(r)))
    return pd.DataFrame(rows, columns=["word_edits", "words", "char_edits", "chars"])


def weighted_rate(edits, lengths, weight_factor: float = WEIGHT_FACTOR) -> np.ndarray | float:
    """Mean of per-line rates edits/lengths, weighted by lengths ** weight_factor.

    Reduces over the last axis, so a 2-D array scores many resamples at once.
    """
    e = np.asarray(edits, dtype=float)
    n = np.asarray(lengths, dtype=float)
    if n.shape[-1] == 0:
        raise ValueError("no lines to score")
    w = n**weight_factor
    return (w * (e / n)).sum(-1) / w.sum(-1)


def aggregate(errors: pd.DataFrame, weight_factor: float = WEIGHT_FACTOR) -> dict[str, float]:
    """Score a line_errors() frame, or any subset of its rows."""
    wer = float(weighted_rate(errors.word_edits, errors.words, weight_factor))
    cer = float(weighted_rate(errors.char_edits, errors.chars, weight_factor))
    return {"score": WER_WEIGHT * wer + CER_WEIGHT * cer, "wer": wer, "cer": cer,
            "public": public_estimate(errors)}


def public_estimate(errors: pd.DataFrame, weight_factor: float = WEIGHT_FACTOR) -> float:
    """The leaderboard's own score, higher is better: edits per line over Zindi's
    reference for each metric, averaged 50/50.

    Edits per line is a *length-weighted* mean, not a plain one. The published columns
    cannot be plain means: at nine decimal places, no line count n <= 20000 makes them
    integer sums of integer edit counts, and the Evaluation page says "longer reference
    transcriptions are weighted more heavily". The exact weight is not recoverable --
    the starter pack imports compute_weighted_{wer,cer} from an evaluations/ module it
    never shipped -- so this uses the same len**weight_factor the rest of the module
    does.

    The choice barely matters in practice: across our finished runs, weight_factor from
    0 to 2 moves the estimate by about 0.001 and reorders nothing, which is why the
    earlier plain mean never misled a decision. Held-out and leaderboard orderings do
    disagree, but the gaps there are one to two standard errors of sampling noise, not
    an artefact of the weighting.
    """
    words = errors.words.to_numpy(float) ** weight_factor
    chars = errors.chars.to_numpy(float) ** weight_factor
    per_line = lambda w, e: float((w * e.to_numpy(float)).sum() / w.sum())  # noqa: E731
    return float(0.5 * (1 - per_line(words, errors.word_edits) / LB_WORD_REF)
                 + 0.5 * (1 - per_line(chars, errors.char_edits) / LB_CHAR_REF))


def score(refs: Iterable[str], hyps: Iterable[object],
          weight_factor: float = WEIGHT_FACTOR) -> float:
    """Competition score for aligned references and predictions."""
    return aggregate(line_errors(refs, hyps), weight_factor)["score"]
