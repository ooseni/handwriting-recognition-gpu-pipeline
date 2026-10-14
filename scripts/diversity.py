"""How much do these runs actually disagree? Decides whether one is worth adding.

    ROAD_RUNS=path/to/runs python scripts/diversity.py qwen4b_all qwen8b_all trocr_f0

Reads each run's test predictions, so it costs nothing and needs no ground truth.

The ensemble pays for disagreement, not for strength: TrOCR scores 0.07 below either
judge and still won 19% of the lines, while two more Qwen runs that overlap the judges
bought +0.0015 between them. Before spending a session adding a model, this says whether
it would bring anything the pool does not already have.

Two numbers per run matter. **Agreement** with the others says how much it duplicates
them. **Solo** says how often it is the only run proposing its text -- that is the text
the ensemble could not otherwise consider, and where a run earns its place.
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import pandas as pd
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from road import paths  # noqa: E402


def load(run: str) -> pd.Series:
    path = paths.runs_dir() / run / f"{run}.csv"
    if not path.exists():
        sys.exit(f"{path} is missing -- has {run} been decoded?")
    return pd.read_csv(path, keep_default_na=False, dtype=str).set_index("ID").Target


def agreement(a: pd.Series, b: pd.Series, ids: list[str]) -> tuple[float, float]:
    """(fraction of lines identical, mean character distance) between two runs."""
    same = sum(a[i] == b[i] for i in ids)
    dist = sum(Levenshtein.distance(a[i], b[i]) for i in ids)
    return same / len(ids), dist / len(ids)


def solo_rate(run: str, preds: dict[str, pd.Series], ids: list[str]) -> float:
    """How often this run is the only one proposing its text -- the text the ensemble
    could not otherwise consider, and the only reason to add a run to the pool."""
    others = [preds[o] for o in preds if o != run]
    return sum(1 for i in ids if all(preds[run][i] != o[i] for o in others)) / len(ids)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        sys.exit(__doc__)
    preds = {run: load(run) for run in argv}
    ids = sorted(set.intersection(*(set(s.index) for s in preds.values())))
    print(f"{len(ids)} lines shared by {len(preds)} runs\n")

    width = max(len(r) for r in preds) + 2
    print("pairwise agreement (identical text) and mean character distance:")
    for a, b in combinations(preds, 2):
        same, dist = agreement(preds[a], preds[b], ids)
        print(f"  {a:<{width}}{b:<{width}}{same:>6.0%} identical"
              f"   {dist:>5.1f} chars apart")

    print("\nper run, how often it is the only one proposing its text:")
    for run in preds:
        print(f"  {run:<{width}}solo on {solo_rate(run, preds, ids):>4.0%} of lines")
    pool = sum(len({preds[o][i] for o in preds}) for i in ids) / len(ids)
    print(f"\n  candidates per line across the pool: {pool:.2f}")
    print("  a run that is never solo adds nothing the others do not already offer")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
