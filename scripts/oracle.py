"""The ceiling on combining these runs: how good could a perfect chooser be?

    ROAD_RUNS=path/to/runs python scripts/oracle.py qwen4b_f0 trocr_f0 --oof

Needs held-out predictions, since it scores against the truth -- so --oof, and runs that
held a fold out. Builds the candidate pool through road.ensemble.candidates(), so it
sees exactly what the ensemble would, including any nbest.csv a run has written.

The oracle is a ceiling, not a forecast. Ours has converted about 35% of it into real
score, and a wider pool also gives the judges more ways to pick wrong -- ens_3judge lost
0.0016 against a narrower one. What the oracle answers is whether there is anything
there at all: if the pool cannot contain a better answer, no chooser will find one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from road import cv, ensemble, metric, paths  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="finished runs to pool")
    ap.add_argument("--oof", action="store_true", default=True,
                    help="score held-out predictions (the only ones with a truth)")
    args = ap.parse_args(argv)

    truth = cv.load_train().set_index("ID").Target.astype(str)
    pool = ensemble.candidates(args.runs, oof=True)
    ids = [i for i in pool if i in truth.index]
    refs = [truth[i] for i in ids]
    per_line = sum(len(pool[i]) for i in ids) / len(ids)
    print(f"{len(ids)} lines, {per_line:.2f} candidates per line\n")

    best_single = -1.0
    for run in args.runs:
        path = paths.runs_dir() / run / "oof.csv"
        if not path.exists():
            continue
        own = pd.read_csv(path, keep_default_na=False, dtype=str).set_index("ID").Target
        if not all(i in own.index for i in ids):
            continue
        s = metric.public_estimate(metric.line_errors(refs, [own[i] for i in ids]))
        best_single = max(best_single, s)
        print(f"  {run:<20}{s:.4f}")

    picked = [min(pool[i], key=lambda t: Levenshtein.distance(truth[i], t)) for i in ids]
    oracle = metric.public_estimate(metric.line_errors(refs, picked))
    print(f"\n  {'ORACLE':<20}{oracle:.4f}   {oracle - best_single:+.4f} over the best single run")
    print(f"  at the ~35% we have converted before: {best_single + 0.35 * (oracle - best_single):.4f}"
          f"  ({0.35 * (oracle - best_single):+.4f})")

    exact = sum(1 for i, r in zip(ids, refs) if r in pool[i])
    print(f"\n  the pool contains the exact truth on {exact} lines ({exact / len(ids):.0%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
