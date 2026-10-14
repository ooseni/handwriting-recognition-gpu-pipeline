"""Re-pick the winning candidate from an ensemble's scores, without a GPU.

    ROAD_RUNS=path/to/runs python scripts/rechoose.py ens_3judge --rule vote

road.ensemble already scored every candidate under every judge and wrote scores.csv, so
the choice itself can be redone offline. A different rule over the same numbers is a
free submission: no forward pass, no training, only a submission slot.

Rules:
  sum    each judge's mean log-probability added up (what road.ensemble does)
  vote   each judge votes for the candidate it likes best, majority wins, ties fall
         back to sum. Needs three judges to mean anything, and is robust to one badly
         calibrated judge in a way that summing is not -- a judge that is confidently
         wrong drags a sum, but carries only one vote.
  max    the candidate any single judge likes most; the least robust of the three, and
         here mostly as a control.
  rank   each judge ranks the candidates and the ranks are added, which keeps the
         ordering each judge implies while discarding the magnitudes it disagrees on.

Writes a validated submission and prints how far the new pick differs from the old, so
a rule that changes nothing can be dropped before it costs a submission.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from road import paths, submission  # noqa: E402

RULES = ("sum", "vote", "max", "rank")


def pick(scores: pd.DataFrame, judges: list[str], rule: str) -> pd.DataFrame:
    """One row per line: the winning candidate under this rule."""
    s = scores.copy()
    s[judges] = s[judges].astype(float)

    if rule == "sum":
        s["key"] = s[judges].sum(axis=1)
    elif rule == "max":
        s["key"] = s[judges].max(axis=1)
    elif rule == "rank":
        # rank within each line, per judge, then add: magnitudes dropped, order kept
        s["key"] = sum(s.groupby("ID")[j].rank(method="average") for j in judges)
    elif rule == "vote":
        votes = pd.Series(0, index=s.index, dtype=float)
        for j in judges:
            winners = s.loc[s.groupby("ID", sort=False)[j].idxmax()].index
            votes.loc[winners] += 1
        # sum breaks ties, scaled small so it can never outweigh a vote
        total = s[judges].sum(axis=1)
        spread = (total.max() - total.min()) or 1.0
        s["key"] = votes + (total - total.min()) / spread * 0.5
    else:
        raise SystemExit(f"--rule must be one of {', '.join(RULES)}")

    return s.loc[s.groupby("ID", sort=False).key.idxmax()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="the ensemble run whose scores.csv to re-read")
    ap.add_argument("--rule", default="vote", choices=RULES)
    ap.add_argument("--out", default=None, help="submission name (default <run>_<rule>)")
    args = ap.parse_args(argv)

    run_dir = paths.runs_dir() / args.run
    scores_path = run_dir / "scores.csv"
    if not scores_path.exists():
        sys.exit(f"{scores_path} is missing -- has {args.run} been combined?")
    scores = pd.read_csv(scores_path, keep_default_na=False, dtype=str)
    judges = [c for c in scores.columns if c not in ("ID", "Target")]
    print(f"{len(scores)} candidates over {scores.ID.nunique()} lines, "
          f"judged by {len(judges)}: {', '.join(judges)}")
    if args.rule == "vote" and len(judges) < 3:
        print(f"  note: voting with {len(judges)} judges is mostly ties, "
              f"so this will land close to --rule sum", flush=True)

    picked = pick(scores, judges, args.rule)
    out = args.out or f"{args.run}_{args.rule}"

    previous = run_dir / f"{args.run}.csv"
    if previous.exists():
        old = pd.read_csv(previous, keep_default_na=False, dtype=str).set_index("ID").Target
        new = picked.set_index("ID").Target
        shared = [i for i in old.index if i in new.index]
        changed = sum(old[i] != new[i] for i in shared)
        print(f"  {args.rule}: {changed} of {len(shared)} lines differ from {args.run} "
              f"({changed / max(len(shared), 1):.0%})")
        if changed == 0:
            print("  identical to what was already submitted -- not worth a slot")

    path = submission.write(picked.ID, picked.Target, out)
    out_dir = paths.runs_dir() / out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / path.name).write_bytes(path.read_bytes())
    print(f"\n{len(picked)} lines -> {out_dir / path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
