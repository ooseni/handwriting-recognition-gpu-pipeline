"""Turn an ensemble's output into pseudo-labels worth training on.

    ROAD_RUNS=path/to/runs python scripts/pseudo_labels.py ens_4b8b

Self-training on the test images is permitted provided the whole process is automated,
so the filter here is a rule rather than a judgement: keep a test line only when every
JUDGE decoded it identically. Agreement between independently trained models is the
cheapest honest confidence signal there is, and it costs nothing extra.

Judges only, not every run in the pool. A candidate-only source like TrOCR is there
because it is right where the judges are wrong, so it dissents constantly; requiring its
agreement confuses "the strong models are confident" with "the weak model concurs" and
throws away most of the data. On our first ensemble that was the difference between 539
usable lines and 195.

Lines the models disagree on are exactly the ones our labels would be wrong on, so
feeding them back would teach the model its own mistakes. --margin additionally keeps
contested lines where the winner beat the loser by more than that in mean log
probability; the default keeps none of them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from road import paths  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="the ensemble run to read, e.g. ens_4b8b")
    ap.add_argument("--margin", type=float, default=None,
                    help="also keep contested lines won by more than this log-prob margin")
    ap.add_argument("--out", default=None, help="where to write (default <run>/pseudo.csv)")
    args = ap.parse_args(argv)

    run_dir = paths.runs_dir() / args.run
    scores_path, sub_path = run_dir / "scores.csv", run_dir / f"{args.run}.csv"
    for path in (scores_path, sub_path):
        if not path.exists():
            sys.exit(f"{path} is missing -- has {args.run} been combined?")

    scores = pd.read_csv(scores_path, keep_default_na=False, dtype=str)
    sub = pd.read_csv(sub_path, keep_default_na=False, dtype=str)
    judges = [c for c in scores.columns if c not in ("ID", "Target")]

    decoded = {}
    for judge in judges:
        own = paths.runs_dir() / judge / f"{judge}.csv"
        if not own.exists():
            sys.exit(f"{own} is missing -- it holds what {judge} decoded on its own")
        decoded[judge] = pd.read_csv(own, keep_default_na=False,
                                     dtype=str).set_index("ID").Target
    agreed = {i for i in sub.ID
              if all(i in d.index for d in decoded.values())
              and len({d[i] for d in decoded.values()}) == 1}
    keep = sub[sub.ID.isin(agreed)]
    print(f"{len(sub)} test lines; the {len(judges)} judges "
          f"({', '.join(judges)}) agreed on {len(agreed)} "
          f"({len(agreed) / max(len(sub), 1):.0%})")

    if args.margin is not None:
        totals = scores.copy()
        totals[judges] = totals[judges].astype(float)
        totals["total"] = totals[judges].sum(axis=1)
        spread = totals.groupby("ID").total.agg(["max", "min"])
        wide = spread[(spread["max"] - spread["min"]) > args.margin].index
        extra = sub[sub.ID.isin(set(wide) - agreed)]
        print(f"  plus {len(extra)} contested lines won by more than {args.margin}")
        keep = pd.concat([keep, extra], ignore_index=True)

    keep = keep[keep.Target.str.strip().ne("")]
    out = Path(args.out) if args.out else run_dir / "pseudo.csv"
    keep[["ID", "Target"]].to_csv(out, index=False)
    print(f"\n{len(keep)} pseudo-labelled lines -> {out}")
    print(f"  use with: python -m road.qwenvl --run <name> --fold all --pseudo {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
