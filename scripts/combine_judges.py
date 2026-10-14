"""Sum every judge's scores over one shared pool of test candidates; write the submission.

    ROAD_RUNS=runs/aws python scripts/combine_judges.py test_matrix_a test_matrix_b \\
        test_matrix_c --out ens_9judge

The scoring was split over several GPUs (cloud/aws/launch.py final), each judging the
same pool, so their scores.csv files are columns of one matrix. This merges them, picks
per line the candidate with the best summed score -- exactly road.ensemble's rule, as
scripts/judge_matrix.py measured it on fold 0 -- and writes the submission through
road.submission, which validates it against Test.csv.

--judges restricts the sum to some of the judges; --against names an earlier
submission to count how many lines the new one changes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_matrix import load_matrix, pick  # noqa: E402
from road import paths, submission  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ensembles", nargs="+", help="ensemble runs that scored the same pool")
    ap.add_argument("--out", required=True)
    ap.add_argument("--judges", nargs="*", default=None, help="default: every judge")
    ap.add_argument("--against", default=None, help="an earlier submission CSV to compare")
    args = ap.parse_args(argv)

    matrix = load_matrix(args.ensembles)
    judges = [c for c in matrix.columns if c not in ("ID", "Target", "candidates", "margin")]
    if args.judges:
        if missing := set(args.judges) - set(judges):
            raise SystemExit(f"not in the matrix: {sorted(missing)}")
        judges = args.judges
    chosen = matrix.loc[pick(matrix, judges)]
    print(f"{chosen.ID.nunique()} lines, {len(matrix)} candidates, "
          f"{len(judges)} judges: {', '.join(judges)}")

    path = submission.write(chosen.ID, chosen.Target, args.out)   # validated, or raises
    out_dir = paths.runs_dir() / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / path.name).write_bytes(path.read_bytes())
    print(f"-> {path}\n-> {out_dir / path.name}")

    if args.against:
        old = pd.read_csv(args.against, keep_default_na=False, dtype=str).set_index("ID").Target
        new = chosen.set_index("ID").Target
        print(f"differs from {Path(args.against).name} on "
              f"{(new != old.loc[new.index]).sum()} of {len(new)} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
