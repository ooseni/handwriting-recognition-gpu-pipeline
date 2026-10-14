"""List training rows whose label looks wrong, for road.qwenvl --drop.

    ROAD_RUNS=path/to/runs python scripts/suspect_rows.py qwen4b_f0

A held-out line where the model produces fluent text of about the right length that
shares almost nothing with the label is evidence about the label, not the model: it
read the image and the label belongs to a different one. Lines the model merely finds
hard come back short, garbled, or partly right.

The rule is therefore two-sided -- a high edit rate AND a prediction of comparable
length -- and it is applied to held-out predictions only, so no row is judged by a
model that trained on it. The competition allows excluding clearly corrupted training
rows provided the choice is documented; the CSV written here is that documentation.

Coverage is limited to the folds you have held-out predictions for. Pass several runs
to cover more.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from road import cv, metric, paths  # noqa: E402

RATE, LENGTH_LOW, LENGTH_HIGH = 0.6, 0.5, 2.0


def main(argv: list[str]) -> int:
    if not argv:
        sys.exit(__doc__)
    truth = cv.load_train().set_index("ID").Target.astype(str)
    rows = []
    for run in argv:
        path = paths.runs_dir() / run / "oof.csv"
        if not path.exists():
            sys.exit(f"{path} is missing")
        oof = pd.read_csv(path, keep_default_na=False, dtype=str)
        refs = [truth[i] for i in oof.ID]
        err = metric.line_errors(refs, oof.Target.tolist())
        rate = (err.char_edits / err.chars.clip(lower=1)).to_numpy()
        ratio = (oof.Target.str.len() / err.chars.clip(lower=1)).to_numpy()
        flag = (rate > RATE) & (ratio >= LENGTH_LOW) & (ratio <= LENGTH_HIGH)
        print(f"{run}: {flag.sum()} of {len(oof)} held-out lines look mislabelled "
              f"({flag.mean():.1%})")
        for k in flag.nonzero()[0]:
            rows.append({"ID": oof.ID.iloc[k], "run": run,
                         "edit_rate": round(float(rate[k]), 3),
                         "length_ratio": round(float(ratio[k]), 2),
                         "why": "held-out prediction is fluent, full length and unrelated"})

    out = paths.runs_dir() / "suspect_rows.csv"
    frame = pd.DataFrame(rows).drop_duplicates("ID")
    frame.to_csv(out, index=False)
    print(f"\n{len(frame)} rows -> {out}")
    print(f"  use with: python -m road.qwenvl --run <name> --fold all --drop {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
