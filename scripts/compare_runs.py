"""Is run B better than run A on the same held-out lines, and where?

    ROAD_RUNS=path/to/runs python scripts/compare_runs.py qwen4b_f0 qwen4b_f0_jitter

A paired comparison: both runs decoded the same fold, so every line is scored twice and
the noise is in the difference, not in each score. That is far tighter than the unpaired
sd 0.0048 on 820 lines -- reading a paired result against the unpaired threshold is how
we once dismissed a real +0.0046 as noise.

It also splits the difference by scan brightness. A change aimed at the dark scans that
helps every line equally is working for some other reason than the one it was built for:
--tall-weight "helped" both scan batches, which pointed at the bootstrap sampling rather
than the reweighting. The mechanism matters because it decides what to carry forward.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from road import cv, metric, paths  # noqa: E402

DARK = 136   # the darkest mean brightness jitter 0.25 ever produces from the median scan


def brightness(image_id: str) -> float:
    with Image.open(paths.image_path(image_id)) as im:
        return float(np.asarray(im.convert("L"), dtype=np.float32).mean())


def paired(ea: pd.DataFrame, eb: pd.DataFrame, draws: int = 1000,
           seed: int = 0) -> tuple[float, float, float]:
    """(mean difference B - A, its sd, share of resamples favouring B)."""
    idx = np.random.default_rng(seed).integers(0, len(ea), size=(draws, len(ea)))
    d = np.array([metric.public_estimate(eb.iloc[k]) - metric.public_estimate(ea.iloc[k])
                  for k in idx])
    return float(d.mean()), float(d.std()), float((d > 0).mean())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="the baseline run")
    ap.add_argument("b", help="the run under test")
    ap.add_argument("--draws", type=int, default=1000)
    args = ap.parse_args(argv)

    truth = cv.load_train().set_index("ID").Target.astype(str)
    load = lambda r: pd.read_csv(paths.runs_dir() / r / "oof.csv",  # noqa: E731
                                 keep_default_na=False, dtype=str).set_index("ID").Target
    a, b = load(args.a), load(args.b)
    ids = [i for i in a.index if i in b.index]
    refs = [truth[i] for i in ids]
    ea = metric.line_errors(refs, [a[i] for i in ids])
    eb = metric.line_errors(refs, [b[i] for i in ids])

    pa, pb = metric.public_estimate(ea), metric.public_estimate(eb)
    diff, sd, share = paired(ea, eb, args.draws)
    print(f"{len(ids)} lines, {sum(a[i] != b[i] for i in ids)} decoded differently\n")
    print(f"  {args.a:<22}{pa:.4f}")
    print(f"  {args.b:<22}{pb:.4f}   {pb - pa:+.4f}")
    print(f"\n  paired bootstrap: {diff:+.4f} +/- {sd:.4f}  ({diff / sd if sd else 0:+.1f} sd), "
          f"{share:.0%} of resamples favour {args.b}")

    lum = np.array([brightness(i) for i in ids])
    print(f"\n  by scan brightness (the darkest jitter 0.25 produces is ~{DARK}):")
    for label, sel in ((f"dark   (< {DARK})", lum < DARK), (f"normal (>= {DARK})", lum >= DARK)):
        if sel.sum() < 20:
            print(f"    {label:<18}{sel.sum():>4} lines -- too few to say")
            continue
        sa, sb = metric.public_estimate(ea[sel]), metric.public_estimate(eb[sel])
        print(f"    {label:<18}{sel.sum():>4} lines   {sa:.4f} -> {sb:.4f}   {sb - sa:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
