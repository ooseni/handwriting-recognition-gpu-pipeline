"""Where a model's errors are. Breaks an out-of-fold file's score down by line length
and scan scale, estimates how much CER each class of difference costs, and lists the
worst lines.

    python scripts/analyse_oof.py runs/trocr_f0/oof.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from road import cv, metric, paths  # noqa: E402

HI_RES_MIN_HEIGHT = 150  # splits the two scan scales, as in profile_dataset.py


def section(title: str) -> None:
    print(f"\n{'=' * 66}\n{title}\n{'=' * 66}")


def breakdown(err: pd.DataFrame, by: pd.Series) -> None:
    rows = {k: {"n": len(g), **metric.aggregate(g)} for k, g in err.groupby(by, observed=True)}
    print(pd.DataFrame.from_dict(rows, orient="index")[["n", "wer", "cer", "score"]]
          .to_string(float_format=lambda x: f"{x:.4f}"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("oof", type=Path, help="CSV of ID,Target[,fold] predictions on train lines")
    ap.add_argument("--worst", type=int, default=15, help="how many worst lines to list")
    args = ap.parse_args(argv)

    truth = cv.load_train().set_index("ID").Target
    oof = pd.read_csv(args.oof, keep_default_na=False, dtype=str)
    ref, hyp = truth.loc[oof.ID].tolist(), oof.Target.tolist()
    err = metric.line_errors(ref, hyp)
    base = metric.aggregate(err)
    print(f"{args.oof}: {len(err)} lines   WER {base['wer']:.4f}  CER {base['cer']:.4f}  "
          f"score {base['score']:.4f}  (0.3*WER + 0.7*CER = "
          f"{0.3 * base['wer'] + 0.7 * base['cer']:.4f})")

    section("BY REFERENCE LENGTH (characters, quartiles)")
    breakdown(err, pd.qcut(err.chars, 4))

    section("BY SCAN SCALE")
    heights = []
    for i in oof.ID:
        with Image.open(paths.image_path(i)) as im:
            heights.append(im.size[1])
    breakdown(err, pd.Series(["hi-res" if h >= HI_RES_MIN_HEIGHT else "lo-res" for h in heights]))

    section("WHAT EACH CLASS OF DIFFERENCE COSTS (CER with that class forgiven)")
    forgive = {
        "case": str.lower,
        "symbols .,;:-'^~& etc": lambda s: re.sub(r"[^\w\s]", "", s),
        "^ marks only": lambda s: s.replace("^", ""),
        "word spacing": lambda s: re.sub(r"\s+", "", s),
    }
    for name, f in forgive.items():
        e = metric.aggregate(metric.line_errors([f(r) or "_" for r in ref], [f(h) for h in hyp]))
        share = (base["cer"] - e["cer"]) / base["cer"] if base["cer"] else 0.0
        wer = "" if name == "word spacing" else f"   WER {e['wer']:.4f}"
        print(f"  {name:24s} CER {e['cer']:.4f}  (~{share:4.0%} of CER){wer}")

    section("SHAPE OF THE ERRORS")
    ratio = pd.Series([len(h.strip()) for h in hyp]) / err.chars
    for label, mask in [("prediction < 80% of reference length", ratio < 0.8),
                        ("prediction > 120% of reference length", ratio > 1.2)]:
        share = err.char_edits[mask].sum() / max(1, err.char_edits.sum())
        print(f"  {label:40s} {int(mask.sum()):4d} lines, {share:5.1%} of character edits")
    line_cer = err.char_edits / err.chars
    top = line_cer.sort_values(ascending=False).index[: max(1, len(err) // 10)]
    print(f"  worst 10% of lines hold {err.char_edits[top].sum() / max(1, err.char_edits.sum()):.1%}"
          " of character edits")
    print(f"  perfect lines: {(err.char_edits == 0).mean():.1%}")

    section(f"WORST {args.worst} LINES BY CER")
    for i in line_cer.sort_values(ascending=False).index[: args.worst]:
        print(f"\n  CER {line_cer[i]:.2f}  {oof.ID[i]}  h={heights[i]}\n  ref  {ref[i]}\n  pred {hyp[i]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
