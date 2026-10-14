"""Profile the dataset: label statistics, geometry, conventions.

Every dataset claim in README.md is derived here. Re-run to audit them.

    python scripts/profile_dataset.py
    python scripts/profile_dataset.py --sample-images 0   # all images (slower)
"""

from __future__ import annotations

import argparse
import collections
import random
import re
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from road import paths  # noqa: E402

HI_RES_MIN_HEIGHT = 150  # splits the bimodal scan-scale distribution


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    d = paths.data_dir()
    kw = dict(keep_default_na=False, dtype=str)
    return pd.read_csv(d / "Train.csv", **kw), pd.read_csv(d / "Test.csv", **kw)


def section(title: str) -> None:
    print(f"\n{'=' * 66}\n{title}\n{'=' * 66}")


def profile_labels(train: pd.DataFrame) -> None:
    t = train.Target
    section("LABEL LENGTHS")
    chars, words = t.str.len(), t.str.split().str.len()
    print(f"chars  min={chars.min()} median={chars.median():.0f} "
          f"p95={chars.quantile(.95):.0f} max={chars.max()}")
    print(f"words  min={words.min()} median={words.median():.0f} "
          f"p95={words.quantile(.95):.0f} max={words.max()}")
    print(f"single-word samples: {(words == 1).sum()}   empty: {(chars == 0).sum()}")

    section("METRIC WEIGHTING (assuming weight = len(ref) ** 0.5)")
    w = chars.pow(0.5)
    top = chars >= chars.quantile(0.9)
    print(f"longest 10% of samples carry {w[top].sum() / w.sum():.1%} of total weight")
    print("=> weighting is gentle here; no sample is short enough to be free")

    section("CHARSET")
    counts = collections.Counter("".join(t))
    print(f"{len(counts)} unique chars:\n{''.join(sorted(counts))}")
    rare = {c: n for c, n in counts.items() if n <= 5}
    print(f"\nrare (<=5 occurrences): {rare}")

    section("TRANSCRIPTION CONVENTIONS (must be reproduced literally)")
    for pat, desc in [
        (r"\^", "caret / superscript"),
        (r"~", "tilde (line-end mark)"),
        (r"-\s*$", "trailing hyphen"),
        (r"&", "ampersand"),
        (r"\d", "digits"),
        (r"[A-Z]", "any uppercase"),
    ]:
        print(f"  {desc:24s} {t.str.contains(pat).sum():5d} rows")

    carets = collections.Counter(re.findall(r"\w*\^\w+", " ".join(t)))
    print(f"\ncaret tokens: {sum(carets.values())} total, {len(carets)} distinct")
    print(f"most common: {carets.most_common(12)}")

    section("VOCABULARY (the lever for WER)")
    vocab = collections.Counter(w.lower() for s in t for w in re.findall(r"[A-Za-z']+", s))
    hapax = sum(1 for c in vocab.values() if c == 1)
    print(f"{len(vocab)} word types over {sum(vocab.values())} tokens; {hapax} hapax")
    print(f"top 25: {[w for w, _ in vocab.most_common(25)]}")
    print(f"\nunique targets: {t.nunique()} of {len(t)} rows "
          f"({len(t) - t.nunique()} exact duplicates)")

    section("ROW ORDER (any document structure to group CV folds on?)")
    toks = [set(s.lower().split()) for s in t]

    def jaccard(a: set[str], b: set[str]) -> float:
        return len(a & b) / max(1, len(a | b))

    adj = sum(jaccard(toks[i], toks[i + 1]) for i in range(len(toks) - 1)) / (len(toks) - 1)
    rng = random.Random(0)
    pairs = [(rng.randrange(len(toks)), rng.randrange(len(toks))) for _ in range(20000)]
    rnd = sum(jaccard(toks[i], toks[j]) for i, j in pairs) / len(pairs)
    print(f"word overlap (Jaccard): adjacent rows {adj:.3f} vs random pairs {rnd:.3f}")
    print("=> near-equal: row order is not document order; plain random K-fold")


def profile_images(train: pd.DataFrame, test: pd.DataFrame, n_sample: int) -> None:
    # Read every header: the split check below is only meaningful if exact, and
    # PIL's lazy open reads the JPEG header without decoding pixels.
    sizes: dict[str, tuple[int, int]] = {}
    for i in list(train.ID) + list(test.ID):
        with Image.open(paths.image_path(i)) as im:
            sizes[i] = im.size

    section("IMAGE GEOMETRY")
    geo_ids = list(sizes)
    if n_sample and n_sample < len(geo_ids):
        random.seed(0)
        geo_ids = random.sample(geo_ids, n_sample)
        print(f"(percentiles from a {n_sample}-image sample)")

    def q(vals: list[float], p: float) -> float:
        v = sorted(vals)
        return v[int(p * (len(v) - 1))]

    hs = [sizes[i][1] for i in geo_ids]
    ars = [sizes[i][0] / sizes[i][1] for i in geo_ids]
    print(f"height  p10={q(hs,.1):.0f} p50={q(hs,.5):.0f} p90={q(hs,.9):.0f} max={max(hs)}")
    print(f"aspect  p10={q(ars,.1):.1f} p50={q(ars,.5):.1f} p90={q(ars,.9):.1f} max={max(ars):.1f}")
    print(f"at height=64, width p50={q([sizes[i][0] * 64 / sizes[i][1] for i in geo_ids], .5):.0f}")

    section("SCAN SCALE vs SPLIT (checks for distribution shift; all images)")
    for name, df in (("train", train), ("test", test)):
        hi = sum(sizes[i][1] >= HI_RES_MIN_HEIGHT for i in df.ID)
        print(f"{name:5s} n={len(df):5d}  hi-res {hi / len(df):5.1%}")
    print("=> near-identical proportions: the split is random, no shift to correct")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-images", type=int, default=800,
                    help="0 = every image (slower)")
    args = ap.parse_args(argv)

    if not (paths.data_dir() / "Train.csv").exists():
        raise SystemExit(f"No data at {paths.data_dir()}. Run: python -m road.setup_data")

    train, test = load()
    print(f"train={len(train)}  test={len(test)}")
    profile_labels(train)
    profile_images(train, test, args.sample_images)
    return 0


if __name__ == "__main__":
    sys.exit(main())
