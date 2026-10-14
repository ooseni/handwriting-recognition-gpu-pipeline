"""StackMix: new labelled training lines, cut and recombined from the training lines.

    python -m road.stackmix --n 4000 --hold-out 0     # fold-0 experiment: never cut fold 0
    python -m road.stackmix --n 4000                  # every training line

Why: three results say the models overfit 4,098 lines -- jitter won as a regulariser
rather than as designed, the baseline regresses at epoch 3, and doubling the model from
4B to 8B changed nothing, which is what a data-starved model looks like. Synthesising
new lines attacks the shortage directly. The host allows it (2026-08-24) provided it is
fully automated and reproducible and uses only the training images.

How: a line is cut into words at the gaps in its ink, and the cut is trusted only if it
passes two checks.

1. The gaps must split the line into exactly as many pieces as the transcript has words.
   That alone is not enough. Cursive breaks inside words and runs across the gaps
   between them, so a merge and a split can cancel and the count still matches -- with
   every label after them shifted by one. On a sample, three of four lines that passed
   the count were misaligned this way.
2. So every piece must also be about as wide per character as the others on its line.
   A merged pair is far too wide for its label and a split fragment far too narrow.
   With both checks the misaligned lines went, and six of six survivors inspected were
   aligned correctly, superscripts and ampersands included.

About 9% of lines pass, giving some 3,900 word crops. New lines take a real run of words
from one line -- so the text reads like the corpus -- and render each word with a crop of
that word from any line, which recombines scribes. Paper colour is matched across the
pieces so the seams do not become something the model learns to spot.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image

from road import cv, paths

PREFIX = paths.SYNTH_PREFIX     # routed to paths.synth_dir()
WORK_H = 64                    # height the cut is searched at
OUT_H = 96                     # height synthetic lines are built at
MIN_GAPS = (4, 5, 6, 7, 8, 10, 12)
BAND = (0.2, 0.85)             # rows carrying the words; ascenders above are the line above
WIDTH_BAND = (0.5, 2.0)        # width per character, as a multiple of the line's median


def _otsu(gray: np.ndarray) -> float:
    hist, _ = np.histogram(gray, bins=256, range=(0, 255))
    p = hist / max(hist.sum(), 1)
    omega, mu = np.cumsum(p), np.cumsum(p * np.arange(256))
    between = (mu[-1] * omega - mu) ** 2 / (omega * (1 - omega) + 1e-12)
    return float(np.argmax(between))


def cuts(gray: np.ndarray, min_gap: int) -> list[tuple[int, int]]:
    """Runs of ink separated by at least `min_gap` empty columns, in `gray`'s pixels."""
    ink = gray <= _otsu(gray)   # Otsu's low class includes the threshold itself
    band = ink[int(ink.shape[0] * BAND[0]):int(ink.shape[0] * BAND[1])]
    col = np.convolve(band.sum(0), np.ones(3) / 3, mode="same")
    empty = col < max(1.0, 0.02 * band.shape[0])
    segs, start, gap = [], None, 0
    for x, is_empty in enumerate(empty):
        if not is_empty:
            if start is None:
                start = x
            gap = 0
        else:
            gap += 1
            if start is not None and gap >= min_gap:
                segs.append((start, x - gap + 1))
                start = None
    if start is not None:
        segs.append((start, len(empty)))
    return [s for s in segs if s[1] - s[0] >= 3]


def consistent(segs: list[tuple[int, int]], words: list[str]) -> bool:
    """Every piece about as wide per character as the rest. len+1 damps one-letter words."""
    r = np.array([(b - a) / (len(w) + 1) for (a, b), w in zip(segs, words)])
    med = np.median(r)
    return bool(med > 0 and np.all((r > WIDTH_BAND[0] * med) & (r < WIDTH_BAND[1] * med)))


def segment(image: Image.Image, words: list[str]) -> list[tuple[int, int]] | None:
    """Word boundaries in the image's own pixels, or None if no cut can be trusted."""
    if not words:
        return None
    gray_img = image.convert("L")
    scale = WORK_H / gray_img.height
    small = gray_img.resize((max(1, round(gray_img.width * scale)), WORK_H))
    gray = np.asarray(small, dtype=np.float32)
    for min_gap in MIN_GAPS:
        segs = cuts(gray, min_gap)
        if len(segs) == len(words) and consistent(segs, words):
            return [(round(a / scale), round(b / scale)) for a, b in segs]
    return None


@dataclass
class Crop:
    image: Image.Image         # RGB, OUT_H high
    word: str
    source: str                # the line it was cut from


def harvest(frame: pd.DataFrame) -> list[Crop]:
    """Word crops from every line whose cut can be trusted, in reading order."""
    crops: list[Crop] = []
    for image_id, text in zip(frame.ID, frame.Target):
        words = str(text).split()
        with Image.open(paths.image_path(image_id)) as im:
            im = im.convert("RGB")
            bounds = segment(im, words)
            if bounds is None:
                continue
            pad = max(1, round(0.04 * im.height))
            for (a, b), word in zip(bounds, words):
                piece = im.crop((max(0, a - pad), 0, min(im.width, b + pad), im.height))
                width = max(1, round(piece.width * OUT_H / piece.height))
                crops.append(Crop(piece.resize((width, OUT_H)), word, image_id))
    return crops


def paper(img: Image.Image) -> np.ndarray:
    """The paper's colour: the bright end of each channel, so ink does not drag it."""
    return np.percentile(np.asarray(img, dtype=np.float32).reshape(-1, 3), 90, axis=0)


def recolour(img: Image.Image, target: np.ndarray) -> Image.Image:
    arr = np.asarray(img, dtype=np.float32) * (target / np.maximum(paper(img), 1.0))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def synthesise(crops: list[Crop], n: int, seed: int = 0) -> list[tuple[Image.Image, str]]:
    """n new lines. Deterministic for a given seed, as the rules require."""
    rng = random.Random(seed)
    by_word: dict[str, list[Crop]] = defaultdict(list)
    by_source: dict[str, list[Crop]] = defaultdict(list)
    for c in crops:
        by_word[c.word].append(c)
        by_source[c.source].append(c)
    sources = sorted(by_source)
    out = []
    for _ in range(n):
        line = by_source[rng.choice(sources)]
        length = rng.randint(min(3, len(line)), min(12, len(line)))
        start = rng.randint(0, len(line) - length)
        words = [c.word for c in line[start:start + length]]
        pieces = [rng.choice(by_word[w]) for w in words]
        target = paper(pieces[0].image)
        parts = [recolour(p.image, target) for p in pieces]
        gaps = [rng.randint(round(0.15 * OUT_H), round(0.45 * OUT_H)) for _ in parts[1:]]
        margin = round(0.1 * OUT_H)
        width = sum(p.width for p in parts) + sum(gaps) + 2 * margin
        canvas = Image.new("RGB", (width, OUT_H), tuple(int(v) for v in target))
        x = margin
        for i, part in enumerate(parts):
            canvas.paste(part, (x, 0))
            x += part.width + (gaps[i] if i < len(gaps) else 0)
        out.append((canvas, " ".join(words)))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=4000, help="synthetic lines to write")
    ap.add_argument("--hold-out", default=None,
                    help="fold 0-4 never to cut from, so a fold-0 test stays clean")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    train = cv.load_train()
    if args.hold_out is not None:
        folds = cv.make_folds(train.ID)
        train = train[train.ID.map(folds) != int(args.hold_out)]
    crops = harvest(train)
    if not crops:
        raise SystemExit("no line could be cut with confidence -- nothing to recombine")
    sources = len({c.source for c in crops})
    print(f"cut {len(crops)} word crops from {sources} of {len(train)} lines "
          f"({sources / len(train):.0%})", flush=True)

    out_dir = paths.synth_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (img, text) in enumerate(synthesise(crops, args.n, args.seed)):
        image_id = f"{PREFIX}{i:06d}"
        img.save(out_dir / f"{image_id}.jpg", quality=92)
        rows.append((image_id, text))
    csv = out_dir / "stackmix.csv"
    pd.DataFrame(rows, columns=["ID", "Target"]).to_csv(csv, index=False)
    print(f"{len(rows)} synthetic lines -> {csv}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
