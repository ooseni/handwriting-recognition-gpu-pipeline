"""Line images -> normalised tensors. Model-agnostic: the output size is a parameter.

Images are read as grayscale and squashed to a fixed (height, width) -- for TrOCR
384x384, the input it was pretrained on (it saw IAM lines squashed the same way).
Augmentation is mild, geometric and photometric only: the ^ ~ & marks must stay
legible.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from torchvision import transforms as T

from road import paths


def load_line(image_id: str) -> Image.Image:
    with Image.open(paths.image_path(image_id)) as im:
        return im.convert("L")


def line_transform(size: tuple[int, int], train: bool = False) -> T.Compose:
    """PIL grayscale -> 1 x H x W tensor in [-1, 1]; random distortions if train."""
    steps: list = [T.Resize(size, interpolation=T.InterpolationMode.BILINEAR)]
    if train:
        steps += [
            T.RandomApply([T.RandomAffine(degrees=1.5, translate=(0.02, 0.04),
                                          scale=(0.93, 1.03), shear=(-6, 6), fill=255)], p=0.7),
            T.ColorJitter(brightness=0.3, contrast=0.3),
            T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.2),
        ]
    steps += [T.ToTensor(), T.Normalize([0.5], [0.5])]
    return T.Compose(steps)


def ink_band(img: Image.Image, on_frac: float = 0.35, min_frac: float = 0.04) -> tuple[int, int] | None:
    """(top, bottom) of the central run of ink rows, or None if no row stands out.

    Crops bleed into the lines above and below -- 57% of hi-res ones carry a whole
    neighbouring line -- so the target line is the band nearest the middle. A row counts
    as ink when it is darker than `on_frac` of the darkest row, measured against the
    parchment background (its 10th percentile).
    """
    rows = 255.0 - np.asarray(img.convert("L"), dtype=np.float32).mean(1)
    rows -= np.percentile(rows, 10)
    if rows.max() <= 0:
        return None
    lit = rows > on_frac * rows.max()
    runs, start, least = [], None, max(3, round(len(lit) * min_frac))
    for i, on in enumerate(lit):
        if on and start is None:
            start = i
        elif not on and start is not None:
            if i - start >= least:
                runs.append((start, i))
            start = None
    if start is not None and len(lit) - start >= least:
        runs.append((start, len(lit)))
    if not runs:
        return None
    middle = len(lit) / 2
    return min(runs, key=lambda r: abs((r[0] + r[1]) / 2 - middle))
