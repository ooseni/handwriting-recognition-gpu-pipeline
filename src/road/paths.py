"""Path resolution that works identically on local Windows and Colab.

Layout (both environments):
    <root>/data/Train.csv, Test.csv, SampleSubmission.csv
    <root>/data/images/<ID>.jpg

On Colab the root is a local SSD dir (fast), NOT the Drive mount -- Drive I/O is
far too slow to feed a dataloader. Drive is used only as the source of the CSVs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Public bucket published by the organisers on the challenge Data page.
IMAGES_URL = "https://storage.googleapis.com/road-handwriting/images.zip"
IMAGES_ZIP_MD5 = "f75061dd4ac41912774d892d8889004f"

# Expected dataset shape -- used by verify_data() to fail loudly on a bad copy.
N_TRAIN = 4098
N_TEST = 1374
N_IMAGES = 5472


def in_colab() -> bool:
    """True when running inside a Google Colab runtime."""
    return "google.colab" in sys.modules or Path("/content").is_dir()


def project_root() -> Path:
    """Repo root.

    Override with ROAD_ROOT to relocate. Falls back to walking up from this
    file until a marker (.git or requirements-local.txt) is found.
    """
    if env := os.environ.get("ROAD_ROOT"):
        return Path(env).expanduser().resolve()

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".git").exists() or (parent / "requirements-local.txt").exists():
            return parent
    # src/road/paths.py -> src/road -> src -> root
    return here.parents[2]


def data_dir() -> Path:
    """Directory holding the CSVs and the images/ subdir."""
    if env := os.environ.get("ROAD_DATA"):
        return Path(env).expanduser().resolve()
    return project_root() / "data"


def images_dir() -> Path:
    return data_dir() / "images"


SYNTH_PREFIX = "stackmix_"


def synth_dir() -> Path:
    """Where road.stackmix writes the lines it synthesises. Kept apart from images/ so
    the competition's own data is never mixed with anything we made."""
    return data_dir() / "synth"


def image_path(image_id: str) -> Path:
    if image_id.startswith(SYNTH_PREFIX):
        return synth_dir() / f"{image_id}.jpg"
    return images_dir() / f"{image_id}.jpg"


def drive_csv_dir() -> Path:
    """Where the three CSVs live on Google Drive (Colab only).

    Upload Train.csv / Test.csv / SampleSubmission.csv to this Drive folder.
    Override with ROAD_DRIVE_CSV.
    """
    if env := os.environ.get("ROAD_DRIVE_CSV"):
        return Path(env)
    return Path("/content/drive/MyDrive/zindi-road/csv")


def outputs_dir() -> Path:
    d = project_root() / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def submissions_dir() -> Path:
    d = project_root() / "submissions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def runs_dir() -> Path:
    """One subdirectory per training run: config, log, predictions, final weights.
    On Colab this is Drive, so a run's results survive a runtime reset.
    Override with ROAD_RUNS.
    """
    if env := os.environ.get("ROAD_RUNS"):
        d = Path(env)
    elif in_colab() and Path("/content/drive/MyDrive").is_dir():
        d = Path("/content/drive/MyDrive/zindi-road/runs")
    else:
        d = project_root() / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def work_dir() -> Path:
    """Fast scratch for files rewritten during training (best-so-far weights).
    On Colab the local SSD, not Drive; elsewhere the same as runs_dir().
    """
    if in_colab():
        d = Path("/content/work")
        d.mkdir(parents=True, exist_ok=True)
        return d
    return runs_dir()


def checkpoints_dir() -> Path:
    """Model checkpoints. On Colab, default to Drive so they survive a runtime
    reset -- checkpoints are the one thing worth the slow I/O.
    """
    if env := os.environ.get("ROAD_CKPT"):
        d = Path(env)
    elif in_colab() and Path("/content/drive/MyDrive").is_dir():
        d = Path("/content/drive/MyDrive/zindi-road/checkpoints")
    else:
        d = project_root() / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d
