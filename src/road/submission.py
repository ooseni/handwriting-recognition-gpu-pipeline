"""Build and validate submission files.

Rules (competition page, SampleSubmission.csv): columns ID,Target, one row per
Test ID, any order. Missing, empty or invalid predictions are "penalised as
incorrect", so clean() never emits an empty string: it substitutes FALLBACK,
which scores at most 1.0 on any line -- never worse than the empty penalty.

    python -m road.submission FILE    # check a file (e.g. one built on Colab) before upload
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

from road import paths

# In 50% of training lines. Against a reference of n >= 3 characters it costs at
# most n character edits and one edit per reference word, so both rates stay <= 1.
FALLBACK = "the"


def load_test() -> pd.DataFrame:
    return pd.read_csv(paths.data_dir() / "Test.csv", keep_default_na=False, dtype=str)


def clean(pred: object) -> str:
    """One line of stripped text, never empty."""
    s = re.sub(r"[\r\n\t]+", " ", pred).strip() if isinstance(pred, str) else ""
    return s or FALLBACK


def problems(sub: pd.DataFrame, test_ids: Iterable[str]) -> list[str]:
    """Everything wrong with a submission frame; empty when it is valid."""
    if list(sub.columns) != ["ID", "Target"]:
        return [f"columns are {list(sub.columns)}, want ['ID', 'Target']"]
    out = []
    if n := sub.ID.duplicated().sum():
        out.append(f"{n} duplicate IDs")
    want, have = set(test_ids), set(sub.ID)
    if n := len(want - have):
        out.append(f"{n} test IDs missing")
    if n := len(have - want):
        out.append(f"{n} IDs not in Test.csv")
    text = sub.Target.map(lambda s: isinstance(s, str) and bool(s.strip()))
    if n := (~text).sum():
        out.append(f"{n} empty or missing predictions")
    if n := sub.Target[text].str.contains(r"[\r\n]").sum():
        out.append(f"{n} predictions span several lines")
    return out


def check_file(path: Path, test_ids: Iterable[str]) -> list[str]:
    """problems() as a default CSV reader sees the file: NA-like text such as
    'NA' or 'null' comes back missing, as it may on the platform."""
    return problems(pd.read_csv(path, dtype=str), test_ids)


def write(ids: Sequence[str], preds: Sequence[object], name: str,
          test_ids: Iterable[str] | None = None) -> Path:
    """Clean predictions, write submissions/<name>.csv and check the written file.

    Raises ValueError, and removes the file, if anything is wrong.
    """
    test_ids = list(load_test().ID if test_ids is None else test_ids)
    path = paths.submissions_dir() / f"{name}.csv"
    pd.DataFrame({"ID": list(ids), "Target": [clean(p) for p in preds]}).to_csv(path, index=False)
    if issues := check_file(path, test_ids):
        path.unlink()
        raise ValueError(f"{path.name}: " + "; ".join(issues))
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", type=Path)
    args = ap.parse_args(argv)

    issues = check_file(args.file, load_test().ID)
    for msg in issues:
        print(f"  FAIL  {msg}")
    print(f"{args.file.name}: " + ("FAIL" if issues else "ok -- ready to upload"))
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
