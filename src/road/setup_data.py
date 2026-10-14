"""Fetch and verify the challenge dataset. Idempotent -- safe to re-run.

Images come from the organisers' public GCS bucket (see paths.IMAGES_URL), so
nothing large needs uploading. The three CSVs are not public, so they are read
from Google Drive on Colab, or from an existing local copy.

Usage
-----
    python -m road.setup_data                 # fetch whatever is missing
    python -m road.setup_data --verify-only   # check an existing copy
    python -m road.setup_data --csv-src DIR   # take CSVs from DIR
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from road import paths

CSV_NAMES = ("Train.csv", "Test.csv", "SampleSubmission.csv")


def _human(n: int) -> str:
    return f"{n / 1024**2:,.1f} MB"


def download(url: str, dest: Path, chunk: int = 1 << 20) -> Path:
    """Stream `url` to `dest`, resuming is not attempted but partials are cleaned."""
    if dest.exists():
        print(f"  [skip] {dest.name} already present ({_human(dest.stat().st_size)})")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {url}")
    try:
        with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
            total = int(resp.headers.get("Content-Length", 0))
            done = 0
            while buf := resp.read(chunk):
                out.write(buf)
                done += len(buf)
                if total:
                    pct = 100 * done / total
                    print(f"\r    {_human(done)} / {_human(total)}  ({pct:5.1f}%)",
                          end="", flush=True)
            print()
        tmp.replace(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return dest


def md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while buf := f.read(chunk):
            h.update(buf)
    return h.hexdigest()


def unzip_images(zip_path: Path, target: Path) -> None:
    """Extract images/*.jpg from the archive into `target` as a flat dir."""
    existing = sum(1 for _ in target.glob("*.jpg")) if target.is_dir() else 0
    if existing >= paths.N_IMAGES:
        print(f"  [skip] {existing} images already extracted")
        return

    target.mkdir(parents=True, exist_ok=True)
    print(f"  extracting to {target}")
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.infolist() if m.filename.lower().endswith(".jpg")]
        for i, m in enumerate(members, 1):
            # Flatten: archive stores them under images/, we want them bare.
            out = target / Path(m.filename).name
            if not out.exists():
                with z.open(m) as src, open(out, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            if i % 500 == 0 or i == len(members):
                print(f"\r    {i}/{len(members)}", end="", flush=True)
    print()


def stage_csvs(src: Path | None) -> None:
    """Copy the three CSVs into data/ from `src` (Drive folder or local dir)."""
    dst = paths.data_dir()
    dst.mkdir(parents=True, exist_ok=True)

    missing = [n for n in CSV_NAMES if not (dst / n).exists()]
    if not missing:
        print("  [skip] all CSVs present")
        return

    if src is None:
        src = paths.drive_csv_dir()

    if not src.is_dir():
        raise SystemExit(
            f"\nCSV source not found: {src}\n"
            f"Missing: {', '.join(missing)}\n\n"
            "On Colab: mount Drive and upload the three CSVs from the Zindi\n"
            f"download to  {paths.drive_csv_dir()}\n"
            "Or pass an explicit directory with --csv-src DIR."
        )

    for name in missing:
        s = src / name
        if not s.exists():
            raise SystemExit(f"Missing {name} in {src}")
        shutil.copy2(s, dst / name)
        print(f"  copied {name} ({s.stat().st_size / 1024:,.0f} KB)")


def verify() -> bool:
    """Structural check: counts line up and every referenced image exists."""
    import pandas as pd

    d = paths.data_dir()
    ok = True

    for name in CSV_NAMES:
        if not (d / name).exists():
            print(f"  FAIL  missing {name}")
            ok = False
    if not ok:
        return False

    train = pd.read_csv(d / "Train.csv", keep_default_na=False, dtype=str)
    test = pd.read_csv(d / "Test.csv", keep_default_na=False, dtype=str)
    sub = pd.read_csv(d / "SampleSubmission.csv", keep_default_na=False, dtype=str)

    checks: list[tuple[str, bool, str]] = [
        ("train rows", len(train) == paths.N_TRAIN, f"{len(train)} (want {paths.N_TRAIN})"),
        ("test rows", len(test) == paths.N_TEST, f"{len(test)} (want {paths.N_TEST})"),
        ("sub rows", len(sub) == paths.N_TEST, f"{len(sub)} (want {paths.N_TEST})"),
        ("test IDs == sub IDs", set(test.ID) == set(sub.ID), ""),
        ("no train/test leak", not (set(train.ID) & set(test.ID)), ""),
        ("no dup train IDs", not train.ID.duplicated().any(), ""),
        ("train has Target", "Target" in train.columns, ""),
        ("no empty targets", (train.Target.str.len() > 0).all(), ""),
    ]

    n_jpg = sum(1 for _ in paths.images_dir().glob("*.jpg")) if paths.images_dir().is_dir() else 0
    checks.append(("image count", n_jpg == paths.N_IMAGES, f"{n_jpg} (want {paths.N_IMAGES})"))

    have = {p.stem for p in paths.images_dir().glob("*.jpg")} if n_jpg else set()
    need = set(train.ID) | set(test.ID)
    absent = need - have
    checks.append(("all referenced images present", not absent,
                   f"{len(absent)} missing" if absent else ""))

    for label, passed, detail in checks:
        mark = "ok  " if passed else "FAIL"
        print(f"  {mark}  {label}" + (f"  -- {detail}" if detail else ""))
        ok &= passed

    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-src", type=Path, default=None,
                    help="directory containing the three CSVs")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--skip-md5", action="store_true",
                    help="skip the archive checksum (saves ~30s)")
    ap.add_argument("--keep-zip", action="store_true",
                    help="keep images.zip after extracting (uses +422MB)")
    args = ap.parse_args(argv)

    print(f"root       : {paths.project_root()}")
    print(f"data       : {paths.data_dir()}")
    print(f"colab      : {paths.in_colab()}")

    if not args.verify_only:
        print("\n[1/3] CSVs")
        stage_csvs(args.csv_src)

        print("\n[2/3] images archive")
        zp = paths.data_dir() / "images.zip"
        n_jpg = (sum(1 for _ in paths.images_dir().glob("*.jpg"))
                 if paths.images_dir().is_dir() else 0)
        if n_jpg < paths.N_IMAGES:
            download(paths.IMAGES_URL, zp)
            if not args.skip_md5:
                got = md5(zp)
                status = "ok" if got == paths.IMAGES_ZIP_MD5 else "MISMATCH"
                print(f"  md5 {got}  [{status}]")
                if got != paths.IMAGES_ZIP_MD5:
                    print("  warning: archive differs from the copy verified on "
                          "2026-08-22 -- organisers may have reissued it.")
            unzip_images(zp, paths.images_dir())
            if not args.keep_zip:
                zp.unlink(missing_ok=True)
                print("  removed images.zip (pass --keep-zip to retain)")
        else:
            print(f"  [skip] {n_jpg} images already extracted")

    print("\n[3/3] verify")
    ok = verify()
    print("\n" + ("PASS -- dataset ready." if ok else "FAIL -- see above."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
