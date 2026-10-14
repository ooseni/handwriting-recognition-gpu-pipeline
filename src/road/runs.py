"""Plumbing shared by the training modules: device and precision, data splits,
training-text normalisation, the learning-rate schedule and the command line."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from road import cv, paths, submission

PRECISIONS = ("auto", "bf16", "fp16", "fp32")


def setup(seed: int) -> torch.device:
    """Seed every RNG; return the device, with fast-math settings on a GPU."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True       # fixed input sizes
        torch.set_float32_matmul_precision("high")  # TF32 wherever fp32 remains
        return torch.device("cuda")
    return torch.device("cpu")


def amp_dtype(device: torch.device, precision: str) -> torch.dtype | None:
    """auto: bf16 on Ampere or newer (L4, A100), fp16 on older GPUs (T4), none on CPU."""
    if precision == "auto":
        if device.type != "cuda":
            return None
        precision = "bf16" if torch.cuda.get_device_capability(device)[0] >= 8 else "fp16"
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[precision]


def splits(fold: str, smoke_lines: int = 0) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    """(fit, held-out or None, test) for a held-out fold 0-4 or "all";
    smoke_lines > 0 cuts each split to that many lines."""
    train, test = cv.load_train(), submission.load_test()
    if fold == "all":
        fit, valid = train, None
    else:
        held = train.ID.map(cv.make_folds(train.ID)).eq(int(fold))
        fit, valid = train[~held], train[held]
    if smoke_lines:
        fit, test = fit.head(smoke_lines), test.head(smoke_lines)
        valid = None if valid is None else valid.head(smoke_lines)
    reset = lambda df: None if df is None else df.reset_index(drop=True)  # noqa: E731
    return reset(fit), reset(valid), reset(test)


def shape_fit(fit: pd.DataFrame, cfg) -> pd.DataFrame:
    """Apply --drop and --pseudo to a training split. Shared by every trainer.

    --drop leaves out rows whose label does not match the image; a model trained on
    those learns to invent text. --pseudo adds test lines under labels a previous run
    decoded, which the rules permit so long as nothing is hand-corrected -- and which
    is worth most to whichever model is weakest, since it is the one short of data.

    --synth adds lines road.stackmix recombined from training crops.

    Takes any run Config carrying drop, pseudo, smoke and batch_size (synth optional).
    """
    if cfg.drop:
        path = Path(cfg.drop)
        if not path.exists():
            raise SystemExit(f"--drop {path} does not exist")
        text = path.read_text(encoding="utf-8")
        ids = {line.split(",")[0].strip() for line in text.splitlines() if line.strip()}
        ids.discard("ID")
        keep = ~fit.ID.isin(ids)
        print(f"dropping {(~keep).sum()} of {len(ids)} listed rows", flush=True)
        fit = fit[keep]

    if cfg.pseudo:
        path = Path(cfg.pseudo)
        if not path.exists():
            raise SystemExit(f"--pseudo {path} does not exist")
        extra = pd.read_csv(path, keep_default_na=False, dtype=str)
        extra = extra[extra.Target.str.strip().ne("")][["ID", "Target"]]
        if cfg.smoke:
            extra = extra.head(2 * cfg.batch_size)
        overlap = set(extra.ID) & set(fit.ID)
        if overlap:
            raise SystemExit(f"--pseudo {path} repeats {len(overlap)} training IDs; "
                             "it should hold test lines only")
        fit = pd.concat([fit, extra], ignore_index=True)
        print(f"added {len(extra)} pseudo-labelled test lines -> {len(fit)} training lines",
              flush=True)

    if getattr(cfg, "synth", ""):
        path = Path(cfg.synth)
        if not path.exists():
            raise SystemExit(f"--synth {path} does not exist -- run road.stackmix first")
        extra = pd.read_csv(path, keep_default_na=False, dtype=str)[["ID", "Target"]]
        extra = extra[extra.Target.str.strip().ne("")]
        if cfg.smoke:
            extra = extra.head(2 * cfg.batch_size)
        fit = pd.concat([fit, extra], ignore_index=True)
        print(f"added {len(extra)} synthetic lines -> {len(fit)} training lines", flush=True)

    return fit.reset_index(drop=True)


def claim(run_dir, overwrite: bool, smoke: bool) -> None:
    """Refuse to overwrite an earlier run's artefacts -- its adapter, OOF and submission
    are the only copy. Smoke runs are throwaway, so they may reuse a name."""
    saved = run_dir / "config.json"
    if saved.exists() and not smoke and not overwrite:
        raise SystemExit(
            f"{run_dir} already holds a finished run. Pick another --run name so its "
            "weights and predictions survive, or pass --overwrite to replace them.")


def clean_target(text: str) -> str:
    """Training text: outer whitespace dropped, inner runs collapsed to one space.
    Scoring always uses the raw targets."""
    return " ".join(text.split())


def warmup_cosine(opt: torch.optim.Optimizer, steps: int,
                  warmup: float) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warm-up over the first `warmup` fraction of `steps`, then cosine to zero."""
    warm = max(1, round(warmup * steps))

    def lr_at(s: int) -> float:
        if s < warm:
            return (s + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, steps - warm))))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_at)


def saved_config(config_cls, run_dir: Path, **overrides):
    """Rebuild a finished run's Config from its config.json.

    Fields the run predates fall back to their default, so a config written before a
    new setting existed still loads -- older runs stay usable after the code moves on.
    """
    meta = json.loads((run_dir / "config.json").read_text())
    known = {f.name: meta[f.name] for f in fields(config_cls) if f.name in meta}
    return config_cls(**(known | overrides))


def parse(config_cls, argv: list[str] | None, description: str,
          smoke_epochs: int = 2, smoke_workers: int = 2):
    """Command line generated from a run Config dataclass; returns (cfg, predict_only).

    --predict-only reloads the saved run's config, keeping only the decode-time flags
    (--beams, --n-best, --decode-batch-size) from the command line; --smoke caps epochs
    and DataLoader workers.
    """
    ap = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run name; artefacts go to runs/<run>")
    defaults = config_cls(run="")
    for f in fields(config_cls)[1:]:
        d = getattr(defaults, f.name)
        flag = "--" + f.name.replace("_", "-")
        if isinstance(d, bool):
            ap.add_argument(flag, action=argparse.BooleanOptionalAction, default=d)
        else:
            ap.add_argument(flag, type=type(d), default=d)
    ap.add_argument("--predict-only", action="store_true",
                    help="skip training; re-decode a saved run (only --beams may change)")
    args = vars(ap.parse_args(argv))
    predict_only = args.pop("predict_only")
    cfg = config_cls(**args)

    if cfg.fold != "all" and cfg.fold not in {str(k) for k in range(cv.N_FOLDS)}:
        ap.error(f"--fold must be 0-{cv.N_FOLDS - 1} or 'all', got {cfg.fold!r}")
    if cfg.precision not in PRECISIONS:
        ap.error(f"--precision must be one of {', '.join(PRECISIONS)}, got {cfg.precision!r}")
    if predict_only:
        saved = paths.runs_dir() / cfg.run / "config.json"
        if not saved.exists():
            ap.error(f"no trained run at {saved.parent}")
        # Decode-time settings may still be changed; everything else comes from the
        # run as it was trained. Keeping --n-best here is what lets a finished run be
        # re-decoded for more candidates without retraining it.
        names = {f.name for f in fields(config_cls)}
        decode_time = {k: getattr(cfg, k) for k in ("beams", "n_best", "decode_batch_size")
                       if k in names}
        cfg = saved_config(config_cls, saved.parent, **decode_time)
    elif cfg.smoke:
        cfg.epochs, cfg.workers = min(cfg.epochs, smoke_epochs), min(cfg.workers, smoke_workers)
    return cfg, predict_only
