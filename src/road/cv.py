"""Cross-validation: fixed folds, a leak-free fit/predict loop, OOF scoring.

Plain random K-fold, because a random split reproduces the train -> test relation:
  * the split itself looks random at line level (hi-res share 25.5% vs 25.1%);
  * row order carries no document structure -- adjacent rows share barely more
    words than random pairs -- so there is nothing to group on;
  * only 12 of 4,098 targets repeat an earlier one, so leakage via repeats is
    negligible.
Lines by the same hand fall on both sides of a fold, as they do across train/test.

Anything fitted on targets -- charset, lexicon, n-gram LM, decoding weights -- must
be fitted inside the fold. run_cv enforces this by never passing validation
targets to the model.

    python -m road.cv             # fold sizes + fingerprint (compare local vs Colab)
    python -m road.cv OOF.csv     # score out-of-fold predictions: ID,Target[,fold]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from road import metric, paths

N_FOLDS = 5
SEED = 42
N_PUBLIC = round(0.3 * paths.N_TEST)  # public LB lines; the other 70% are private

FitPredict = Callable[[pd.DataFrame, pd.DataFrame], Sequence[str]]


def load_train() -> pd.DataFrame:
    return pd.read_csv(paths.data_dir() / "Train.csv", keep_default_na=False, dtype=str)


def make_folds(ids: Sequence[str], n_folds: int = N_FOLDS, seed: int = SEED) -> pd.Series:
    """ID -> fold, sizes within one line of each other.

    IDs are ranked by a salted SHA-256, so the assignment depends only on the set
    of IDs: not on row order, platform, or numpy's RNG (whose streams may change
    between versions). Local and Colab get identical folds.
    """
    ranked = sorted(ids, key=lambda i: hashlib.sha256(f"{seed}:{i}".encode()).hexdigest())
    return pd.Series(np.arange(len(ranked)) % n_folds, index=ranked, name="fold")


def fingerprint(folds: pd.Series) -> str:
    """Short hash of an ID -> fold assignment, for comparing machines."""
    blob = "\n".join(f"{i},{k}" for i, k in sorted(folds.items()))
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def run_cv(fit_predict: FitPredict, train: pd.DataFrame,
           folds: pd.Series | None = None, only: Sequence[int] | None = None) -> pd.DataFrame:
    """Out-of-fold predictions as a frame of ID, Target, fold.

    fit_predict(fit_part, valid_part) gets the other folds with targets and the
    held-out fold without them, and returns one prediction per held-out row.
    `only` runs a subset of folds, e.g. [0] while iterating.
    """
    folds = make_folds(train.ID) if folds is None else folds
    fold_of = train.ID.map(folds)
    if fold_of.isna().any():
        raise ValueError(f"{fold_of.isna().sum()} train IDs have no fold")
    parts = []
    for k in sorted(fold_of.unique()) if only is None else only:
        held = fold_of.eq(k)
        fit_part = train[~held].reset_index(drop=True)
        valid_part = train[held].drop(columns="Target").reset_index(drop=True)
        pred = list(fit_predict(fit_part, valid_part))
        if len(pred) != len(valid_part):
            raise ValueError(f"fold {k}: {len(pred)} predictions for {len(valid_part)} lines")
        parts.append(pd.DataFrame({"ID": valid_part.ID, "Target": pred, "fold": k}))
    return pd.concat(parts, ignore_index=True)


def oof_errors(oof: pd.DataFrame, train: pd.DataFrame, folds: pd.Series) -> pd.DataFrame:
    """metric.line_errors for each OOF row, tagged with its ID and fold."""
    if oof.ID.duplicated().any():
        raise ValueError("duplicate IDs in OOF predictions")
    unknown = set(oof.ID) - set(train.ID)
    if unknown:
        raise ValueError(f"{len(unknown)} OOF IDs are not in Train.csv, e.g. {sorted(unknown)[:3]}")
    fold = oof.ID.map(folds).to_numpy()
    if "fold" in oof.columns and (pd.to_numeric(oof["fold"]).to_numpy() != fold).any():
        raise ValueError("OOF 'fold' column disagrees with make_folds(): these lines "
                         "may have been predicted by a model that trained on them")
    err = metric.line_errors(train.set_index("ID").Target.loc[oof.ID], oof.Target)
    err.insert(0, "fold", fold)
    err.insert(0, "ID", oof.ID.to_numpy())
    return err


def fold_table(errors: pd.DataFrame, weight_factor: float = metric.WEIGHT_FACTOR) -> pd.DataFrame:
    """n, score, wer, cer per fold, plus an 'all' row over every scored line."""
    rows = {k: {"n": len(g), **metric.aggregate(g, weight_factor)}
            for k, g in errors.groupby("fold")}
    rows["all"] = {"n": len(errors), **metric.aggregate(errors, weight_factor)}
    return pd.DataFrame.from_dict(rows, orient="index")


def lb_noise(errors: pd.DataFrame, n_lines: int, n_draws: int = 2000, seed: int = 0,
             weight_factor: float = metric.WEIGHT_FACTOR) -> float:
    """Standard deviation of the score over random n_lines samples of these lines.

    With n_lines = N_PUBLIC it is how far a public-LB score can sit from the
    model's true score through the luck of which lines were drawn.
    """
    idx = np.random.default_rng(seed).integers(0, len(errors), size=(n_draws, n_lines))
    col = lambda c: errors[c].to_numpy(float)[idx]  # noqa: E731
    wer = metric.weighted_rate(col("word_edits"), col("words"), weight_factor)
    cer = metric.weighted_rate(col("char_edits"), col("chars"), weight_factor)
    return float((metric.WER_WEIGHT * wer + metric.CER_WEIGHT * cer).std())


def report(err: pd.DataFrame, folds: pd.Series,
           weight_factor: float = metric.WEIGHT_FACTOR) -> None:
    """Print per-fold scores and diagnostics for an oof_errors() frame."""
    wf = weight_factor
    sizes = folds.value_counts()
    partial = {int(k): f"{n}/{sizes[k]}" for k, n in err.fold.value_counts().items()
               if n < sizes[k]}
    if partial:
        print(f"WARNING partially covered folds {partial} -- crashed or filtered run?\n")

    table = fold_table(err, wf)
    print(table.to_string(float_format=lambda x: f"{x:.4f}"))
    by_fold = table.drop(index="all")
    if len(by_fold) > 1:
        print(f"fold sd  {by_fold.score.std():.4f}")

    sens = "   ".join(f"{a:g} -> {metric.aggregate(err, a)['score']:.4f}"
                      for a in sorted({0.0, wf, 1.0}))
    print(f"\nweight_factor  {sens}")
    print(f"edits per line: {err.word_edits.mean():.3f} words, {err.char_edits.mean():.3f} chars"
          f"  -> public LB ~{metric.public_estimate(err):.3f}")
    print(f"lines with rate > 1 (insertion-heavy): "
          f"WER {(err.word_edits > err.words).sum()}, CER {(err.char_edits > err.chars).sum()}")
    n_priv = paths.N_TEST - N_PUBLIC
    print(f"LB sampling noise (1 sd): public {N_PUBLIC} lines "
          f"+/-{lb_noise(err, N_PUBLIC, weight_factor=wf):.4f}, "
          f"private {n_priv} lines +/-{lb_noise(err, n_priv, weight_factor=wf):.4f}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("oof", nargs="?", type=Path,
                    help="out-of-fold predictions CSV: ID,Target[,fold]")
    ap.add_argument("--weight-factor", type=float, default=metric.WEIGHT_FACTOR)
    args = ap.parse_args(argv)

    train = load_train()
    folds = make_folds(train.ID)
    sizes = folds.value_counts()
    print(f"folds  {N_FOLDS} x {sizes.min()}-{sizes.max()} lines   seed {SEED}   "
          f"fingerprint {fingerprint(folds)}")
    if args.oof is None:
        return 0

    oof = pd.read_csv(args.oof, keep_default_na=False, dtype=str)
    err = oof_errors(oof, train, folds)
    print(f"oof    {args.oof.name}: {len(err)}/{len(train)} train lines\n")
    report(err, folds, args.weight_factor)
    return 0


if __name__ == "__main__":
    sys.exit(main())
