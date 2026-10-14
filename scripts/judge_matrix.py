"""Which judges should pick the text? Every combination, measured on fold 0.

    ROAD_RUNS=... python scripts/judge_matrix.py oof_matrix_a oof_matrix_b oof_matrix_c \\
        --baseline qwen4b_f0 qwen4b_f0_tall

Reads the scores.csv that road.ensemble wrote with --oof -- one shared pool of fold-0
candidates, scored under every judge -- and for each subset of judges picks per line the
candidate with the best total, exactly as road.ensemble does, then scores the picks on
fold 0. Each candidate's errors are computed once, so all 255 subsets of eight judges
take seconds.

Why fold 0 and not the leaderboard: two runs of one recipe that differed only in their
seed scored 0.9066 and 0.9014 on the public 412 lines. Single scores there cannot tell
recipes apart, while a paired comparison on 820 held-out lines resolves about +/- 0.002.

Reports every judge alone, the oracle, the baseline subset, the best subset of each size
and the overall top, each against the baseline as a paired bootstrap.
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from road import cv, metric, paths  # noqa: E402


def load_matrix(ensembles: list[str]) -> pd.DataFrame:
    """One row per (ID, Target), one column per judge, from several scores.csv files."""
    merged = None
    for name in ensembles:
        frame = pd.read_csv(paths.runs_dir() / name / "scores.csv", keep_default_na=False,
                            dtype={"ID": str, "Target": str}).set_index(["ID", "Target"])
        merged = frame if merged is None else merged.join(
            frame[[c for c in frame.columns if c not in merged.columns]], how="outer")
    if merged is None:
        raise SystemExit("no ensembles given")
    if merged.isna().any().any():
        gaps = merged.isna().sum()
        raise SystemExit(f"judges missing scores for some candidates -- were the pools "
                         f"different? {gaps[gaps > 0].to_dict()}")
    return merged.reset_index()


def candidate_errors(matrix: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """metric.line_errors for every candidate against its line's true text."""
    truth = train.set_index("ID").Target.loc[matrix.ID]
    err = metric.line_errors(truth, matrix.Target)
    err.index = matrix.index
    return err


def pick(matrix: pd.DataFrame, judges: list[str]) -> np.ndarray:
    """Row index of each line's candidate with the best summed score."""
    total = matrix[list(judges)].sum(axis=1)
    return total.groupby(matrix.ID, sort=True).idxmax().to_numpy()


def paired(ea: pd.DataFrame, eb: pd.DataFrame, draws: int = 1000,
           seed: int = 0) -> tuple[float, float, float]:
    """(mean B - A, its sd, share of resamples favouring B) over the same lines."""
    a, b = ea.reset_index(drop=True), eb.reset_index(drop=True)
    idx = np.random.default_rng(seed).integers(0, len(a), size=(draws, len(a)))
    d = np.array([metric.public_estimate(b.iloc[k]) - metric.public_estimate(a.iloc[k])
                  for k in idx])
    return float(d.mean()), float(d.std()), float((d > 0).mean())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ensembles", nargs="+", help="ensemble runs written with --oof")
    ap.add_argument("--baseline", nargs="+", required=True, help="the judges to beat")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args(argv)

    matrix = load_matrix(args.ensembles)
    judges = [c for c in matrix.columns if c not in ("ID", "Target", "candidates", "margin")]
    if missing := set(args.baseline) - set(judges):
        raise SystemExit(f"baseline judges not in the matrix: {sorted(missing)}")
    train = cv.load_train()
    err = candidate_errors(matrix, train)
    n_lines = matrix.ID.nunique()
    print(f"{n_lines} lines, {len(matrix)} candidates ({len(matrix) / n_lines:.1f} per line), "
          f"{len(judges)} judges: {', '.join(judges)}\n")

    def errors_of(subset) -> pd.DataFrame:
        return err.loc[pick(matrix, subset)]

    score = {}
    for k in range(1, len(judges) + 1):
        for subset in combinations(judges, k):
            score[subset] = metric.public_estimate(errors_of(subset))
    oracle = err.assign(ID=matrix.ID, total=err.word_edits * 4.58 + err.char_edits)
    best_rows = oracle.groupby("ID", sort=True).total.idxmin().to_numpy()
    base = tuple(sorted(args.baseline, key=judges.index))
    base_err = errors_of(base)

    def line(name: str, subset_err: pd.DataFrame, s: float, with_ci: bool = True) -> str:
        if not with_ci:
            return f"  {s:.4f}  {name}"
        d, sd, share = paired(base_err, subset_err)
        return f"  {s:.4f}  {d:+.4f} +/- {sd:.4f} ({d / sd if sd else 0:+.1f} sd, {share:.0%})  {name}"

    print("each judge alone:")
    for j in judges:
        print(line(j, errors_of((j,)), score[(j,)], with_ci=False))
    print(f"\noracle (best candidate per line): {metric.public_estimate(err.loc[best_rows]):.4f}")
    print(f"baseline {' + '.join(base)}: {score[base]:.4f}\n")
    print("best subset of each size, vs the baseline (paired, 820 lines):")
    for k in range(1, len(judges) + 1):
        subset = max((s for s in score if len(s) == k), key=score.get)
        print(line(" + ".join(subset), errors_of(subset), score[subset]))
    print(f"\ntop {args.top} subsets overall:")
    for subset in sorted(score, key=score.get, reverse=True)[:args.top]:
        print(line(" + ".join(subset), errors_of(subset), score[subset]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
