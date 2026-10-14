"""Text-only baselines. They never look at the images: their job is to run the
CV -> submission path end to end and to set a floor every recogniser must beat.

    python -m road.baselines   # 5-fold CV, OOF to outputs/, submissions/baseline_modal.csv
"""

from __future__ import annotations

import sys

import pandas as pd

from road import cv, metric, paths, submission


def modal_line(fit: pd.DataFrame, valid: pd.DataFrame) -> list[str]:
    """The most common training line, for every row."""
    return [fit.Target.mode()[0]] * len(valid)


def main() -> int:
    train, test = cv.load_train(), submission.load_test()
    folds = cv.make_folds(train.ID)

    oof = cv.run_cv(modal_line, train, folds)
    oof_path = paths.outputs_dir() / "oof_baseline_modal.csv"
    oof.to_csv(oof_path, index=False)
    print(f"5-fold CV, line re-picked inside each fold (OOF -> {oof_path})\n")
    cv.report(cv.oof_errors(oof, train, folds), folds)

    preds = modal_line(train, test)
    path = submission.write(test.ID, preds, "baseline_modal")

    # Train and test lines are exchangeable (random split), so the submitted
    # constant's score on all of train estimates its test score directly. A public
    # LB far outside this band means the LB's weighting or display differs from ours.
    err = metric.line_errors(train.Target, preds[:1] * len(train))
    est = {a: metric.aggregate(err, a)["score"] for a in (0.0, metric.WEIGHT_FACTOR, 1.0)}
    print(f"\nsubmission  {path}\nconstant    {preds[0]!r}")
    print(f"expected public LB  {est[metric.WEIGHT_FACTOR]:.4f} "
          f"+/- {cv.lb_noise(err, cv.N_PUBLIC):.4f} (1 sd)   "
          f"[weight_factor 0: {est[0.0]:.4f}, 1: {est[1.0]:.4f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
