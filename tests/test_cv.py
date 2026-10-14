"""CV harness: fold properties, leak-freedom of run_cv, OOF scoring. The last test
uses the real data and is skipped when data/ is absent."""

from __future__ import annotations

import pandas as pd
import pytest

from road import cv, metric, paths

IDS = [f"id{i:04d}" for i in range(103)]


def _toy_train() -> pd.DataFrame:
    return pd.DataFrame({"ID": IDS, "Target": [f"line number {i}" for i in range(len(IDS))]})


def test_folds_balanced_deterministic_and_order_free():
    a = cv.make_folds(IDS)
    b = cv.make_folds(list(reversed(IDS)))
    assert a.sort_index().equals(b.sort_index())
    assert sorted(a.index) == IDS
    sizes = a.value_counts()
    assert len(sizes) == cv.N_FOLDS and sizes.max() - sizes.min() <= 1
    assert cv.fingerprint(a) == cv.fingerprint(b)
    assert cv.fingerprint(cv.make_folds(IDS, seed=1)) != cv.fingerprint(a)


def test_run_cv_never_shows_validation_targets():
    seen = []

    def fit_predict(fit_part, valid_part):
        assert "Target" not in valid_part.columns
        assert not set(fit_part.ID) & set(valid_part.ID)
        seen.extend(valid_part.ID)
        return ["x"] * len(valid_part)

    oof = cv.run_cv(fit_predict, _toy_train())
    assert sorted(seen) == IDS and sorted(oof.ID) == IDS
    assert (oof.fold == oof.ID.map(cv.make_folds(IDS))).all()


def test_run_cv_rejects_wrong_prediction_count():
    with pytest.raises(ValueError, match="predictions"):
        cv.run_cv(lambda fit, valid: ["x"], _toy_train(), only=[0])


def test_oof_scoring_matches_metric_and_rejects_foreign_folds():
    train = _toy_train()
    folds = cv.make_folds(train.ID)
    oof = cv.run_cv(lambda fit, valid: ["line number"] * len(valid), train)
    table = cv.fold_table(cv.oof_errors(oof, train, folds))
    truth = train.set_index("ID").Target.loc[oof.ID]
    assert table.loc["all", "score"] == pytest.approx(metric.score(truth, oof.Target))
    assert table.loc["all", "n"] == len(IDS)

    shifted = oof.assign(fold=(oof.fold + 1) % cv.N_FOLDS)
    with pytest.raises(ValueError, match="fold"):
        cv.oof_errors(shifted, train, folds)


def test_lb_noise_vanishes_when_every_line_scores_the_same():
    train = _toy_train()
    oof = cv.run_cv(lambda fit, valid: [""] * len(valid), train)  # every line: rate 1
    err = cv.oof_errors(oof, train, cv.make_folds(train.ID))
    assert cv.lb_noise(err, 40) == pytest.approx(0.0)


@pytest.mark.skipif(not (paths.data_dir() / "Train.csv").exists(),
                    reason="dataset not present (python -m road.setup_data)")
def test_real_folds_and_empty_baseline():
    train = cv.load_train()
    folds = cv.make_folds(train.ID)
    sizes = folds.value_counts()
    assert len(folds) == paths.N_TRAIN and sizes.max() - sizes.min() <= 1
    oof = cv.run_cv(lambda fit, valid: [""] * len(valid), train, folds)
    # The page says empty output is "penalised as incorrect": exactly 1.0 here.
    assert metric.aggregate(cv.oof_errors(oof, train, folds))["score"] == pytest.approx(1.0)
