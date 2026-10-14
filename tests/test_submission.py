"""Submission building: cleaning, validation, and the written file as a default
CSV reader sees it."""

from __future__ import annotations

import pandas as pd
import pytest

from road import metric, submission

IDS = ["a1", "b2", "c3"]


def test_clean_never_returns_empty_or_multiline():
    got = [submission.clean(p) for p in ["", "  ", None, float("nan"), " y^e\nsaid "]]
    assert got == [submission.FALLBACK] * 4 + ["y^e said"]


def test_fallback_never_scores_worse_than_empty():
    refs = ["Signed and witnessed before the clerk of the court", "20th day of July 1640", "W^m Jones"]
    err = metric.line_errors(refs, [submission.FALLBACK] * len(refs))
    assert (err.word_edits <= err.words).all() and (err.char_edits <= err.chars).all()


def test_problems_catches_each_defect():
    good = pd.DataFrame({"ID": IDS, "Target": ["x", "y", "z"]})
    assert submission.problems(good, IDS) == []
    for bad in [good.rename(columns={"Target": "Pred"}),
                good.iloc[:2],                             # missing ID
                pd.concat([good, good.iloc[:1]]),          # duplicate ID
                good.assign(ID=["a1", "b2", "zz"]),        # foreign ID
                good.assign(Target=["x", " ", "z"]),       # blank prediction
                good.assign(Target=["x", "y\nz", "z"])]:   # multi-line prediction
        assert submission.problems(bad, IDS)


@pytest.fixture
def subdir(tmp_path, monkeypatch):
    monkeypatch.setattr(submission.paths, "submissions_dir", lambda: tmp_path)
    return tmp_path


def test_write_fills_gaps_and_survives_a_default_reader(subdir):
    path = submission.write(IDS, ["x", None, ""], "t", test_ids=IDS)
    assert pd.read_csv(path).Target.tolist() == ["x", submission.FALLBACK, submission.FALLBACK]


def test_write_refuses_text_a_reader_would_drop(subdir):
    with pytest.raises(ValueError, match="missing predictions"):
        submission.write(IDS, ["x", "NA", "z"], "t", test_ids=IDS)
    assert not (subdir / "t.csv").exists()
