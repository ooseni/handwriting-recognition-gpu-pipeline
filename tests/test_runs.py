"""Shared training plumbing: precision policy and data splits."""

from __future__ import annotations

import dataclasses
import json

import pytest

torch = pytest.importorskip("torch")

from road import cv, paths, runs  # noqa: E402


def test_amp_policy_on_cpu():
    cpu = torch.device("cpu")
    assert runs.amp_dtype(cpu, "auto") is None
    assert runs.amp_dtype(cpu, "bf16") is torch.bfloat16
    assert runs.amp_dtype(cpu, "fp16") is torch.float16
    assert runs.amp_dtype(cpu, "fp32") is None


@pytest.mark.skipif(not (paths.data_dir() / "Train.csv").exists(), reason="dataset not present")
def test_splits_follow_the_cv_folds():
    fit, valid, test = runs.splits("0")
    folds = cv.make_folds(cv.load_train().ID)
    assert set(valid.ID) == set(folds[folds == 0].index)
    assert not set(fit.ID) & set(valid.ID) and len(fit) + len(valid) == paths.N_TRAIN
    assert len(test) == paths.N_TEST

    fit, valid, test = runs.splits("all", smoke_lines=8)
    assert valid is None and len(fit) == len(test) == 8


def test_claim_refuses_to_overwrite_a_finished_run(tmp_path):
    run = tmp_path / "r"
    run.mkdir()
    runs.claim(run, overwrite=False, smoke=False)  # nothing saved there yet
    (run / "config.json").write_text("{}")
    with pytest.raises(SystemExit, match="another --run name"):
        runs.claim(run, overwrite=False, smoke=False)
    runs.claim(run, overwrite=True, smoke=False)   # explicit replacement is allowed
    runs.claim(run, overwrite=False, smoke=True)   # smoke runs are throwaway


def test_saved_config_defaults_fields_the_run_predates(tmp_path):
    """A run trained before a setting existed must still load: its config.json has no
    entry for that field, and the finished runs on Drive are worth more than the code."""
    from road import qwenvl

    written = {f.name: getattr(qwenvl.Config(run="old"), f.name)
               for f in dataclasses.fields(qwenvl.Config)}
    written.pop("keep_weights")            # as an older version would have saved it
    written["beams"] = 5
    (tmp_path / "config.json").write_text(json.dumps(written))

    cfg = runs.saved_config(qwenvl.Config, tmp_path)
    assert cfg.beams == 5 and cfg.keep_weights is False
    assert runs.saved_config(qwenvl.Config, tmp_path, beams=1).beams == 1


def test_predict_only_keeps_the_decode_time_flags(tmp_path, monkeypatch):
    """A finished run can be re-decoded for more candidates without retraining it, so
    --beams and --n-best must survive the config reload while everything else does not."""
    from road import qwenvl

    monkeypatch.setenv("ROAD_RUNS", str(tmp_path))
    run = tmp_path / "r"
    run.mkdir()
    trained = qwenvl.Config(run="r", beams=3, n_best=1, epochs=7, lr=0.005)
    (run / "config.json").write_text(json.dumps(dataclasses.asdict(trained)))

    cfg, predict_only = qwenvl.parse(["--run", "r", "--predict-only",
                                      "--beams", "8", "--n-best", "5", "--epochs", "99",
                                      "--decode-batch-size", "2"])
    assert predict_only
    assert (cfg.beams, cfg.n_best) == (8, 5)     # decode-time: taken from the CLI
    assert cfg.decode_batch_size == 2            # what lets an L4 re-decode an A100 run
    assert (cfg.epochs, cfg.lr) == (7, 0.005)    # training: taken from the run
