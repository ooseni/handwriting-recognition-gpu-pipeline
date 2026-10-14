"""Combining runs: candidate gathering, the joint-score choice, the log-probability
itself, and the whole path on a tiny run with the real processor."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

import pandas as pd  # noqa: E402

from road import ensemble, paths, qwenvl  # noqa: E402

needs_data = pytest.mark.skipif(not (paths.data_dir() / "Train.csv").exists(),
                                reason="dataset not present")


def _run_with(tmp_path, name, rows):
    run = tmp_path / "runs" / name
    run.mkdir(parents=True)
    pd.DataFrame(rows, columns=["ID", "Target"]).to_csv(run / f"{name}.csv", index=False)
    return run


def test_candidates_are_deduplicated_per_line(tmp_path, monkeypatch):
    monkeypatch.setenv("ROAD_RUNS", str(tmp_path / "runs"))
    _run_with(tmp_path, "a", [("l1", "the said"), ("l2", "heires and assignes")])
    _run_with(tmp_path, "b", [("l1", "the said"), ("l2", "heires & assignes")])
    per_line = ensemble.candidates(["a", "b"])
    assert per_line == {"l1": ["the said"], "l2": ["heires and assignes", "heires & assignes"]}
    with pytest.raises(SystemExit, match="missing"):
        ensemble.candidates(["a", "nope"])


def test_choose_takes_the_jointly_best_candidate():
    scores = pd.DataFrame({
        "ID": ["l1", "l1", "l2", "l2"],
        "Target": ["good", "bad", "x", "y"],
        "a": [-0.10, -0.20, -0.50, -0.40],   # run a prefers "y" for l2
        "b": [-0.15, -0.30, -0.30, -0.60],   # run b prefers "x", and more strongly
    })
    picked = ensemble.choose(scores, ["a", "b"]).set_index("ID")
    assert picked.loc["l1", "Target"] == "good"
    assert picked.loc["l2", "Target"] == "x"   # -0.80 total beats -1.00
    assert picked.loc["l1", "candidates"] == 2
    assert picked.loc["l2", "margin"] == pytest.approx(0.20)


def test_answer_logprob_is_the_training_loss_negated(tmp_path):
    torch.manual_seed(0)
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(qwenvl.MODEL_ID)
    model = qwenvl.tiny_model().eval()
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (640, 40), (235, 225, 200))
    ImageDraw.Draw(im).text((8, 12), "Sealed and delivered", fill=(40, 30, 20))
    batch = qwenvl.Collate(proc)([(qwenvl.resize(im, 32, 3200), "Sealed and delivered")])
    with torch.no_grad():
        logprob = ensemble.answer_logprob(model, batch)
        loss = qwenvl.answer_loss(model, batch)
    assert logprob.shape == (1,)
    assert logprob.item() == pytest.approx(-loss.item(), rel=1e-4)


@needs_data
def test_ensemble_scores_a_tiny_run_and_writes_a_submission(tmp_path, monkeypatch):
    monkeypatch.setenv("ROAD_RUNS", str(tmp_path / "runs"))
    monkeypatch.setattr(paths, "submissions_dir", lambda: tmp_path)
    cfg, _ = qwenvl.parse(["--run", "r", "--fold", "all", "--smoke", "--model", "tiny",
                           "--workers", "0", "--batch-size", "4"])
    qwenvl.train(cfg)
    qwenvl.predict(cfg)

    assert ensemble.main(["--runs", "r", "r", "--out", "ens", "--partial",
                             "--batch-size", "4"]) == 0
    out = tmp_path / "runs" / "ens"
    sub = pd.read_csv(out / "ens.csv", keep_default_na=False)
    assert len(sub) == 2 * cfg.batch_size and (sub.Target.str.len() > 0).all()
    scores = pd.read_csv(out / "scores.csv")
    assert len(scores) == len(sub) and (scores["r"] <= 0).all()  # log-probabilities

    # a run may propose text without judging it: TrOCR is right where the Qwen runs
    # are not, but is a poor judge of its own output.
    _run_with(tmp_path, "alt", [(i, f"{t} x") for i, t in zip(sub.ID, sub.Target)])
    assert ensemble.main(["--runs", "r", "--candidates-from", "alt", "--out", "ens2",
                          "--partial", "--batch-size", "4"]) == 0
    scored = pd.read_csv(tmp_path / "runs" / "ens2" / "scores.csv", keep_default_na=False)
    assert list(scored.columns) == ["ID", "Target", "r"]   # alt scored nothing
    assert len(scored) == 2 * len(sub)                     # yet its text was considered

    # the same ensemble again, reusing ens2: nothing may be scored a second time, and
    # what comes back must be exactly what was stored
    monkeypatch.setattr(ensemble, "score", lambda *a, **k: pytest.fail("rescored"))
    assert ensemble.main(["--runs", "r", "--candidates-from", "alt", "--out", "ens3",
                          "--reuse", "ens2", "--partial", "--batch-size", "4"]) == 0
    again = pd.read_csv(tmp_path / "runs" / "ens3" / "scores.csv", keep_default_na=False)
    pd.testing.assert_frame_equal(again, scored)


def test_trocr_logprob_is_its_training_loss_negated():
    """The TrOCR judge must report the same quantity as the Qwen one -- mean
    log-probability per answer token -- or summing their scores is meaningless."""
    from road import trocr

    torch.manual_seed(0)
    tok = trocr.load_tokenizer(trocr.MODEL_ID)
    model = trocr.tiny_model(tok).eval()
    start, pad = trocr.start_token(model), tok.pad_token_id
    size = model.config.encoder.image_size

    labels = torch.tensor(trocr.encode(tok, ["Sealed and delivered"])[0]).unsqueeze(0)
    pixel_values = torch.rand(1, 3, size, size)
    with torch.no_grad():
        logprob = ensemble.decoder_logprob(model, pixel_values, labels, start, pad)
        logits = model(pixel_values=pixel_values,
                       decoder_input_ids=trocr.shift_right(labels, start, pad),
                       use_cache=False).logits
        loss = torch.nn.functional.cross_entropy(
            logits.float().flatten(0, 1), labels.flatten(), ignore_index=-100)
    assert logprob.shape == (1,)
    assert logprob.item() == pytest.approx(-loss.item(), rel=1e-4)


def test_padding_is_ignored_when_scoring_a_batch():
    """Short candidates are padded with -100; if those positions leaked into the mean,
    a short line's score would depend on whatever it was batched with."""
    from road import trocr

    torch.manual_seed(0)
    tok = trocr.load_tokenizer(trocr.MODEL_ID)
    model = trocr.tiny_model(tok).eval()
    start, pad = trocr.start_token(model), tok.pad_token_id
    size = model.config.encoder.image_size

    short = torch.tensor(trocr.encode(tok, ["ye"])[0])
    long = torch.tensor(trocr.encode(tok, ["To all Christian people"])[0])
    pixel_values = torch.rand(1, 3, size, size)

    with torch.no_grad():
        alone = ensemble.decoder_logprob(model, pixel_values, short.unsqueeze(0), start, pad)
        padded = torch.nn.utils.rnn.pad_sequence([short, long], batch_first=True,
                                                 padding_value=-100)
        together = ensemble.decoder_logprob(model, pixel_values.repeat(2, 1, 1, 1),
                                            padded, start, pad)
    assert together[0].item() == pytest.approx(alone.item(), rel=1e-4)


def test_load_judge_picks_by_what_the_run_saved(tmp_path, monkeypatch):
    monkeypatch.setenv("ROAD_RUNS", str(tmp_path))
    (tmp_path / "bare").mkdir()
    (tmp_path / "bare" / "bare.csv").write_text("ID,Target\nl1,x\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="--candidates-from"):
        ensemble.load_judge("bare")
    with pytest.raises(SystemExit, match="--candidates-from"):
        ensemble.load_judge("missing")


@needs_data
def test_a_trocr_run_judges_through_the_cli(tmp_path, monkeypatch):
    """The path that will run on Colab: a TrOCR run loaded as a judge by --runs, with
    no adapter anywhere in sight."""
    from road import trocr

    monkeypatch.setenv("ROAD_RUNS", str(tmp_path / "runs"))
    monkeypatch.setattr(paths, "submissions_dir", lambda: tmp_path)
    cfg, _ = trocr.parse(["--run", "t", "--fold", "all", "--smoke", "--model", "tiny",
                          "--workers", "0", "--batch-size", "4"])
    trocr.train(cfg)          # train/predict directly: main() deletes smoke weights
    trocr.predict(cfg)
    assert (tmp_path / "runs" / "t" / "model").is_dir()
    assert isinstance(ensemble.load_judge("t"), ensemble.TrocrJudge)

    assert ensemble.main(["--runs", "t", "--out", "te", "--partial",
                          "--batch-size", "4"]) == 0
    scores = pd.read_csv(tmp_path / "runs" / "te" / "scores.csv", keep_default_na=False)
    sub = pd.read_csv(tmp_path / "runs" / "te" / "te.csv", keep_default_na=False)
    assert len(scores) == len(sub) and (scores["t"] <= 0).all()


def test_candidates_prefer_the_nbest_file(tmp_path, monkeypatch):
    """A run's nbest.csv holds every beam it kept. When one exists the pool should draw
    on all of them -- more candidates per line is the point of keeping them."""
    monkeypatch.setenv("ROAD_RUNS", str(tmp_path / "runs"))
    run = tmp_path / "runs" / "a"
    run.mkdir(parents=True)
    pd.DataFrame({"ID": ["l1", "l2"], "Target": ["the said", "heires"]}).to_csv(
        run / "a.csv", index=False)
    assert ensemble.candidates(["a"]) == {"l1": ["the said"], "l2": ["heires"]}

    pd.DataFrame({"ID": ["l1", "l1", "l2"], "rank": [0, 1, 0],
                  "Target": ["the said", "ye said", "heires"]}).to_csv(
        run / "nbest.csv", index=False)
    assert ensemble.candidates(["a"]) == {"l1": ["the said", "ye said"], "l2": ["heires"]}

    pd.DataFrame({"ID": ["l1"], "rank": [0], "Target": ["held out"]}).to_csv(
        run / "oof_nbest.csv", index=False)
    assert ensemble.candidates(["a"], oof=True) == {"l1": ["held out"]}


def test_beams_caps_how_many_kept_beams_enter_the_pool(tmp_path, monkeypatch):
    """Scoring is one forward pass per candidate per judge, so the pool size is the
    cost. The beams are rank-ordered, so a cap drops the least likely first."""
    monkeypatch.setenv("ROAD_RUNS", str(tmp_path / "runs"))
    run = tmp_path / "runs" / "a"
    run.mkdir(parents=True)
    pd.DataFrame({"ID": ["l1"] * 4, "rank": [0, 1, 2, 3],
                  "Target": ["best", "second", "third", "fourth"]}).to_csv(
        run / "nbest.csv", index=False)

    assert ensemble.candidates(["a"])["l1"] == ["best", "second", "third", "fourth"]
    assert ensemble.candidates(["a"], beams=2)["l1"] == ["best", "second"]
    assert ensemble.candidates(["a"], beams=1)["l1"] == ["best"]

    # a run with no nbest.csv is unaffected by the cap
    pd.DataFrame({"ID": ["l1"], "Target": ["only"]}).to_csv(run / "a.csv", index=False)
    (run / "nbest.csv").unlink()
    assert ensemble.candidates(["a"], beams=1)["l1"] == ["only"]


def test_reuse_scores_only_what_no_earlier_ensemble_holds(tmp_path, monkeypatch):
    """Scoring is the whole cost of an ensemble. A pair a judge already scored must not
    be scored again, and its stored score must come back exactly -- including texts
    pandas would otherwise read as missing."""
    monkeypatch.setenv("ROAD_RUNS", str(tmp_path / "runs"))
    for name, frame in {
        "old": {"ID": ["l1", "l1", "l2"], "Target": ["the said", "NA", ""],
                "a": [-0.125, -0.5, -0.75], "b": [-0.25, -1.0, -1.5]},
        "older": {"ID": ["l2"], "Target": ["new text"], "c": [-2.0]},
    }.items():
        (tmp_path / "runs" / name).mkdir(parents=True)
        pd.DataFrame(frame).to_csv(tmp_path / "runs" / name / "scores.csv", index=False)
    asked = []

    def fake(run, pairs, batch_size=16):
        asked.extend((run, p) for p in pairs)
        return [-9.0] * len(pairs)

    monkeypatch.setattr(ensemble, "score", fake)
    known = ensemble.known_scores(["old", "older"])
    pairs = [("l1", "the said"), ("l1", "NA"), ("l2", ""), ("l2", "new text")]

    assert ensemble.score_reusing("a", pairs, known) == [-0.125, -0.5, -0.75, -9.0]
    assert asked == [("a", ("l2", "new text"))]          # only the pair "old" lacked
    asked.clear()
    assert ensemble.score_reusing("c", pairs, known) == [-9.0, -9.0, -9.0, -2.0]
    assert [p for _, p in asked] == pairs[:3]            # "older" held the last one
    asked.clear()
    assert ensemble.score_reusing("d", pairs, ensemble.known_scores([])) == [-9.0] * 4
    assert len(asked) == 4                               # a judge nobody used: all of it
    with pytest.raises(SystemExit, match="missing"):
        ensemble.known_scores(["nope"])
