"""TrOCR pipeline on CPU: image transform, label round-trip, batching, and a tiny
random model through the real train/decode/save code in every precision the GPU
path can take. The `slow` test loads the real 1.3 GB checkpoint (pytest -m slow)."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from road import cv, data, paths, runs, trocr  # noqa: E402

needs_data = pytest.mark.skipif(not (paths.data_dir() / "Train.csv").exists(),
                                reason="dataset not present")


@pytest.fixture(scope="module")
def tok():
    return trocr.load_tokenizer(trocr.MODEL_ID)


def _line_image(w: int = 1110, h: int = 62) -> Image.Image:
    im = Image.new("L", (w, h), 235)
    ImageDraw.Draw(im).text((10, h // 3), "Sealed and delivered", fill=20)
    return im


def test_transform_shape_range_and_determinism():
    im = _line_image()
    ev = data.line_transform((384, 384))
    a = ev(im)
    assert a.shape == (1, 384, 384) and torch.equal(a, ev(im))
    assert a.min() >= -1 and a.max() <= 1
    tr = data.line_transform((384, 384), train=True)
    torch.manual_seed(0)
    outs = [tr(im) for _ in range(6)]
    assert all(o.shape == (1, 384, 384) for o in outs)
    assert not all(torch.equal(outs[0], o) for o in outs[1:])


@needs_data
def test_eval_transform_matches_the_checkpoints_image_processor():
    # The image processor alone: TrOCRProcessor would also build the tokenizer through
    # AutoTokenizer, which fails for this checkpoint (see trocr.load_tokenizer).
    proc = transformers.AutoImageProcessor.from_pretrained(trocr.MODEL_ID)
    for image_id in cv.load_train().ID[:4]:
        im = data.load_line(image_id)
        ours = data.line_transform((384, 384))(im).repeat(3, 1, 1)
        theirs = proc(images=im.convert("RGB"), return_tensors="pt").pixel_values[0]
        assert ours.shape == theirs.shape
        assert (ours - theirs).abs().mean() < 0.02


@needs_data
def test_every_training_target_survives_tokenisation(tok):
    targets = [runs.clean_target(t) for t in cv.load_train().Target]
    back = trocr.decode_ids(tok, trocr.encode(tok, targets))
    bad = [(t, b) for t, b in zip(targets, back) if t != b]
    assert not bad, bad[:5]


def test_labels_are_text_then_eos_without_bos(tok):
    ids = trocr.encode(tok, ["By this publique Act"])[0]
    assert ids[-1] == tok.eos_token_id and tok.bos_token_id not in ids
    # First token without a prefix space -- the token the checkpoint emits for "By".
    assert tok.convert_ids_to_tokens(ids[:1]) == ["By"]


def test_shift_right_and_collate():
    y = torch.tensor([[5, 6, 2, -100], [7, 2, -100, -100]])
    assert trocr.shift_right(y, start_id=2, pad_id=1).tolist() == [[2, 5, 6, 2], [2, 7, 2, 1]]
    xs, ys = trocr.collate([(torch.zeros(3, 4, 4), torch.tensor([0, 7, 2])),
                            (torch.zeros(3, 4, 4), torch.tensor([0, 2]))])
    assert xs.shape == (2, 3, 4, 4) and ys.tolist() == [[0, 7, 2], [0, 2, -100]]
    assert trocr.collate([(torch.zeros(3, 4, 4), None)])[1] is None


def test_cli_parsing():
    cfg, predict_only = trocr.parse(["--run", "x", "--smoke", "--no-augment", "--lr", "1e-4"])
    assert not predict_only and cfg.smoke and not cfg.augment and cfg.lr == 1e-4
    assert cfg.epochs == trocr.SMOKE_EPOCHS and cfg.workers <= trocr.SMOKE_WORKERS
    for bad in (["--run", "x", "--fold", "5"], ["--run", "x", "--precision", "fp8"]):
        with pytest.raises(SystemExit):
            trocr.parse(bad)


@pytest.mark.parametrize("precision", ["fp32", "bf16", "fp16"])
def test_tiny_model_trains_decodes_and_round_trips(tok, precision, tmp_path):
    torch.manual_seed(0)
    model = trocr.tiny_model(tok)
    start = trocr.start_token(model)
    labels = trocr.encode(tok, ["y^e said W^m", "Sealed and delivered", "20th of July 1640", "&c ~"])
    items = [(torch.randn(3, 32, 32), torch.tensor(y)) for y in labels]
    batches = [trocr.collate(items[:2]), trocr.collate(items[2:])]
    cfg = trocr.Config(run="t", precision=precision, grad_accum=2)
    device = torch.device("cpu")
    dtype = runs.amp_dtype(device, precision)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    scaler = torch.amp.GradScaler("cpu", enabled=dtype is torch.float16)
    before = [p.detach().clone() for p in model.parameters()]

    loss = trocr.train_epoch(model, batches, opt, sched, scaler, device, dtype, cfg,
                             start, tok.pad_token_id)
    assert math.isfinite(loss)
    changed = any(not torch.equal(b, p) for b, p in zip(before, model.parameters()))
    # fp16: GradScaler may skip an overflowing first step and lower its scale instead.
    assert changed or (dtype is torch.float16 and scaler.get_scale() < 2.0**16)

    gen = trocr.generation_config(tok, start, max_new_tokens=6, beams=2)
    preds = trocr.decode(model, [(torch.randn(2, 3, 32, 32), None)], tok, gen, device, dtype)
    # decode returns the kept beams per line, one here since gen asks for one
    assert len(preds) == 2 and all(len(c) == 1 and isinstance(c[0], str) for c in preds)

    model.save_pretrained(tmp_path)
    tok.save_pretrained(tmp_path)
    again = transformers.VisionEncoderDecoderModel.from_pretrained(tmp_path)
    assert trocr.start_token(again) == start
    line = ["y^e said W^m &c  ~ 1640 ."]
    assert trocr.encode(trocr.load_tokenizer(tmp_path), line) == trocr.encode(tok, line)
    for (k, a), b in zip(model.state_dict().items(), again.state_dict().values()):
        assert torch.equal(a, b), k


@needs_data
def test_fold_all_training_then_predict_only(tmp_path, monkeypatch):
    """The two paths the smoke runs skip: --fold all (no held-out set, weights saved at
    the end) and --predict-only (re-decode a saved run with other beams)."""
    monkeypatch.setenv("ROAD_RUNS", str(tmp_path / "runs"))
    monkeypatch.setattr(paths, "submissions_dir", lambda: tmp_path)
    cfg, _ = trocr.parse(["--run", "r", "--fold", "all", "--smoke", "--model", "tiny",
                          "--workers", "0", "--batch-size", "4"])
    trocr.train(cfg)
    run = tmp_path / "runs" / "r"
    assert (run / "model" / "config.json").exists() and not (run / "oof.csv").exists()

    assert trocr.main(["--run", "r", "--predict-only", "--beams", "2"]) == 0
    sub = pd.read_csv(run / "r.csv", keep_default_na=False)
    assert len(sub) == 2 * cfg.batch_size and (sub.Target.str.len() > 0).all()


@pytest.mark.slow
@needs_data
def test_real_checkpoint_through_our_forward_and_decode(tok):
    """The real checkpoint through our forward and decode code. Also pins the label
    layout: after its start token the pretrained decoder goes straight to text -- no
    <s> -- which is why encode() emits `text </s>`."""
    model = transformers.VisionEncoderDecoderModel.from_pretrained(trocr.MODEL_ID).eval()
    start = trocr.start_token(model)
    size = (model.config.encoder.image_size,) * 2
    ds = trocr.Lines(cv.load_train().ID[:2], size)
    pv, _ = trocr.collate([ds[0], ds[1]])
    y = trocr.collate([(pv[0], torch.tensor(t)) for t in trocr.encode(tok, ["the said", "Witness"])])[1]
    with torch.inference_mode():
        logits = model(pixel_values=pv, decoder_input_ids=trocr.shift_right(y, start, tok.pad_token_id),
                       use_cache=False).logits
        ids = model.generate(pixel_values=pv,
                             generation_config=trocr.generation_config(tok, start, 12, beams=2))
    assert logits.shape[:2] == y.shape and logits.shape[-1] == model.config.decoder.vocab_size
    assert ids[:, 0].eq(start).all(), ids[:, :3]
    special = {tok.bos_token_id, tok.eos_token_id, tok.pad_token_id}
    assert not set(ids[:, 1].tolist()) & special, tok.convert_ids_to_tokens(ids[:, 1].tolist())
    assert all(t.strip() for t in trocr.decode_ids(tok, ids))


@needs_data
def test_trocr_takes_the_same_drop_and_pseudo_levers(tmp_path):
    """The levers live in runs.shape_fit so every trainer gets them. TrOCR is the
    model most short of data, so pseudo-labels are worth most to it."""
    cfg = trocr.Config(run="x", fold="0")
    fit, _, test = trocr.splits(cfg)

    dropped = tmp_path / "bad.csv"
    pd.DataFrame({"ID": list(fit.ID[:3]), "why": "mislabelled"}).to_csv(dropped, index=False)
    kept = runs.shape_fit(fit, trocr.Config(run="x", fold="0", drop=str(dropped)))
    assert len(kept) == len(fit) - 3

    extra = tmp_path / "pseudo.csv"
    pd.DataFrame({"ID": list(test.ID[:4]), "Target": ["the said"] * 3 + [""]}).to_csv(
        extra, index=False)
    grown = runs.shape_fit(fit, trocr.Config(run="x", fold="0", pseudo=str(extra)))
    assert len(grown) == len(fit) + 3          # the empty label is not trained on


def test_decode_keeps_the_beams_it_is_asked_for(tok):
    """TrOCR is our only architecturally different source, so its alternative beams are
    the most valuable ones in the pool."""
    torch.manual_seed(0)
    model = trocr.tiny_model(tok).eval()
    start = trocr.start_token(model)
    batch = [(torch.randn(2, 3, 32, 32), None)]

    cpu = torch.device("cpu")
    one = trocr.decode(model, batch, tok,
                       trocr.generation_config(tok, start, 6, 4), cpu, None)
    three = trocr.decode(model, batch, tok,
                         trocr.generation_config(tok, start, 6, 4, n_best=3), cpu, None)
    assert [len(c) for c in one] == [1, 1]
    assert [len(c) for c in three] == [3, 3]
    assert [c[0] for c in three] == [c[0] for c in one]   # best beam unchanged
