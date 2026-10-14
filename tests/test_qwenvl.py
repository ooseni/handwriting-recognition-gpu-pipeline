"""Qwen-VL pipeline on CPU with the real processor: resizing, chat formatting,
answer-only labels, the trimmed-logits loss, LoRA placement, and a tiny random Qwen3-VL
through the real train/decode/adapter code in every precision the GPU path can take."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

import pandas as pd  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from peft import PeftModel  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

from road import cv, data, paths, qwenvl, runs  # noqa: E402

needs_data = pytest.mark.skipif(not (paths.data_dir() / "Train.csv").exists(),
                                reason="dataset not present")
TEXTS = ["y^e said W^m &c", "Signed and witnessed before the clerk of the court"]


@pytest.fixture(scope="module")
def proc():
    return AutoProcessor.from_pretrained(qwenvl.MODEL_ID)


@pytest.fixture
def tiny():
    torch.manual_seed(0)
    return qwenvl.tiny_model()


def _line(text: str = "Sealed and delivered", w: int = 640, h: int = 40) -> Image.Image:
    im = Image.new("RGB", (w, h), (235, 225, 200))
    ImageDraw.Draw(im).text((8, h // 3), text, fill=(40, 30, 20))
    return qwenvl.resize(im, 32, 3200)  # small: the processor upscales to its minimum


def _batch(proc, texts=TEXTS):
    return qwenvl.Collate(proc)([(_line(t), t) for t in texts])


def test_resize_keeps_aspect_and_caps_width():
    assert qwenvl.resize(Image.new("RGB", (1110, 62)), 160, 3200).size == (2865, 160)
    assert qwenvl.resize(Image.new("RGB", (6000, 100)), 160, 3200).size == (3200, 160)


def _ruled(w=1200, h=300, bands=((40, 60), (130, 170), (250, 270))) -> Image.Image:
    """A crop with several ink bands, like the neighbours that bleed into a real one."""
    im = Image.new("RGB", (w, h), (235, 225, 200))
    draw = ImageDraw.Draw(im)
    for top, bottom in bands:
        draw.rectangle([0, top, w, bottom], fill=(40, 30, 20))
    return im


def test_band_crop_is_off_by_default():
    cfg = qwenvl.Config(run="t")  # the plain squash that scored best
    plain = qwenvl.resize(_ruled(1200, 100), cfg.height, cfg.max_width)
    assert qwenvl.prepare(_ruled(1200, 100), cfg).size == plain.size


def test_ink_band_picks_the_central_line():
    band = data.ink_band(_ruled())
    assert band is not None and abs(band[0] - 130) <= 3 and abs(band[1] - 170) <= 3
    assert data.ink_band(Image.new("RGB", (400, 80), (235, 225, 200))) is None


def test_prepare_gives_both_scan_scales_the_same_text_size():
    cfg = qwenvl.Config(run="t", band=True)
    lo = qwenvl.prepare(_ruled(1200, 100, ((18, 26), (45, 60), (85, 94))), cfg)
    hi = qwenvl.prepare(_ruled(4800, 400, ((70, 104), (180, 240), (330, 368))), cfg)
    for out in (lo, hi):  # neighbours cropped away, text scaled to cfg.band_px
        band = data.ink_band(out)
        assert band is not None and abs((band[1] - band[0]) - cfg.band_px) <= 0.25 * cfg.band_px


def test_width_floor_lifts_a_narrow_hand():
    cfg = qwenvl.Config(run="t", band=True)
    narrow = _ruled(2000, 400, ((70, 104), (180, 240), (330, 368)))
    assert qwenvl.prepare(narrow, cfg).width / 2000 > 1.05 * cfg.band_px / 60


def test_prepare_falls_back_when_no_band_stands_out():
    cfg = qwenvl.Config(run="t", band=True)
    blank = Image.new("RGB", (1200, 90), (235, 225, 200))
    assert qwenvl.prepare(blank, cfg).height == cfg.height


def test_prepare_clamps_the_aspect_ratio_the_processor_refuses(proc):
    cfg = qwenvl.Config(run="t", band=True)
    out = qwenvl.prepare(_ruled(9000, 300, ((145, 155),)), cfg)  # a thin band, very wide crop
    assert out.width / out.height <= 200
    proc.image_processor(out, return_tensors="pt")  # the component that raised on 204.8


def test_prepare_respects_the_token_budget():
    cfg = qwenvl.Config(run="t", band=True)
    out = qwenvl.prepare(_ruled(9000, 300, ((60, 80), (130, 170), (240, 260))), cfg)
    assert (out.width // 32) * (out.height // 32) <= cfg.max_tokens


def test_prompt_plus_answer_is_the_chat_template(proc):
    col = qwenvl.Collate(proc)
    user = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": qwenvl.PROMPT}]}]
    answer = [{"role": "assistant", "content": [{"type": "text", "text": TEXTS[1]}]}]
    full = proc.apply_chat_template(user + answer, tokenize=False)
    assert col.prompt + TEXTS[1] + col.eos == full.rstrip("\n")


def test_left_padding_and_labels_cover_exactly_the_answer(proc):
    b = _batch(proc)
    tok, mask = proc.tokenizer, b["attention_mask"]
    # Left padding: every row is 0...0 1...1, so answers end every row.
    assert (mask[:, -1] == 1).all() and (mask[:, 0] == 0).any() and (mask.diff(dim=1) >= 0).all()
    for r, text in enumerate(TEXTS):
        lab = b["labels"][r]
        assert tok.decode(lab[lab != -100]) == text + tok.eos_token


def test_trimmed_logits_give_the_full_loss(proc, tiny):
    b = _batch(proc)
    with torch.no_grad():
        full = tiny.eval()(**{k: v for k, v in b.items() if k != "labels"}).logits
        want = F.cross_entropy(full[:, :-1].flatten(0, 1), b["labels"][:, 1:].flatten(),
                               ignore_index=-100)
        got = qwenvl.answer_loss(tiny, b)
    assert got.item() == pytest.approx(want.item(), rel=1e-5)


def test_lora_reaches_vision_and_text_but_not_the_head(tiny):
    model = qwenvl.with_lora(tiny, qwenvl.Config(run="t"))
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert names and all("lora_" in n for n in names)
    assert any("visual" in n for n in names) and any("language_model" in n for n in names)
    assert not any("lm_head" in n for n in names)


def test_incompatible_torchao_fails_fast_with_the_fix(monkeypatch):
    def colab_torchao():
        raise ImportError("Found an incompatible version of torchao. Found version 0.10.0")

    monkeypatch.setattr(qwenvl.peft.import_utils, "is_torchao_available", colab_torchao)
    with pytest.raises(SystemExit, match="pip uninstall -y torchao"):
        qwenvl.check_peft()


def test_generation_never_samples(proc):
    gen = qwenvl.generation_config(proc.tokenizer, 40, beams=3)
    assert gen.do_sample is False and gen.num_beams == 3 and gen.max_new_tokens == 40
    assert proc.tokenizer.eos_token_id in gen.eos_token_id


def test_cli_parsing():
    cfg, predict_only = qwenvl.parse(["--run", "x", "--smoke", "--lora-r", "8", "--no-checkpointing"])
    assert not predict_only and cfg.smoke and cfg.lora_r == 8 and not cfg.checkpointing
    assert cfg.epochs == qwenvl.SMOKE_EPOCHS and cfg.workers <= qwenvl.SMOKE_WORKERS
    with pytest.raises(SystemExit):
        qwenvl.parse(["--run", "x", "--fold", "9"])


@pytest.mark.parametrize("precision", ["fp32", "bf16", "fp16"])
def test_tiny_model_trains_decodes_and_round_trips(proc, tiny, precision, tmp_path):
    cfg = qwenvl.Config(run="t", precision=precision, grad_accum=2, lora_dropout=0.0)
    device = torch.device("cpu")
    dtype = runs.amp_dtype(device, precision)
    tiny.save_pretrained(tmp_path / "base")
    model = qwenvl.with_lora(tiny, cfg)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    scaler = torch.amp.GradScaler("cpu", enabled=dtype is torch.float16)
    before = [p.detach().clone() for p in params]

    loss = qwenvl.train_epoch(model, [_batch(proc), _batch(proc, TEXTS[::-1])], opt, sched,
                              scaler, device, dtype, cfg)
    assert math.isfinite(loss)
    changed = any(not torch.equal(b, p) for b, p in zip(before, params))
    # fp16: GradScaler may skip an overflowing first step and lower its scale instead.
    assert changed or (dtype is torch.float16 and scaler.get_scale() < 2.0**16)

    prompts = qwenvl.Collate(proc)([(_line(), None)] * 2)
    gen = qwenvl.generation_config(proc.tokenizer, 6, beams=2)
    preds = [c[0] for c in qwenvl.decode(model, [prompts], proc.tokenizer,
                                         gen, device, dtype)]
    assert len(preds) == 2 and all(isinstance(p, str) for p in preds)

    model.save_pretrained(tmp_path / "adapter")
    base = AutoModelForImageTextToText.from_pretrained(tmp_path / "base", dtype=torch.float32)
    again = PeftModel.from_pretrained(base, tmp_path / "adapter").merge_and_unload().eval()
    model.eval()
    with torch.no_grad():
        a, b = model(**prompts).logits, again(**prompts).logits
    assert torch.allclose(a, b, atol=1e-4)


@needs_data
def test_real_lines_keep_a_bounded_token_grid(proc):
    cfg = qwenvl.Config(run="t")
    ds = qwenvl.Lines(cv.load_train().ID[:12], cfg)
    grid = qwenvl.Collate(proc)([ds[i] for i in range(len(ds))])["image_grid_thw"]
    tokens = grid[:, 1] * grid[:, 2] / 4  # 16 px patches, merged 2x2
    assert (tokens <= cfg.max_tokens).all() and (tokens >= 60).all()


@needs_data
def test_fold_all_training_then_predict_only(tmp_path, monkeypatch):
    """The two paths the smoke runs skip: --fold all (no held-out set, adapter saved at
    the end) and --predict-only (re-decode a saved run with other beams)."""
    monkeypatch.setenv("ROAD_RUNS", str(tmp_path / "runs"))
    monkeypatch.setattr(paths, "submissions_dir", lambda: tmp_path)
    cfg, _ = qwenvl.parse(["--run", "r", "--fold", "all", "--smoke", "--model", "tiny",
                           "--workers", "0", "--batch-size", "4", "--keep-weights"])
    qwenvl.train(cfg)            # weights kept: two re-decodes follow
    run = tmp_path / "runs" / "r"
    assert (run / "adapter" / "adapter_config.json").exists() and not (run / "oof.csv").exists()

    assert qwenvl.main(["--run", "r", "--predict-only", "--beams", "2"]) == 0
    sub = pd.read_csv(run / "r.csv", keep_default_na=False)
    assert len(sub) == 2 * cfg.batch_size and (sub.Target.str.len() > 0).all()

    # the recovery on GCP: the trained adapter re-decoded in smaller batches, n-best kept
    assert qwenvl.main(["--run", "r", "--predict-only", "--beams", "2", "--n-best", "2",
                        "--decode-batch-size", "1"]) == 0
    nbest = pd.read_csv(run / "nbest.csv", keep_default_na=False)
    assert len(nbest) == 2 * len(sub) and set(nbest.ID) == set(sub.ID)


def test_decoding_batches_by_decode_batch_size_and_training_by_batch_size():
    cfg = qwenvl.Config(run="x", batch_size=8, decode_batch_size=2, workers=0)
    ds = list(range(16))
    collate = lambda b: b  # noqa: E731
    cpu = torch.device("cpu")
    assert qwenvl.make_loader(ds, collate, cfg, cpu, train=True).batch_size == 8
    assert qwenvl.make_loader(ds, collate, cfg, cpu, train=False).batch_size == 2
    cfg.decode_batch_size = 0                    # unset: decode as it always did
    assert qwenvl.make_loader(ds, collate, cfg, cpu, train=False).batch_size == 8


@needs_data
def test_drop_leaves_out_listed_rows(tmp_path):
    fit, _, _ = qwenvl.splits(qwenvl.Config(run="x", fold="0"))
    listed = list(fit.ID[:3])
    for name, text in (("ids.txt", "\n".join(listed)),
                       ("ids.csv", "ID,why\n" + "\n".join(f"{i},mislabelled" for i in listed))):
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        kept = runs.shape_fit(fit, qwenvl.Config(run="x", fold="0", drop=str(path)))
        assert len(kept) == len(fit) - 3 and not set(kept.ID) & set(listed)
    with pytest.raises(SystemExit, match="does not exist"):
        runs.shape_fit(fit, qwenvl.Config(run="x", drop=str(tmp_path / "nope.txt")))


@needs_data
def test_tall_weight_reweights_without_changing_the_step_count():
    """Duplicating tall rows would lengthen the epoch, confounding the reweighting with
    simply training longer. Sampling keeps the number of optimiser steps identical."""
    cfg = qwenvl.Config(run="x", fold="0", tall_weight=2.0)
    fit, _, _ = qwenvl.splits(cfg)
    tall = [qwenvl.source_height(i) > cfg.tall_px for i in fit.ID]
    weights = qwenvl.tall_weights(fit, cfg)
    assert len(weights) == len(fit)
    assert {w for w, t in zip(weights, tall) if t} == {2.0}
    assert {w for w, t in zip(weights, tall) if not t} == {1.0}
    assert qwenvl.tall_weights(fit, qwenvl.Config(run="x", fold="0")) is None

    drawn = list(torch.utils.data.WeightedRandomSampler(
        weights, num_samples=len(fit), replacement=True,
        generator=torch.Generator().manual_seed(0)))
    assert len(drawn) == len(fit)                      # same steps as an unweighted epoch
    share = sum(tall[i] for i in drawn) / len(drawn)
    assert 0.36 < share < 0.46                         # about 41% of each epoch


@needs_data
def test_pseudo_labels_join_training_and_must_be_test_lines(tmp_path):
    cfg = qwenvl.Config(run="x", fold="0")
    fit, _, test = qwenvl.splits(cfg)
    good = tmp_path / "pseudo.csv"
    pd.DataFrame({"ID": list(test.ID[:5]),
                  "Target": ["the said"] * 4 + [""]}).to_csv(good, index=False)
    shaped = runs.shape_fit(fit, qwenvl.Config(run="x", fold="0", pseudo=str(good)))
    assert len(shaped) == len(fit) + 4          # the empty label is not trained on
    assert shaped.Target.iloc[-1] == "the said"

    leaky = tmp_path / "leaky.csv"
    pd.DataFrame({"ID": list(fit.ID[:2]), "Target": ["x", "y"]}).to_csv(leaky, index=False)
    with pytest.raises(SystemExit, match="repeats 2 training IDs"):
        runs.shape_fit(fit, qwenvl.Config(run="x", fold="0", pseudo=str(leaky)))


def test_jitter_widens_the_photometric_augmentation():
    """12% of the scans are darker than jitter 0.25 ever shows the model, and the worst
    misread lines sit in that tail. The default must not move: every finished run used
    it, and their configs are the comparison."""
    def ranges(aug):
        cj = next(t for t in aug.transforms if type(t).__name__ == "ColorJitter")
        return cj.brightness, cj.contrast

    assert qwenvl.Config(run="x").jitter == 0.25
    assert ranges(qwenvl.augmenter()) == ((0.75, 1.25), (0.75, 1.25))
    assert ranges(qwenvl.augmenter(0.5)) == ((0.5, 1.5), (0.5, 1.5))

    cfg = qwenvl.Config(run="x", jitter=0.5)
    ds = qwenvl.Lines(["a"], cfg, ["t"], train=True)
    assert ranges(ds.aug) == ((0.5, 1.5), (0.5, 1.5))
    assert qwenvl.Lines(["a"], cfg, ["t"], train=False).aug is None   # never at decode


def test_decode_keeps_the_beams_it_is_asked_for(proc, tiny):
    """Beam search ranks every beam and then throws all but one away. Keeping them is
    free, and the ensemble can only choose between candidates it is given."""
    device, dtype = torch.device("cpu"), None
    images = [qwenvl.resize(Image.new("RGB", (320, 40), (240, 235, 220)), 32, 3200)] * 2
    prompts = qwenvl.Collate(proc)([(im, None) for im in images])

    one = qwenvl.decode(tiny, [prompts], proc.tokenizer,
                        qwenvl.generation_config(proc.tokenizer, 6, 4), device, dtype)
    assert [len(c) for c in one] == [1, 1]

    three = qwenvl.decode(tiny, [prompts], proc.tokenizer,
                          qwenvl.generation_config(proc.tokenizer, 6, 4, n_best=3),
                          device, dtype)
    assert [len(c) for c in three] == [3, 3]
    assert [c[0] for c in three] == [c[0] for c in one]   # best beam is unchanged


def test_more_kept_beams_than_beams_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("ROAD_RUNS", str(tmp_path))
    run = tmp_path / "r"
    run.mkdir()
    (run / "config.json").write_text('{"max_new_tokens": 8}', encoding="utf-8")
    cfg = qwenvl.Config(run="r", model="tiny", beams=2, n_best=5)
    with pytest.raises(SystemExit, match="at least as many beams"):
        qwenvl.predict(cfg)


def test_heavy_augmentation_adds_to_the_default_without_changing_it():
    """Every finished run used the default, so it must not move; heavy is opt-in."""
    light, heavy = qwenvl.augmenter(0.5), qwenvl.augmenter(0.5, heavy=True)
    assert len(light.transforms) == 2 and len(heavy.transforms) == 6
    assert qwenvl.Config(run="x").aug_heavy is False

    img = Image.new("RGB", (640, 160), (235, 225, 200))
    ImageDraw.Draw(img).text((20, 60), "W^m Tho: & said", fill=(40, 30, 20))
    torch.manual_seed(0)
    for _ in range(5):                          # every branch can fire; size never changes
        assert heavy(img).size == img.size

    cfg = qwenvl.Config(run="x", aug_heavy=True)
    assert len(qwenvl.Lines(["a"], cfg, ["t"], train=True).aug.transforms) == 6
