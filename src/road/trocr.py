"""TrOCR fine-tuning: train on the other folds (or all), keep the epoch that scores
best on the held-out fold, then beam-decode that fold (OOF) and the test set into a
submission.

    python -m road.trocr --run trocr_f0                        # hold out fold 0
    python -m road.trocr --run trocr_all --fold all --epochs 15
    python -m road.trocr --run trocr_f0 --predict-only --beams 8
    python -m road.trocr --run preflight --smoke               # whole path, a few lines
    python -m road.trocr --run t --smoke --model tiny          # same on a CPU, in seconds

Artefacts land in paths.runs_dir()/<run> -- Drive on Colab, so they survive a
reset: config.json, log.csv, oof.csv, <run>.csv (the submission) and model/.
During training the best weights stay on local disk and are copied once at the end.

Choices that matter:
  * images are squashed to 384x384, the input TrOCR was pretrained on;
  * training text drops outer whitespace and collapses inner runs; scoring always
    uses the raw targets;
  * labels are `text </s>`, no <s>: after its start token the pretrained decoder goes
    straight to text (pinned by a test on the real checkpoint);
  * decoding uses a GenerationConfig built from scratch, so the checkpoint's shipped
    defaults don't leak in: use_cache off (slow decoding) and a 20-token default
    length (cuts longer lines short). Detokenising keeps spaces exactly (no " ." ->
    "." clean-up).
"""

from __future__ import annotations

import gc
import json
import math
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import DataLoader, Dataset
from transformers import (GenerationConfig, RobertaTokenizer, TrOCRConfig, ViTConfig,
                          VisionEncoderDecoderConfig, VisionEncoderDecoderModel)

from road import cv, data, metric, paths, qwenvl, runs, submission

MODEL_ID = "microsoft/trocr-base-handwritten"
SMOKE_EPOCHS, SMOKE_WORKERS = 2, 2  # smoke also cuts every split to 2 batches of lines


@dataclass
class Config:
    run: str
    fold: str = "0"            # held-out fold 0-4, or "all" to train on every line
    model: str = MODEL_ID      # hub id, local path, or "tiny" (random, for tests)
    epochs: int = 20
    batch_size: int = 16
    grad_accum: int = 1
    lr: float = 4e-5
    weight_decay: float = 0.01
    warmup: float = 0.1        # fraction of optimiser steps spent warming up
    label_smoothing: float = 0.0
    patience: int = 5          # stop after this many epochs without a better score
    beams: int = 4             # final decoding; the per-epoch check is greedy
    n_best: int = 1            # beams kept per line; >1 writes nbest.csv for the ensemble
    precision: str = "auto"    # auto | bf16 | fp16 | fp32
    workers: int = 4
    augment: bool = True
    seed: int = 42
    overwrite: bool = False    # replace an existing run's artefacts instead of refusing
    smoke: bool = False
    drop: str = ""             # file of training IDs to leave out, one per line or a CSV
    pseudo: str = ""           # submission CSV whose test lines join the training set
    synth: str = ""            # road.stackmix CSV of synthetic training lines


# ---- text <-> tokens ----------------------------------------------------------

def encode(tok, texts) -> list[list[int]]:
    """Token ids `text </s>`. No <s>: after its start token (</s>) the pretrained
    decoder goes straight to text."""
    return [ids + [tok.eos_token_id] for ids in tok(list(texts), add_special_tokens=False).input_ids]


def decode_ids(tok, ids) -> list[str]:
    texts = tok.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return [t.strip() for t in texts]


def shift_right(labels: torch.Tensor, start_id: int, pad_id: int) -> torch.Tensor:
    """Teacher-forcing decoder inputs: the start token, then labels[:-1]."""
    dec = labels.new_full(labels.shape, pad_id)
    dec[:, 1:] = labels[:, :-1]
    dec[:, 0] = start_id
    return dec.masked_fill(dec == -100, pad_id)


# ---- data ----------------------------------------------------------------------

class Lines(Dataset):
    """(3 x H x W image, token ids or None) per line."""

    def __init__(self, ids, size: tuple[int, int], train: bool = False, labels=None):
        self.ids, self.labels = list(ids), labels
        self.tf = data.line_transform(size, train=train)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        x = self.tf(data.load_line(self.ids[i])).repeat(3, 1, 1)
        return x, None if self.labels is None else torch.tensor(self.labels[i])


def collate(batch):
    xs = torch.stack([x for x, _ in batch])
    if batch[0][1] is None:
        return xs, None
    return xs, torch.nn.utils.rnn.pad_sequence([y for _, y in batch], batch_first=True,
                                               padding_value=-100)


def make_loader(ds: Dataset, cfg: Config, device: torch.device, train: bool) -> DataLoader:
    return DataLoader(ds, batch_size=cfg.batch_size * (1 if train else 2), shuffle=train,
                      num_workers=cfg.workers, collate_fn=collate,
                      pin_memory=device.type == "cuda", persistent_workers=cfg.workers > 0,
                      generator=torch.Generator().manual_seed(cfg.seed))


# ---- model ---------------------------------------------------------------------

def load_tokenizer(name_or_path) -> RobertaTokenizer:
    """TrOCR's tokenizer, by its own class. transformers 5's AutoTokenizer routes the
    vision-encoder-decoder model type to a generic backend that cannot build this
    checkpoint's tokenizer from vocab.json + merges.txt (it fails on Colab too);
    RobertaTokenizer, the class its tokenizer_config names, can."""
    return RobertaTokenizer.from_pretrained(name_or_path)


def tiny_model(tok) -> VisionEncoderDecoderModel:
    """A random one-layer TrOCR on 32x32 input: the whole pipeline runs on a CPU in
    seconds (--model tiny). Its scores mean nothing."""
    enc = ViTConfig(image_size=32, patch_size=16, hidden_size=32, num_hidden_layers=1,
                    num_attention_heads=2, intermediate_size=64)
    dec = TrOCRConfig(vocab_size=len(tok), d_model=32, decoder_layers=1,
                      decoder_attention_heads=2, decoder_ffn_dim=64,
                      max_position_embeddings=256, pad_token_id=tok.pad_token_id,
                      bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
                      decoder_start_token_id=tok.eos_token_id)
    model = VisionEncoderDecoderModel(
        config=VisionEncoderDecoderConfig.from_encoder_decoder_configs(enc, dec))
    model.generation_config.decoder_start_token_id = tok.eos_token_id
    return model


def start_token(model) -> int:
    start = model.generation_config.decoder_start_token_id
    if start is None:
        start = model.config.decoder_start_token_id
    if start is None:
        raise ValueError("model has no decoder_start_token_id")
    return int(start)


def generation_config(tok, start_id: int, max_new_tokens: int, beams: int,
                      n_best: int = 1) -> GenerationConfig:
    return GenerationConfig(decoder_start_token_id=start_id, bos_token_id=tok.bos_token_id,
                            eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
                            max_new_tokens=max_new_tokens, num_beams=beams,
                            num_return_sequences=n_best,
                            early_stopping=beams > 1, do_sample=False)


def train_epoch(model, batches, opt, sched, scaler, device, dtype, cfg: Config,
                start_id: int, pad_id: int, log_every: int = 50) -> float:
    """One pass over `batches`; returns the mean training loss."""
    model.train()
    total, n, bad = 0.0, 0, 0
    opt.zero_grad(set_to_none=True)
    for step, (pv, y) in enumerate(batches, 1):
        pv, y = pv.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            # use_cache=False: teacher forcing needs no KV cache; building one wastes memory.
            logits = model(pixel_values=pv, decoder_input_ids=shift_right(y, start_id, pad_id),
                           use_cache=False).logits
        loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten(),
                               ignore_index=-100, label_smoothing=cfg.label_smoothing)
        scaler.scale(loss / cfg.grad_accum).backward()
        if step % cfg.grad_accum == 0 or step == len(batches):
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
        if torch.isfinite(loss):
            total, n, bad = total + loss.item(), n + 1, 0
        else:
            # fp16 overflow: GradScaler skips that step. Persistent means divergence.
            bad += 1
            if bad >= 10:
                raise FloatingPointError(f"10 consecutive non-finite losses (step {step})")
        if log_every and step % log_every == 0:
            print(f"    step {step:4d}/{len(batches)}  loss {total / max(n, 1):.4f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}", flush=True)
    return total / max(n, 1)


@torch.inference_mode()
def decode(model, batches, tok, gen: GenerationConfig, device, dtype) -> list[list[str]]:
    """Per line, the kept beams best first -- one string unless the generation config
    asks for more. Beam search ranks them all and then throws all but the top away."""
    model.eval()
    n_best = gen.num_return_sequences or 1
    out: list[list[str]] = []
    for pv, _ in batches:
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            ids = model.generate(pixel_values=pv.to(device, non_blocking=True),
                                 generation_config=gen)
        texts = decode_ids(tok, ids)
        out += [texts[i:i + n_best] for i in range(0, len(texts), n_best)]
    return out


# ---- runs ----------------------------------------------------------------------

def splits(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    """runs.splits for this config; smoke cuts each split to two batches of lines."""
    return runs.splits(cfg.fold, 2 * cfg.batch_size if cfg.smoke else 0)


def _save(model, tok, where: Path) -> None:
    where.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(where)
    tok.save_pretrained(where)


def train(cfg: Config) -> None:
    device = runs.setup(cfg.seed)
    dtype = runs.amp_dtype(device, cfg.precision)
    run_dir = paths.runs_dir() / cfg.run
    runs.claim(run_dir, cfg.overwrite, cfg.smoke)
    run_dir.mkdir(parents=True, exist_ok=True)
    work = paths.work_dir() / cfg.run / "model"
    fit, valid, _ = splits(cfg)
    fit = runs.shape_fit(fit, cfg)

    tok = load_tokenizer(MODEL_ID if cfg.model == "tiny" else cfg.model)
    model = tiny_model(tok) if cfg.model == "tiny" else VisionEncoderDecoderModel.from_pretrained(cfg.model)
    model.to(device)
    start_id = start_token(model)
    size = (model.config.encoder.image_size,) * 2
    labels = encode(tok, fit.Target.map(runs.clean_target))
    max_new = max(map(len, labels)) + 8

    fit_dl = make_loader(Lines(fit.ID, size, train=cfg.augment, labels=labels), cfg, device, True)
    val_dl = None if valid is None else make_loader(Lines(valid.ID, size), cfg, device, False)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * math.ceil(len(fit_dl) / cfg.grad_accum)
    sched = runs.warmup_cosine(opt, steps, cfg.warmup)
    scaler = torch.amp.GradScaler(device.type, enabled=dtype is torch.float16)
    greedy = generation_config(tok, start_id, max_new, beams=1)

    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    meta = {**asdict(cfg), "device": gpu, "amp": str(dtype), "start_id": start_id,
            "max_new_tokens": max_new, "image_size": size[0], "n_fit": len(fit),
            "n_valid": 0 if valid is None else len(valid), "steps": steps,
            "torch": torch.__version__, "transformers": transformers.__version__}
    (run_dir / "config.json").write_text(json.dumps(meta, indent=2))
    print(f"{cfg.run}: fit {len(fit)} lines, held out {meta['n_valid']}, {gpu}, "
          f"amp {dtype}, {cfg.epochs} epochs x {len(fit_dl)} batches", flush=True)

    best, stale, log = -math.inf, 0, []
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        row = {"epoch": epoch, "loss": train_epoch(model, fit_dl, opt, sched, scaler, device,
                                                   dtype, cfg, start_id, tok.pad_token_id)}
        msg = f"  epoch {epoch:2d}  loss {row['loss']:.4f}"
        if val_dl is not None:
            preds = [c[0] for c in decode(model, val_dl, tok, greedy, device, dtype)]
            row |= metric.aggregate(metric.line_errors(valid.Target, preds))
            if row["public"] > best:  # the leaderboard's own score, higher is better
                best, stale = row["public"], 0
                _save(model, tok, work)
            else:
                stale += 1
            msg += (f"  held-out {row['score']:.4f}, public ~{row['public']:.4f} "
                    f"(WER {row['wer']:.4f}, CER {row['cer']:.4f})"
                    + ("  *best" if stale == 0 else ""))
        row["secs"] = round(time.time() - t0, 1)
        log.append(row)
        pd.DataFrame(log).to_csv(run_dir / "log.csv", index=False)
        if device.type == "cuda":
            msg += f"  peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB"
        print(f"{msg}  {row['secs']:.0f}s", flush=True)
        if val_dl is not None and stale >= cfg.patience:
            print(f"  no improvement in {cfg.patience} epochs -- stopping", flush=True)
            break

    if val_dl is None:
        _save(model, tok, work)
    final = run_dir / "model"
    if work.resolve() != final.resolve():
        shutil.copytree(work, final, dirs_exist_ok=True)
    print(("trained on every line" if val_dl is None else f"best held-out public ~{best:.4f}")
          + f" -> {final}", flush=True)


def predict(cfg: Config) -> None:
    """Beam-decode the held-out fold (OOF, scored) and the test set (submission)."""
    if cfg.n_best > cfg.beams:
        raise SystemExit(f"--n-best {cfg.n_best} needs at least as many beams "
                         f"(--beams is {cfg.beams})")
    device = runs.setup(cfg.seed)
    dtype = runs.amp_dtype(device, cfg.precision)
    run_dir = paths.runs_dir() / cfg.run
    meta = json.loads((run_dir / "config.json").read_text())
    tok = load_tokenizer(run_dir / "model")
    model = VisionEncoderDecoderModel.from_pretrained(run_dir / "model").to(device)
    gen = generation_config(tok, meta["start_id"], meta["max_new_tokens"], cfg.beams,
                            cfg.n_best)
    size = (meta["image_size"],) * 2
    _, valid, test = splits(cfg)

    if valid is not None:
        t0 = time.time()
        cands = decode(model, make_loader(Lines(valid.ID, size), cfg, device, False),
                       tok, gen, device, dtype)
        preds = [c[0] for c in cands]
        if cfg.n_best > 1:
            qwenvl.write_nbest(cands, valid.ID, run_dir / "oof_nbest.csv")
        oof = pd.DataFrame({"ID": valid.ID, "Target": preds, "fold": int(cfg.fold)})
        oof.to_csv(run_dir / "oof.csv", index=False)
        train_df = cv.load_train()
        folds = cv.make_folds(train_df.ID)
        print(f"\nheld-out fold {cfg.fold}, {cfg.beams} beams ({time.time() - t0:.0f}s):\n", flush=True)
        cv.report(cv.oof_errors(oof, train_df, folds), folds)

    t0 = time.time()
    cands = decode(model, make_loader(Lines(test.ID, size), cfg, device, False),
                   tok, gen, device, dtype)
    preds = [c[0] for c in cands]
    if cfg.n_best > 1:
        qwenvl.write_nbest(cands, test.ID, run_dir / "nbest.csv")
    path = submission.write(test.ID, preds, cfg.run, test_ids=test.ID if cfg.smoke else None)
    shutil.copy2(path, run_dir / path.name)
    print(f"\ntest: {len(preds)} lines decoded ({time.time() - t0:.0f}s) -> {run_dir / path.name}",
          flush=True)


def parse(argv: list[str] | None = None) -> tuple[Config, bool]:
    return runs.parse(Config, argv, __doc__, SMOKE_EPOCHS, SMOKE_WORKERS)


def main(argv: list[str] | None = None) -> int:
    cfg, predict_only = parse(argv)
    if not predict_only:
        train(cfg)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    predict(cfg)
    if cfg.smoke:
        # Smoke weights are useless; don't leave 1.3 GB of them on Drive.
        for d in {paths.work_dir() / cfg.run / "model", paths.runs_dir() / cfg.run / "model"}:
            shutil.rmtree(d, ignore_errors=True)
        print("\nSMOKE OK -- train, save, reload, decode and submission all ran.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
