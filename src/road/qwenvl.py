"""Qwen3-VL fine-tuning with LoRA. Each line image is scaled to a fixed height and put
in a one-turn chat -- "Transcribe the handwritten line exactly." -- whose answer is the
transcription. Train on the other folds (or all), keep the epoch that
scores best on the held-out fold, then decode that fold (OOF) and the test set.

    python -m road.qwenvl --run qwen4b_f0                        # hold out fold 0
    python -m road.qwenvl --run qwen4b_all --fold all --epochs 4
    python -m road.qwenvl --run qwen4b_f0 --predict-only --beams 4
    python -m road.qwenvl --run preflight --smoke                # whole path, a few lines
    python -m road.qwenvl --run t --smoke --model tiny           # same on a CPU

Artefacts land in paths.runs_dir()/<run> (Drive on Colab): config.json, log.csv,
oof.csv, <run>.csv (the submission) and adapter/ (LoRA weights only).

Choices that matter, each checked against the real processor and config:
  * lines are squashed to a fixed 160 px height, aspect kept. Cropping each line to its
    band of ink and scaling by text size instead (--band) was tried twice and lost both
    times -- 0.8879 on the leaderboard against 0.8973 for the plain squash -- so it is
    off by default, code and tests kept for a third idea;
  * LoRA on every linear layer, vision tower included -- the domain shift is visual;
  * only the answer (transcription + <|im_end|>) is trained on, and logits are computed
    for the answer positions only: the same loss for a fraction of the memory;
  * batches pad on the left, so every row ends with its answer and generation batches;
  * decoding builds its GenerationConfig from scratch: the checkpoint ships
    do_sample=True with temperature 0.7.
"""

from __future__ import annotations

import gc
import json
import math
import shutil
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import peft
import torch
import torch.nn.functional as F
import transformers
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, GenerationConfig

from road import cv, data, metric, paths, runs, submission

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
PROMPT = "Transcribe the handwritten line exactly."
SMOKE_EPOCHS, SMOKE_WORKERS = 2, 2  # smoke also cuts every split to 2 batches of lines
WIDTH_FLOOR_BOOST = 1.6  # most a narrow hand may be enlarged beyond its band scale
MAX_ASPECT = 180         # the processor refuses width/height above 200


@dataclass
class Config:
    run: str
    fold: str = "0"            # held-out fold 0-4, or "all" to train on every line
    model: str = MODEL_ID      # hub id, local path, or "tiny" (random, for tests)
    epochs: int = 3
    batch_size: int = 8
    grad_accum: int = 2
    lr: float = 1e-4
    weight_decay: float = 0.0
    warmup: float = 0.05       # fraction of optimiser steps spent warming up
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    band: bool = False         # band-crop and scale by text size: tried twice, worse
    band_px: int = 48          # the line's band of ink is scaled to this height
    margin: float = 1.2        # kept above and below the band, in band heights:
                               # 0.6 clipped ascenders on 16% of lines and cost 13%
    target_width: int = 3400   # width floor, so narrow hands get the same px per char
    height: int = 160          # fallback height when no ink band stands out
    max_width: int = 4096
    max_tokens: int = 700      # visual tokens per line; one covers 32x32 pixels
    patience: int = 2          # stop after this many epochs without a better score
    beams: int = 3             # final decoding; the per-epoch check is greedy
    n_best: int = 1            # beams kept per line; >1 writes nbest.csv for the ensemble
    decode_batch_size: int = 0  # lines per decoding batch; 0 = batch_size. Beams multiply
                                # it: 8 lines x 8 beams did not fit a 22 GB L4
    precision: str = "auto"    # auto | bf16 | fp16 | fp32
    workers: int = 4
    augment: bool = True
    jitter: float = 0.25       # brightness/contrast range; 0.5 covers every scan
    aug_heavy: bool = False    # elastic, ink thickness and blur on top
    checkpointing: bool = True  # gradient checkpointing: far less memory, ~30% slower
    seed: int = 42
    overwrite: bool = False    # replace an existing run's artefacts instead of refusing
    smoke: bool = False
    keep_weights: bool = False  # --smoke deletes its weights; keep them when a later
                                # command (road.ensemble) has to load the run
    drop: str = ""             # file of training IDs to leave out, one per line or a CSV
    tall_weight: float = 1.0   # draw lines from the harder scan batch this much more often
    tall_px: int = 150         # a source image taller than this is from that batch
    pseudo: str = ""           # submission CSV whose test lines join the training set
    synth: str = ""            # road.stackmix CSV of synthetic training lines


# ---- data ----------------------------------------------------------------------

def resize(img: Image.Image, height: int, max_width: int) -> Image.Image:
    """Scale to `height`, keeping the aspect ratio up to `max_width`."""
    width = max(1, min(max_width, round(img.width * height / img.height)))
    return img.resize((width, height), Image.BICUBIC)


def prepare(img: Image.Image, cfg: Config) -> Image.Image:
    """Crop to the central text band plus a margin and scale so the text is a constant
    size -- measured on the ink, not on the crop, which frames the two scan scales
    differently. Bounded by max_width and a token budget."""
    if not cfg.band:  # the plain squash that scored best; --band to try the crop again
        return resize(img, cfg.height, cfg.max_width)
    w, h = img.size
    band = data.ink_band(img)
    if band is None or not (0.03 * h <= band[1] - band[0] <= 0.6 * h):
        return resize(img, cfg.height, cfg.max_width)

    tall = band[1] - band[0]
    top, bottom = max(0, round(band[0] - cfg.margin * tall)), min(h, round(band[1] + cfg.margin * tall))
    thin = math.ceil(w / MAX_ASPECT) - (bottom - top)   # too wide for its height?
    if thin > 0:
        top, bottom = max(0, top - math.ceil(thin / 2)), min(h, bottom + math.ceil(thin / 2))
    img = img.crop((0, top, w, bottom))
    if img.height * MAX_ASPECT < w:                     # the whole crop is that thin: pad
        edge = tuple(np.asarray(img)[[0, -1]].reshape(-1, 3).mean(0).round().astype(int))
        img = ImageOps.expand(img, (0, 0, 0, math.ceil(w / MAX_ASPECT) - img.height), fill=edge)
    scale = cfg.band_px / tall
    scale = min(max(scale, min(cfg.target_width / w, WIDTH_FLOOR_BOOST * scale)), cfg.max_width / w)
    out = [max(1, round(w * scale)), max(1, round(img.height * scale))]
    for _ in range(8):  # count tokens as the processor does: 16 px patches, merged 2x2,
        tokens = math.ceil(out[0] / 32) * math.ceil(out[1] / 32)  # with sides rounded up
        if tokens <= cfg.max_tokens:
            break
        shrink = (cfg.max_tokens / tokens) ** 0.5
        out = [max(1, int(out[0] * shrink)), max(1, int(out[1] * shrink))]
    return img.resize(tuple(out), Image.BICUBIC)


def _ink_thicker(img: Image.Image) -> Image.Image:
    """Dark ink on light paper: a minimum filter spreads the ink by a pixel."""
    return img.filter(ImageFilter.MinFilter(3))


def _ink_thinner(img: Image.Image) -> Image.Image:
    return img.filter(ImageFilter.MaxFilter(3))


def augmenter(jitter: float = 0.25, heavy: bool = False) -> T.Compose:
    """Mild geometric and photometric noise; the ^ ~ & marks must stay legible.

    `jitter` is the brightness and contrast range. The scans are not uniformly lit:
    mean brightness runs from 111 to 224 with a median of 192, and at 0.25 the
    augmentation only ever spans about 136 to 227 -- so 12% of the training images are
    darker than anything the model is shown. The lines it misreads worst sit in exactly
    that dark tail, on stained paper with the next line bleeding through. 0.5 covers
    every image in the set.

    `heavy` adds the augmentations handwriting recognition normally relies on and this
    pipeline did not use: elastic distortion for the variation between one scribe's
    letterforms and another's, ink thickness for pen and scan weight, and blur for
    focus. Jitter 0.5 won on fold 0 as a regulariser -- normal scans gained more than
    the dark ones it was aimed at -- and three results now say the models overfit 4,098
    lines, so more of this is the direction the evidence points. Kept mild enough that
    a caret or a colon survives it. Module-level functions rather than lambdas, so
    DataLoader workers can pickle them.
    """
    steps = [T.RandomApply([T.RandomAffine(degrees=1.0, translate=(0.01, 0.04),
                                           scale=(0.95, 1.03), shear=(-4, 4),
                                           fill=(255, 255, 255))], p=0.5)]
    if heavy:
        steps += [
            T.RandomApply([T.ElasticTransform(alpha=25.0, sigma=6.0,
                                              fill=(255, 255, 255))], p=0.3),
            T.RandomApply([T.Lambda(_ink_thicker)], p=0.2),
            T.RandomApply([T.Lambda(_ink_thinner)], p=0.2),
        ]
    steps.append(T.ColorJitter(brightness=jitter, contrast=jitter, saturation=0.2))
    if heavy:
        steps.append(T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.3))
    return T.Compose(steps)


class Lines(Dataset):
    """(RGB line image, band-cropped and scaled; transcription or None) per line."""

    def __init__(self, ids, cfg: Config, targets=None, train: bool = False):
        self.ids, self.targets = list(ids), None if targets is None else list(targets)
        self.cfg = cfg
        self.aug = augmenter(cfg.jitter, cfg.aug_heavy) if train and cfg.augment else None

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        with Image.open(paths.image_path(self.ids[i])) as im:
            img = prepare(im.convert("RGB"), self.cfg)
        if self.aug is not None:
            img = self.aug(img)
        return img, None if self.targets is None else self.targets[i]


class Collate:
    """Chat-formats a batch through the processor, padded on the left. With targets,
    adds labels covering only the answer: the transcription and its <|im_end|>."""

    def __init__(self, processor):
        self.proc = processor
        tok = processor.tokenizer
        tok.padding_side = "left"
        user = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        self.prompt = processor.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
        self.eos = tok.eos_token
        self.im_start = tok.convert_tokens_to_ids("<|im_start|>")
        self.header = len(tok("<|im_start|>assistant\n", add_special_tokens=False).input_ids)

    def __call__(self, batch) -> dict[str, torch.Tensor]:
        images, targets = zip(*batch)
        answers = targets[0] is not None
        texts = [self.prompt + t + self.eos for t in targets] if answers else [self.prompt] * len(images)
        enc = dict(self.proc(text=texts, images=list(images), return_tensors="pt", padding=True))
        if answers:
            ids = enc["input_ids"]
            labels = torch.full_like(ids, -100)
            for r in range(len(ids)):
                start = int((ids[r] == self.im_start).nonzero()[-1]) + self.header
                labels[r, start:] = ids[r, start:]
            enc["labels"] = labels
        return enc


def make_loader(ds: Dataset, collate: Collate, cfg: Config, device: torch.device,
                train: bool, weights: list[float] | None = None) -> DataLoader:
    """With weights, draws len(ds) samples per epoch in proportion to them instead of
    shuffling. Drawing rather than duplicating keeps the number of optimiser steps the
    same as an unweighted run, so a weighted run differs from it in one thing only."""
    gen = torch.Generator().manual_seed(cfg.seed)
    sampler = None if weights is None else WeightedRandomSampler(
        weights, num_samples=len(ds), replacement=True, generator=gen)
    size = cfg.batch_size if train else (cfg.decode_batch_size or cfg.batch_size)
    return DataLoader(ds, batch_size=size, shuffle=train and sampler is None,
                      sampler=sampler, num_workers=cfg.workers,
                      collate_fn=collate, pin_memory=device.type == "cuda",
                      persistent_workers=cfg.workers > 0, generator=gen)


# ---- model ---------------------------------------------------------------------

def processor_id(cfg: Config) -> str:
    return MODEL_ID if cfg.model == "tiny" else cfg.model


def tiny_model():
    """A random two-layer Qwen3-VL built from the real config -- same token ids, patch
    and merge sizes and rope layout -- so the real processor drives it. The whole
    pipeline runs on a CPU (--model tiny); its scores mean nothing."""
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    t, v = cfg.text_config, cfg.vision_config
    t.hidden_size, t.intermediate_size, t.num_hidden_layers = 64, 128, 2
    t.num_attention_heads, t.num_key_value_heads, t.head_dim = 4, 2, 16
    for rope in (getattr(t, "rope_parameters", None), getattr(t, "rope_scaling", None)):
        if isinstance(rope, dict):
            rope["mrope_section"] = [4, 2, 2]  # must sum to head_dim / 2
    v.depth, v.hidden_size, v.intermediate_size, v.num_heads = 2, 64, 128, 4
    v.out_hidden_size, v.deepstack_visual_indexes = 64, [0, 1]
    return AutoModelForImageTextToText.from_config(cfg)


def check_peft() -> None:
    """Fail in seconds, not after loading 9 GB of weights: peft 0.20 refuses to build any
    LoRA layer while a torchao older than 0.16 is importable (Colab preinstalls 0.10)."""
    try:
        peft.import_utils.is_torchao_available()
    except ImportError as e:
        raise SystemExit(f"{e}\nNothing here uses torchao: pip uninstall -y torchao") from None


def with_lora(model, cfg: Config):
    """LoRA on every linear layer except the output head -- vision tower included."""
    return get_peft_model(model, LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                                            lora_dropout=cfg.lora_dropout,
                                            target_modules="all-linear"))


def generation_config(tok, max_new_tokens: int, beams: int,
                      n_best: int = 1) -> GenerationConfig:
    return GenerationConfig(do_sample=False, num_beams=beams, max_new_tokens=max_new_tokens,
                            num_return_sequences=n_best,
                            early_stopping=beams > 1, pad_token_id=tok.pad_token_id,
                            eos_token_id=[tok.eos_token_id, tok.pad_token_id])


def answer_loss(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Cross-entropy on the answer tokens. With left padding every row ends with its
    answer, so logits are needed for the last `keep` positions only."""
    labels = batch["labels"]
    keep = int((labels != -100).sum(1).max()) + 1
    logits = model(**{k: v for k, v in batch.items() if k != "labels"}, logits_to_keep=keep).logits
    return F.cross_entropy(logits[:, :-1].flatten(0, 1).float(),
                           labels[:, -(keep - 1):].flatten(), ignore_index=-100)


def train_epoch(model, batches, opt, sched, scaler, device, dtype, cfg: Config,
                log_every: int = 50) -> float:
    """One pass over `batches`; returns the mean training loss."""
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    total, n, bad = 0.0, 0, 0
    opt.zero_grad(set_to_none=True)
    for step, batch in enumerate(batches, 1):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            loss = answer_loss(model, batch)
        scaler.scale(loss / cfg.grad_accum).backward()
        if step % cfg.grad_accum == 0 or step == len(batches):
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
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
    asks for more. Beam search ranks every beam anyway and then discards all but the
    top; keeping them costs nothing and gives an ensemble more to choose between."""
    model.eval()
    n_best = gen.num_return_sequences or 1
    out: list[list[str]] = []
    for batch in batches:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            ids = model.generate(**batch, generation_config=gen)
        new = ids[:, batch["input_ids"].shape[1]:]
        texts = [t.strip() for t in tok.batch_decode(new, skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=False)]
        # generate returns n_best rows per input, grouped and already ranked
        out += [texts[i:i + n_best] for i in range(0, len(texts), n_best)]
    return out


def write_nbest(cands: list[list[str]], ids, path: Path) -> None:
    """Every kept beam, one row per candidate, best first."""
    rows = [(i, rank, text) for i, beams in zip(ids, cands)
            for rank, text in enumerate(beams)]
    pd.DataFrame(rows, columns=["ID", "rank", "Target"]).to_csv(path, index=False)
    print(f"  {len(rows)} candidates over {len(cands)} lines -> {path}", flush=True)


# ---- runs ----------------------------------------------------------------------

def splits(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    """runs.splits for this config; smoke cuts each split to two batches of lines."""
    return runs.splits(cfg.fold, 2 * cfg.batch_size if cfg.smoke else 0)


def source_height(image_id: str) -> int:
    """Height of the original scan, before any resizing. Reads the header only."""
    with Image.open(paths.image_path(image_id)) as im:
        return im.height


def tall_weights(fit: pd.DataFrame, cfg: Config) -> list[float] | None:
    """Sampling weight per training line, or None when every line weighs the same.

    The large-res scan batch is a quarter of the lines and carries over 40% of the
    errors, and matched comparisons say it is harder rather than mishandled, so it wants
    more of the training signal. At weight 2 it is 41% of what the model sees per epoch.
    """
    if cfg.tall_weight == 1.0:
        return None
    tall = [source_height(i) > cfg.tall_px for i in fit.ID]
    share = cfg.tall_weight * sum(tall) / (cfg.tall_weight * sum(tall) + (len(tall) - sum(tall)))
    print(f"weighting {sum(tall)} large-res lines x{cfg.tall_weight:g}: "
          f"{share:.0%} of each epoch, over the same {len(fit)} draws", flush=True)
    return [cfg.tall_weight if t else 1.0 for t in tall]


def train(cfg: Config) -> None:
    device = runs.setup(cfg.seed)
    dtype = runs.amp_dtype(device, cfg.precision)
    run_dir = paths.runs_dir() / cfg.run
    runs.claim(run_dir, cfg.overwrite, cfg.smoke)
    run_dir.mkdir(parents=True, exist_ok=True)
    work = paths.work_dir() / cfg.run / "adapter"
    fit, valid, _ = splits(cfg)
    fit = runs.shape_fit(fit, cfg)

    proc = AutoProcessor.from_pretrained(processor_id(cfg))
    tok = proc.tokenizer
    if cfg.model == "tiny":
        base = tiny_model()
        base.save_pretrained(run_dir / "tiny_base")  # predict() needs the same random base
    else:
        base = AutoModelForImageTextToText.from_pretrained(cfg.model, dtype=dtype or torch.float32)
    model = with_lora(base, cfg).to(device)
    if cfg.checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    targets = [runs.clean_target(t) for t in fit.Target]
    max_new = max(map(len, tok(targets, add_special_tokens=False).input_ids)) + 8
    collate = Collate(proc)
    fit_dl = make_loader(Lines(fit.ID, cfg, targets, train=True), collate, cfg, device,
                         True, tall_weights(fit, cfg))
    val_dl = None if valid is None else make_loader(Lines(valid.ID, cfg), collate, cfg, device, False)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * math.ceil(len(fit_dl) / cfg.grad_accum)
    sched = runs.warmup_cosine(opt, steps, cfg.warmup)
    scaler = torch.amp.GradScaler(device.type, enabled=dtype is torch.float16)
    greedy = generation_config(tok, max_new, beams=1)

    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    meta = {**asdict(cfg), "device": gpu, "amp": str(dtype), "max_new_tokens": max_new,
            "n_fit": len(fit), "n_valid": 0 if valid is None else len(valid), "steps": steps,
            "trainable_params": trainable, "torch": torch.__version__,
            "transformers": transformers.__version__, "peft": peft.__version__}
    (run_dir / "config.json").write_text(json.dumps(meta, indent=2))
    print(f"{cfg.run}: fit {len(fit)} lines, held out {meta['n_valid']}, {gpu}, amp {dtype}, "
          f"{cfg.epochs} epochs x {len(fit_dl)} batches, {trainable / 1e6:.1f}M trainable",
          flush=True)

    best, stale, log = -math.inf, 0, []
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        row = {"epoch": epoch,
               "loss": train_epoch(model, fit_dl, opt, sched, scaler, device, dtype, cfg)}
        msg = f"  epoch {epoch:2d}  loss {row['loss']:.4f}"
        if val_dl is not None:
            preds = [c[0] for c in decode(model, val_dl, tok, greedy, device, dtype)]
            row |= metric.aggregate(metric.line_errors(valid.Target, preds))
            if row["public"] > best:  # the leaderboard's own score, higher is better
                best, stale = row["public"], 0
                model.save_pretrained(work)
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
        model.save_pretrained(work)
    final = run_dir / "adapter"
    if work.resolve() != final.resolve():
        shutil.copytree(work, final, dirs_exist_ok=True)
    print(("trained on every line" if val_dl is None else f"best held-out public ~{best:.4f}")
          + f" -> {final}", flush=True)


def predict(cfg: Config) -> None:
    """Decode the held-out fold (OOF, scored) and the test set (submission) with the
    saved adapter merged into the base model."""
    if cfg.n_best > cfg.beams:
        raise SystemExit(f"--n-best {cfg.n_best} needs at least as many beams "
                         f"(--beams is {cfg.beams})")
    device = runs.setup(cfg.seed)
    dtype = runs.amp_dtype(device, cfg.precision)
    run_dir = paths.runs_dir() / cfg.run
    meta = json.loads((run_dir / "config.json").read_text())
    proc = AutoProcessor.from_pretrained(processor_id(cfg))
    tok = proc.tokenizer
    base_src = run_dir / "tiny_base" if cfg.model == "tiny" else cfg.model
    base = AutoModelForImageTextToText.from_pretrained(base_src, dtype=dtype or torch.float32)
    model = PeftModel.from_pretrained(base, run_dir / "adapter").merge_and_unload().to(device)
    gen = generation_config(tok, meta["max_new_tokens"], cfg.beams, cfg.n_best)
    collate = Collate(proc)
    _, valid, test = splits(cfg)

    if valid is not None:
        t0 = time.time()
        cands = decode(model, make_loader(Lines(valid.ID, cfg), collate, cfg, device, False),
                       tok, gen, device, dtype)
        preds = [c[0] for c in cands]
        if cfg.n_best > 1:
            write_nbest(cands, valid.ID, run_dir / "oof_nbest.csv")
        oof = pd.DataFrame({"ID": valid.ID, "Target": preds, "fold": int(cfg.fold)})
        oof.to_csv(run_dir / "oof.csv", index=False)
        train_df = cv.load_train()
        folds = cv.make_folds(train_df.ID)
        print(f"\nheld-out fold {cfg.fold}, {cfg.beams} beams ({time.time() - t0:.0f}s):\n", flush=True)
        cv.report(cv.oof_errors(oof, train_df, folds), folds)

    t0 = time.time()
    cands = decode(model, make_loader(Lines(test.ID, cfg), collate, cfg, device, False),
                   tok, gen, device, dtype)
    preds = [c[0] for c in cands]
    if cfg.n_best > 1:
        write_nbest(cands, test.ID, run_dir / "nbest.csv")
    path = submission.write(test.ID, preds, cfg.run, test_ids=test.ID if cfg.smoke else None)
    shutil.copy2(path, run_dir / path.name)
    print(f"\ntest: {len(preds)} lines decoded ({time.time() - t0:.0f}s) -> {run_dir / path.name}",
          flush=True)


def parse(argv: list[str] | None = None) -> tuple[Config, bool]:
    return runs.parse(Config, argv, __doc__, SMOKE_EPOCHS, SMOKE_WORKERS)


def main(argv: list[str] | None = None) -> int:
    cfg, predict_only = parse(argv)
    check_peft()
    if not predict_only:
        train(cfg)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    predict(cfg)
    if cfg.smoke:
        if not cfg.keep_weights:
            # Smoke weights are useless; don't leave them on Drive.
            for d in {paths.work_dir() / cfg.run / "adapter",
                      paths.runs_dir() / cfg.run / "adapter",
                      paths.runs_dir() / cfg.run / "tiny_base"}:
                shutil.rmtree(d, ignore_errors=True)
        print("\nSMOKE OK -- train, save, reload, decode and submission all ran.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
