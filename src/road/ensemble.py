"""Pick, per line, the transcription that several fine-tuned models jointly prefer.

Each run contributes its decoded predictions as candidates; every candidate is then
scored by every model -- the mean log-probability of the answer under teacher forcing --
and the candidate with the best total wins. Scoring every candidate under every model is
what makes the comparison fair: raw confidences from differently calibrated models are
not comparable, but "how likely does each model think this exact text is" is. The score
is per token, so a long candidate is not penalised for its length.

    python -m road.ensemble --runs qwen4b_all qwen8b_all --out ens_4b8b
    python -m road.ensemble --runs a b --out e --candidates-from trocr_f0
    python -m road.ensemble --runs a b trocr_all --out e   # a TrOCR run can judge too
    python -m road.ensemble --runs a b --out e --oof      # combine held-out folds instead
    python -m road.ensemble --runs a b --out e2 --reuse e  # score only what e did not

A run can supply candidates without judging them: --candidates-from adds its text to the
pool without loading its weights.

Scoring is the whole cost -- one forward pass per candidate per judge -- and most of a new
ensemble's pool was usually scored by an earlier one. --reuse reads those ensembles'
scores.csv and only scores the (line, text, judge) triples none of them holds, so adding
one run to a pool costs that run's new candidates, not the pool. It is not bit-exact: in
bf16 on a GPU a judge's score moves with how its batch was padded, so two sessions that
score the same pair agree to a median of ~1e-3, not exactly. Rescoring has that noise
too -- taking one judge's scores from another session flipped 11 of 1,374 picks, all near
ties -- so reusing costs nothing a fresh run would not. A judge retrained under the same
name is the one real hazard: its old scores belong to other weights.

Judging is not limited to one architecture. A run is loaded by what it saved -- an
adapter/ is a Qwen LoRA run, a model/ is a TrOCR run -- and both report the same
quantity, the mean log-probability per answer token, so the scores add up. They tokenise
differently though, and a mean per token depends on how finely the text is split, so
main() prints each judge's scale: if one sits far from the others, a plain sum is letting
it dominate and the comparison is no longer fair.

Why that matters here: the two Qwen judges correlate at 0.883 and sit 2.1 characters
apart, so selection is two opinions from one family. TrOCR sits 4.8 characters away and
is solo on two thirds of lines. A weaker but independent judge may choose better than a
third similar one -- or may drag the sum. Measure it with --oof before submitting.

Each run's own config.json supplies its line preparation, so runs that prepared their
images differently can still be combined. Writes the submission through road.submission,
so it is validated, plus a scores.csv of every candidate under every model -- the
evidence for whether the ensemble is doing anything at all.
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd
import torch
from peft import PeftModel
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForImageTextToText, AutoProcessor,
                          VisionEncoderDecoderModel)

from road import cv, paths, qwenvl, runs, submission, trocr


class QwenScoring(Dataset):
    """(prepared image, candidate text) pairs, one row per candidate."""

    def __init__(self, pairs: list[tuple[str, str]], cfg: qwenvl.Config):
        self.pairs, self.cfg = pairs, cfg

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        image_id, text = self.pairs[i]
        with Image.open(paths.image_path(image_id)) as im:
            return qwenvl.prepare(im.convert("RGB"), self.cfg), text


@torch.inference_mode()
def answer_logprob(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Mean log-probability per answer token, one number per row."""
    labels = batch["labels"]
    keep = int((labels != -100).sum(1).max()) + 1
    logits = model(**{k: v for k, v in batch.items() if k != "labels"}, logits_to_keep=keep).logits
    logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    target = labels[:, -(keep - 1):]
    kept = target != -100
    token = logp.gather(-1, target.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return (token * kept).sum(1) / kept.sum(1).clamp_min(1)


@torch.inference_mode()
def decoder_logprob(model, pixel_values, labels, start_id: int, pad_id: int):
    """The same quantity for an encoder-decoder run: mean log-probability per label
    token. Simpler than the decoder-only case -- the decoder is given only the answer,
    so its positions line up with the labels and there is no prompt to trim."""
    logits = model(pixel_values=pixel_values,
                   decoder_input_ids=trocr.shift_right(labels, start_id, pad_id),
                   use_cache=False).logits
    logp = torch.log_softmax(logits.float(), dim=-1)
    keep = labels != -100
    token = logp.gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return (token * keep).sum(1) / keep.sum(1).clamp_min(1)


class QwenJudge:
    """A Qwen LoRA run, its adapter merged back into the base model."""

    def __init__(self, run: str):
        run_dir = paths.runs_dir() / run
        self.cfg = runs.saved_config(qwenvl.Config, run_dir)
        self.device = runs.setup(self.cfg.seed)
        self.dtype = runs.amp_dtype(self.device, self.cfg.precision)
        self.proc = AutoProcessor.from_pretrained(qwenvl.processor_id(self.cfg))
        base_src = run_dir / "tiny_base" if self.cfg.model == "tiny" else self.cfg.model
        base = AutoModelForImageTextToText.from_pretrained(
            base_src, dtype=self.dtype or torch.float32)
        self.model = (PeftModel.from_pretrained(base, run_dir / "adapter")
                      .merge_and_unload().to(self.device).eval())

    def score(self, pairs: list[tuple[str, str]], batch_size: int) -> list[float]:
        loader = DataLoader(QwenScoring(pairs, self.cfg), batch_size=batch_size,
                            shuffle=False, num_workers=self.cfg.workers,
                            collate_fn=qwenvl.Collate(self.proc),
                            pin_memory=self.device.type == "cuda")
        out: list[float] = []
        for batch in loader:
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(self.device.type, dtype=self.dtype,
                                enabled=self.dtype is not None):
                out += answer_logprob(self.model, batch).float().tolist()
        return out


class TrocrJudge:
    """A TrOCR run scoring text it did not propose. Its own config.json carries the
    image size and start token it was trained with, so what it is shown here matches
    what it was trained on."""

    def __init__(self, run: str):
        run_dir = paths.runs_dir() / run
        self.cfg = runs.saved_config(trocr.Config, run_dir)
        meta = json.loads((run_dir / "config.json").read_text())
        self.device = runs.setup(self.cfg.seed)
        self.dtype = runs.amp_dtype(self.device, self.cfg.precision)
        self.tok = trocr.load_tokenizer(run_dir / "model")
        self.model = VisionEncoderDecoderModel.from_pretrained(
            run_dir / "model").to(self.device).eval()
        self.size = (meta["image_size"],) * 2
        self.start_id = meta["start_id"]

    def score(self, pairs: list[tuple[str, str]], batch_size: int) -> list[float]:
        labels = trocr.encode(self.tok, [t for _, t in pairs])
        ds = trocr.Lines([i for i, _ in pairs], self.size, train=False, labels=labels)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=self.cfg.workers, collate_fn=trocr.collate,
                            pin_memory=self.device.type == "cuda")
        out: list[float] = []
        for pixel_values, batch_labels in loader:
            pixel_values = pixel_values.to(self.device, non_blocking=True)
            batch_labels = batch_labels.to(self.device, non_blocking=True)
            with torch.autocast(self.device.type, dtype=self.dtype,
                                enabled=self.dtype is not None):
                out += decoder_logprob(self.model, pixel_values, batch_labels,
                                       self.start_id,
                                       self.tok.pad_token_id).float().tolist()
        return out


def load_judge(run: str):
    """The right judge for a finished run, chosen by what the run saved."""
    run_dir = paths.runs_dir() / run
    if (run_dir / "adapter").is_dir():
        return QwenJudge(run)
    if (run_dir / "model").is_dir():
        return TrocrJudge(run)
    raise SystemExit(
        f"{run_dir} holds neither an adapter/ (Qwen) nor a model/ (TrOCR), so it cannot "
        "judge. Pass it to --candidates-from to use its text without its weights.")


def score(run: str, pairs: list[tuple[str, str]], batch_size: int = 16) -> list[float]:
    """Mean answer log-probability of every (line, candidate) pair under this run."""
    judge = load_judge(run)
    out = judge.score(pairs, batch_size)
    del judge
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def known_scores(ensembles: list[str]) -> pd.DataFrame:
    """Every (ID, Target) the given ensembles scored, one column per judge they used."""
    known = None
    for name in ensembles:
        path = paths.runs_dir() / name / "scores.csv"
        if not path.exists():
            raise SystemExit(f"{path} is missing -- nothing to reuse from {name}")
        frame = pd.read_csv(path, keep_default_na=False,
                            dtype={"ID": str, "Target": str}).set_index(["ID", "Target"])
        frame = frame[[c for c in frame.columns if c not in ("candidates", "margin")]]
        known = frame if known is None else known.combine_first(frame)
    return known if known is not None else pd.DataFrame()


def score_reusing(run: str, pairs: list[tuple[str, str]], known: pd.DataFrame,
                  batch_size: int = 16) -> list[float]:
    """score(), taking whatever `known` already holds for this judge."""
    have: dict[tuple[str, str], float] = {}
    if run in known.columns:
        col = known[run].dropna()
        have = dict(zip(col.index, col.to_numpy(dtype=float)))
    todo = [p for p in pairs if p not in have]
    if todo:
        have.update(zip(todo, score(run, todo, batch_size)))
    print(f"  {run}: {len(pairs) - len(todo)} of {len(pairs)} scores reused, "
          f"{len(todo)} computed", flush=True)
    return [have[p] for p in pairs]


def candidates(run_names: list[str], oof: bool = False,
               beams: int | None = None) -> dict[str, list[str]]:
    """Each run's predictions, de-duplicated per line, in run order.

    `beams` caps how many of each run's kept beams enter the pool, best first. Scoring
    costs one forward pass per candidate per judge, so this is the dial to turn when a
    run would otherwise take hours: the beams are rank-ordered, so a lower cap drops the
    least likely ones first.
    """
    per_line: dict[str, list[str]] = {}
    for run in run_names:
        run_dir = paths.runs_dir() / run
        # nbest.csv holds every beam this run kept, its own prediction first; when a run
        # has one, use it -- more candidates is the whole point of the pool.
        nbest = run_dir / ("oof_nbest.csv" if oof else "nbest.csv")
        path = nbest if nbest.exists() else run_dir / ("oof.csv" if oof else f"{run}.csv")
        if not path.exists():
            raise SystemExit(f"{path} is missing -- has {run} been decoded?")
        preds = pd.read_csv(path, keep_default_na=False, dtype=str)
        if beams is not None and "rank" in preds.columns:
            preds = preds[preds["rank"].astype(int) < beams]
        for image_id, text in zip(preds.ID, preds.Target):
            seen = per_line.setdefault(image_id, [])
            if text not in seen:
                seen.append(text)
    return per_line


def choose(scores: pd.DataFrame, run_names: list[str]) -> pd.DataFrame:
    """Per line, the candidate with the best total score, and the margin over the worst."""
    totals = scores.assign(total=scores[run_names].sum(axis=1))
    best = totals.loc[totals.groupby("ID", sort=False).total.idxmax()]
    spread = totals.groupby("ID", sort=False).total.agg(["max", "min", "count"])
    return pd.DataFrame({
        "ID": best.ID.to_numpy(),
        "Target": best.Target.to_numpy(),
        "candidates": spread["count"].loc[best.ID].to_numpy(),
        "margin": (spread["max"] - spread["min"]).loc[best.ID].to_numpy(),
    })


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="finished run names to combine")
    ap.add_argument("--out", required=True, help="name for the combined submission")
    ap.add_argument("--candidates-from", nargs="+", default=[], metavar="RUN",
                    help="runs that contribute candidate text but do not score it")
    ap.add_argument("--oof", action="store_true", help="combine held-out predictions instead")
    ap.add_argument("--partial", action="store_true",
                    help="the runs cover only part of the test set, as a smoke run does; "
                         "check only the lines they cover")
    ap.add_argument("--beams", type=int, default=None, metavar="K",
                    help="use only each run's best K kept beams; scoring costs one "
                         "forward pass per candidate per judge, so this bounds the run")
    ap.add_argument("--reuse", nargs="+", default=[], metavar="ENS",
                    help="earlier ensembles whose scores.csv to reuse: a candidate a "
                         "judge already scored there is not scored again")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args(argv)

    pool = args.runs + args.candidates_from
    per_line = candidates(pool, args.oof, args.beams)
    pairs = [(image_id, text) for image_id, texts in per_line.items() for text in texts]
    agreed = sum(1 for texts in per_line.values() if len(texts) == 1)
    extra = f", {len(args.candidates_from)} of them candidates only" if args.candidates_from else ""
    print(f"{len(per_line)} lines, {len(pairs)} candidates from {len(pool)} runs{extra}; "
          f"every run agrees on {agreed} ({agreed / max(len(per_line), 1):.0%})", flush=True)

    scores = pd.DataFrame({"ID": [p[0] for p in pairs], "Target": [p[1] for p in pairs]})
    known = known_scores(args.reuse)
    for run in args.runs:   # only the judges score; the rest merely proposed text
        scores[run] = score_reusing(run, pairs, known, args.batch_size)
        col = scores[run]
        print(f"  scored under {run}  (mean {col.mean():+.3f}, sd {col.std():.3f})",
              flush=True)
    picked = choose(scores, args.runs)

    out_dir = paths.runs_dir() / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out_dir / "scores.csv", index=False)
    for run in pool:  # how often each run's own candidate survived
        own = pd.read_csv(paths.runs_dir() / run / ("oof.csv" if args.oof else f"{run}.csv"),
                          keep_default_na=False, dtype=str).set_index("ID").Target
        kept = picked.set_index("ID").Target.eq(own.loc[picked.ID].to_numpy()).mean()
        print(f"  {run}: its own text kept on {kept:.0%} of lines")

    if args.oof:
        train = cv.load_train()
        folds = cv.make_folds(train.ID)
        oof = picked[["ID", "Target"]].assign(fold=picked.ID.map(folds).to_numpy())
        oof.to_csv(out_dir / "oof.csv", index=False)
        print()
        cv.report(cv.oof_errors(oof, train, folds), folds)
    else:
        path = submission.write(picked.ID, picked.Target, args.out,
                                test_ids=picked.ID if args.partial else None)
        (out_dir / path.name).write_bytes(path.read_bytes())
        print(f"\n{len(picked)} lines -> {out_dir / path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
