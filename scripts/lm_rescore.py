"""Add a character language model of the transcriptions to the ensemble's choice.

    # measure: fold 0, where the truth is known -- sweeps the LM weight
    ROAD_RUNS=path/to/runs python scripts/lm_rescore.py oof_2judge --measure
    # apply: re-pick a test ensemble with a chosen weight, no GPU
    ROAD_RUNS=path/to/runs python scripts/lm_rescore.py ens_nbest --weight 0.3

Why: the judges are the models whose biases the ensemble is trying to correct. Two thirds
of our word errors are a different word, many of them the model modernising period
spelling (prsence -> presence), and a Qwen judge scores the modern form higher because
it is what that Qwen believes. Widening the pool put the right archaic form in front of
the judges far more often and changed nothing, because they kept preferring the wrong
one. A model of the transcriptions themselves has the archaic spellings built in.

Trained only on the competition's training transcriptions, so it is rules-clean. With
--measure it is trained without fold 0, so nothing it scores is something it has seen.
The score is the mean log-probability per character, the same per-unit convention the
judges use, and it enters the sum with a weight tuned on fold 0 -- at zero it is the
plain ensemble.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from road import cv, metric, paths, submission  # noqa: E402

BOS, EOS = "\x02", "\x03"


class CharLM:
    """Interpolated Witten-Bell character n-gram. Small, exact, and fast enough to score
    tens of thousands of candidates on a CPU."""

    def __init__(self, texts, n: int = 6):
        self.n = n
        self.counts = [defaultdict(Counter) for _ in range(n)]
        for text in texts:
            s = BOS * (n - 1) + text + EOS
            for i in range(n - 1, len(s)):
                for k in range(n):
                    self.counts[k][s[i - k:i]][s[i]] += 1
        self.totals = [{h: sum(t.values()) for h, t in order.items()} for order in self.counts]
        self.uniform = 1.0 / (len({c for t in texts for c in t}) + 1)

    def prob(self, history: str, c: str) -> float:
        p = self.uniform
        for k in range(self.n):
            h = history[len(history) - k:] if k else ""
            table = self.counts[k].get(h)
            if not table:
                continue
            total, types = self.totals[k][h], len(table)
            lam = total / (total + types)
            p = lam * table[c] / total + (1 - lam) * p
        return p

    def score(self, text: str) -> float:
        """Mean log-probability per character, end marker included."""
        s = BOS * (self.n - 1) + text + EOS
        lps = [math.log(self.prob(s[i - self.n + 1:i], s[i])) for i in range(self.n - 1, len(s))]
        return sum(lps) / len(lps)


def choose(scores: pd.DataFrame, judges: list[str], weight: float) -> pd.Series:
    """Per line, the candidate with the best judge total plus weight x LM score."""
    key = scores[judges].astype(float).sum(axis=1) + weight * scores["lm"]
    best = scores.assign(key=key).groupby("ID", sort=False).key.idxmax()
    return scores.loc[best].set_index("ID").Target


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="an ensemble run with a scores.csv")
    ap.add_argument("--measure", action="store_true",
                    help="held-out run: sweep the weight against the truth")
    ap.add_argument("--weight", type=float, default=None, help="LM weight to apply")
    ap.add_argument("--order", type=int, default=6, help="n-gram order")
    args = ap.parse_args(argv)

    run_dir = paths.runs_dir() / args.run
    scores = pd.read_csv(run_dir / "scores.csv", keep_default_na=False, dtype=str)
    judges = [c for c in scores.columns if c not in ("ID", "Target")]
    train = cv.load_train()

    if args.measure:
        folds = cv.make_folds(train.ID)
        held = set(scores.ID)
        texts = train[~train.ID.isin(held)].Target.astype(str)
        print(f"LM trained on {len(texts)} lines, none of them the {len(held)} scored here")
    else:
        texts = train.Target.astype(str)
        print(f"LM trained on all {len(texts)} training lines")
    lm = CharLM(texts, args.order)
    scores["lm"] = [lm.score(t) for t in scores.Target]
    print(f"scored {len(scores)} candidates; LM mean {scores.lm.mean():+.3f}, "
          f"sd {scores.lm.std():.3f}\n")

    if args.measure:
        truth = train.set_index("ID").Target.astype(str)
        ids = list(dict.fromkeys(scores.ID))
        refs = [truth[i] for i in ids]
        err = {}
        print(f"  {'weight':>8}{'public':>10}{'delta':>10}{'lines changed':>16}")
        base_pick = choose(scores, judges, 0.0)
        for w in (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0):
            pick = choose(scores, judges, w)
            err[w] = metric.line_errors(refs, [pick[i] for i in ids])
            pub = metric.public_estimate(err[w])
            base = metric.public_estimate(err[0.0])
            changed = sum(pick[i] != base_pick[i] for i in ids)
            print(f"  {w:>8.2f}{pub:>10.4f}{pub - base:>+10.4f}{changed:>16}")
        best = max(err, key=lambda w: metric.public_estimate(err[w]))
        if best > 0:
            rng = np.random.default_rng(0)
            idx = rng.integers(0, len(ids), size=(500, len(ids)))
            d = np.array([metric.public_estimate(err[best].iloc[k])
                          - metric.public_estimate(err[0.0].iloc[k]) for k in idx])
            print(f"\n  best weight {best}: paired bootstrap {d.mean():+.4f} +/- {d.std():.4f}, "
                  f"{(d > 0).mean():.0%} of resamples favour it")
        return 0

    if args.weight is None:
        sys.exit("--weight is required unless --measure")
    pick = choose(scores, judges, args.weight)
    old = pd.read_csv(run_dir / f"{args.run}.csv", keep_default_na=False,
                      dtype=str).set_index("ID").Target
    changed = sum(pick[i] != old[i] for i in pick.index if i in old.index)
    print(f"weight {args.weight}: {changed} of {len(pick)} lines differ from {args.run}")
    out = f"{args.run}_lm"
    path = submission.write(pick.index, pick.values, out)
    out_dir = paths.runs_dir() / out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / path.name).write_bytes(path.read_bytes())
    print(f"{len(pick)} lines -> {out_dir / path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
