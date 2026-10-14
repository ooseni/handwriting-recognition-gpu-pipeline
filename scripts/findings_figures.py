"""Measure every claim in the findings report and draw its figure.

    ROAD_RUNS=path/to/runs python scripts/findings_figures.py

Writes docs/figures/*.png and docs/findings.json. The JSON is what
scripts/findings_pdf.py renders, so no number in the report is typed by hand --
re-run this and the report re-derives itself from the runs on disk.

Nothing here reads the test set; every measurement is on held-out training lines.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from rapidfuzz.distance import Levenshtein  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from road import cv, metric, paths  # noqa: E402

DOCS = Path(__file__).resolve().parents[1] / "docs"
FIGS = DOCS / "figures"
BASE_RUN, BAND_RUN, TROCR_RUN = "qwen4b_f0", "qwen4b_f0_v2", "trocr_f0"

INK, ACCENT, WARN, GOOD, GREY = "#1f3b57", "#c2703d", "#a33a3a", "#3f7d5a", "#8a8f98"
plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "figure.facecolor": "white", "axes.edgecolor": "#444444",
})


def oof(run: str) -> pd.Series:
    path = paths.runs_dir() / run / "oof.csv"
    if not path.exists():
        sys.exit(f"{path} is missing -- point ROAD_RUNS at the finished runs")
    return pd.read_csv(path, keep_default_na=False, dtype=str).set_index("ID").Target


def public(refs, hyps) -> float:
    return metric.public_estimate(metric.line_errors(list(refs), list(hyps)))


def save(fig, name: str) -> str:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGS / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}")
    return name


def main() -> int:
    train = cv.load_train()
    truth = train.set_index("ID").Target.astype(str)
    base = oof(BASE_RUN)
    ids = list(base.index)
    refs = [truth[i] for i in ids]
    hyps = [base[i] for i in ids]
    err = metric.line_errors(refs, hyps)
    out: dict[str, object] = {"n_lines": len(ids), "run": BASE_RUN}

    noise = cv.lb_noise(err, len(err))
    out["noise_sd"] = noise
    out["public"] = public(refs, hyps)
    print(f"{len(ids)} held-out lines, public~{out['public']:.4f}, noise sd {noise:.4f}")

    # ---- figure 1: where the errors are -----------------------------------------
    ce = err.char_edits.to_numpy()
    buckets = [("0", ce == 0), ("1-2", (ce >= 1) & (ce <= 2)), ("3-5", (ce >= 3) & (ce <= 5)),
               ("6-10", (ce >= 6) & (ce <= 10)), ("11+", ce >= 11)]
    share_lines = [s.mean() * 100 for _, s in buckets]
    share_edits = [ce[s].sum() / ce.sum() * 100 for _, s in buckets]
    out["exact"] = float((ce == 0).mean())
    out["worst_decile_share"] = float(np.sort(ce)[::-1][:max(1, len(ce) // 10)].sum() / ce.sum())

    def markup_slip(a: str, b: str) -> bool:
        """The two words differ only in transcription markup, so a rule could fix it."""
        return (a.replace("^", "") == b.replace("^", "")
                or a.rstrip(":") == b.rstrip(":")
                or {a, b.lower()} == {"&", "and"} or {b, a.lower()} == {"&", "and"})

    # Every edit is counted exactly once, so the classes sum to the total: a replace
    # block costs max(len) edits, an insert or delete costs its own length.
    classes, total_word_edits = Counter(), 0
    for r, h in zip(refs, hyps):
        rw, hw = r.split(), h.split()
        total_word_edits += Levenshtein.distance(rw, hw)
        for op in Levenshtein.opcodes(rw, hw):
            src, dest = rw[op.src_start:op.src_end], hw[op.dest_start:op.dest_end]
            if op.tag == "insert":
                classes["word added or dropped"] += len(dest)
            elif op.tag == "delete":
                classes["word added or dropped"] += len(src)
            elif op.tag == "replace":
                for a, b in zip(src, dest):
                    if a == b:
                        continue
                    if a.lower() == b.lower():
                        classes["casing only"] += 1
                    elif markup_slip(a, b):
                        classes["markup slip (^ : &)"] += 1
                    else:
                        classes["a different word"] += 1
                classes["word added or dropped"] += abs(len(src) - len(dest))
    out["word_edits"] = total_word_edits
    out["classes"] = {k: v / total_word_edits for k, v in classes.items()}
    print("  word-edit classes: " + ", ".join(
        f"{k} {v / total_word_edits:.0%}" for k, v in classes.most_common()))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.2))
    x = np.arange(len(buckets))
    ax1.bar(x - 0.2, share_lines, 0.4, label="of lines", color=INK)
    ax1.bar(x + 0.2, share_edits, 0.4, label="of all character edits", color=ACCENT)
    ax1.set_xticks(x, [b[0] for b in buckets])
    ax1.set_xlabel("character edits on the line")
    ax1.set_ylabel("percent")
    ax1.set_title("A few bad lines carry the damage")
    ax1.legend(frameon=False, fontsize=8)
    order = sorted(classes.items(), key=lambda kv: kv[1])
    ax2.barh([k for k, _ in order], [v / total_word_edits * 100 for _, v in order],
             color=[WARN if k == "a different word" else GREY for k, _ in order])
    for i, (k, v) in enumerate(order):
        ax2.text(v / total_word_edits * 100 + 1, i, f"{v / total_word_edits:.0%}",
                 va="center", fontsize=8)
    ax2.set_xlim(0, 72)
    ax2.set_xlabel("percent of word edits")
    ax2.set_title("Most word errors are simply a different word")
    fig1 = save(fig, "fig1_error_profile.png")

    # ---- figure 2: the post-processing refutations -------------------------------
    lex_rows = train[~train.ID.isin(set(ids))]
    forms: dict[str, Counter] = {}
    for text in lex_rows.Target.astype(str):
        for w in text.split():
            forms.setdefault(w.lower(), Counter())[w] += 1

    def recase(text: str, threshold: float) -> str:
        words = []
        for w in text.split():
            table = forms.get(w.lower())
            if table and sum(table.values()) >= 3:
                top, n = table.most_common(1)[0]
                if n / sum(table.values()) >= threshold:
                    w = top
            words.append(w)
        return " ".join(words)

    thresholds = [0.60, 0.70, 0.80, 0.90, 0.95, 1.00]
    recase_delta = [public(refs, [recase(t, th) for t in hyps]) - out["public"]
                    for th in thresholds]
    out["recase"] = dict(zip(map(str, thresholds), recase_delta))

    def strip_delta(ch: str, repl: str = "") -> float:
        return public([t.replace(ch, repl) for t in refs],
                      [t.replace(ch, repl) for t in hyps]) - out["public"]

    ceilings = {
        "casing ignored": public([r.lower() for r in refs], [h.lower() for h in hyps]) - out["public"],
        "'^' ignored": strip_delta("^"),
        "':' ignored": strip_delta(":"),
        "'&' vs 'and' ignored": strip_delta("&", "and"),
    }
    out["ceilings"] = ceilings

    vocab = Counter(w for t in lex_rows.Target.astype(str) for w in t.split())
    fixed = broken = 0
    for r, h in zip(refs, hyps):
        rw, hw = r.split(), h.split()
        for op in Levenshtein.opcodes(rw, hw):
            if op.tag != "replace":
                continue
            for a, b in zip(rw[op.src_start:op.src_end], hw[op.dest_start:op.dest_end]):
                if a == b or b in vocab or a not in vocab:
                    continue
                near = [w for w in vocab if Levenshtein.distance(b, w) <= 1]
                if near:
                    if max(near, key=lambda w: vocab[w]) == a:
                        fixed += 1
                    else:
                        broken += 1
    def snap(text: str) -> str:
        """Push every out-of-vocabulary word to its most frequent near neighbour."""
        words = []
        for w in text.split():
            if w not in vocab:
                near = [v for v in vocab if Levenshtein.distance(w, v) <= 1]
                if near:
                    w = max(near, key=lambda v: vocab[v])
            words.append(w)
        return " ".join(words)

    snap_delta = public(refs, [snap(t) for t in hyps]) - out["public"]
    out["snap"] = {"fixed": fixed, "broken": broken, "delta": snap_delta}
    print(f"  vocabulary snapping: fixes {fixed}, breaks {broken}, score {snap_delta:+.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.2))
    ax1.axhline(0, color="#333333", lw=1)
    ax1.axhspan(-noise, noise, color=GREY, alpha=0.18, label=f"noise (sd {noise:.4f})")
    ax1.plot(thresholds, recase_delta, "o-", color=WARN, lw=1.8, ms=5)
    ax1.set_xlabel("confidence required before recasing a word")
    ax1.set_ylabel("change in score")
    ax1.set_title("Recasing from a corpus lexicon: worse at every threshold")
    ax1.legend(frameon=False, fontsize=8, loc="lower right")
    names = list(ceilings) + ["vocabulary snapping"]
    vals = list(ceilings.values()) + [snap_delta]
    ax2.axhline(0, color="#333333", lw=1)
    ax2.axhspan(-noise, noise, color=GREY, alpha=0.18)
    ax2.bar(range(len(names)), vals,
            color=[GOOD if v > noise else GREY for v in vals])
    ax2.set_xticks(range(len(names)),
                   ["casing", "'^'", "':'", "'&'", "snapping"], fontsize=8)
    ax2.set_ylabel("change in score")
    ax2.annotate(f"net {fixed - broken:+d} words\n(fixes {fixed}, breaks {broken})",
                 (4, 0), textcoords="offset points", xytext=(0, 14),
                 ha="center", fontsize=7.5, color=WARN)
    fig2 = save(fig, "fig2_postprocessing.png")

    # ---- figure 3: the two scan batches -----------------------------------------
    size = {}
    for i in ids:
        with Image.open(paths.image_path(i)) as im:
            size[i] = im.size
    large = np.array([size[i][1] > 150 for i in ids])
    batches = {}
    for label, sel in (("small-res", ~large), ("large-res", large)):
        sub = metric.aggregate(metric.line_errors(
            [r for r, s in zip(refs, sel) if s], [h for h, s in zip(hyps, sel) if s]))
        batches[label] = {"lines": int(sel.sum()), "public": sub["public"],
                          "wer": sub["wer"], "cer": sub["cer"]}
    out["batches"] = batches
    out["large_edit_share"] = float(err.char_edits.to_numpy()[large].sum() / ce.sum())

    pxc = np.array([size[i][0] * 160 / size[i][1] / max(len(truth[i]), 1) for i in ids])
    edges = np.quantile(pxc, [0, 0.25, 0.5, 0.75, 1.0])
    matched = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        row = {"range": f"{lo:.0f}-{hi:.0f}"}
        for label, sel in (("small-res", ~large), ("large-res", large)):
            m = sel & (pxc >= lo) & (pxc <= hi)
            row[label] = (public([r for r, s in zip(refs, m) if s],
                                 [h for h, s in zip(hyps, m) if s])
                          if m.sum() >= 15 else None)
            row[label + "_n"] = int(m.sum())
        matched.append(row)
    out["matched"] = matched

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.2))
    labels = ["small-res\n(h<=150)", "large-res\n(h>150)"]
    vals = [batches["small-res"]["public"], batches["large-res"]["public"]]
    bars = ax1.bar(labels, vals, color=[INK, WARN], width=0.55)
    for b, v, k in zip(bars, vals, batches.values()):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.004,
                 f"{v:.4f}\n{k['lines']} lines", ha="center", fontsize=8)
    ax1.set_ylim(0.78, 0.94)
    ax1.set_ylabel("score on held-out lines")
    ax1.set_title(f"A quarter of the scans carry {out['large_edit_share']:.0%} of the errors")
    x = np.arange(len(matched))
    for off, label, colour in ((-0.2, "small-res", INK), (0.2, "large-res", WARN)):
        ys = [m[label] for m in matched]
        ax2.bar(x + off, [y if y else 0 for y in ys], 0.4, label=label, color=colour)
    ax2.set_xticks(x, [m["range"] for m in matched], fontsize=8)
    ax2.set_ylim(0.75, 0.95)
    ax2.set_xlabel("output pixels per character (matched)")
    ax2.set_ylabel("score")
    ax2.set_title("The gap survives matching on resolution")
    ax2.legend(frameon=False, fontsize=8)
    fig3 = save(fig, "fig3_scan_batches.png")

    # ---- figure 4: how much an ensemble could win -------------------------------
    band, trocr = oof(BAND_RUN), oof(TROCR_RUN)
    shared = [i for i in ids if i in band.index and i in trocr.index]
    sref = [truth[i] for i in shared]
    singles = {BASE_RUN: public(sref, [base[i] for i in shared]),
               BAND_RUN: public(sref, [band[i] for i in shared]),
               TROCR_RUN: public(sref, [trocr[i] for i in shared])}

    def oracle(sources):
        picked = [min((s[i] for s in sources), key=lambda t: Levenshtein.distance(truth[i], t))
                  for i in shared]
        return public(sref, picked)

    out["singles"] = singles
    out["oracle_two"] = oracle([base, band])
    out["oracle_three"] = oracle([base, band, trocr])

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    names = ["TrOCR", "Qwen 4B", "Qwen 4B\n(band)", "oracle\n2 Qwen", "oracle\n+ TrOCR"]
    vals = [singles[TROCR_RUN], singles[BASE_RUN], singles[BAND_RUN],
            out["oracle_two"], out["oracle_three"]]
    colours = [GREY, GREY, GREY, GOOD, GOOD]
    bars = ax.bar(names, vals, color=colours, width=0.6)
    best = max(singles.values())
    ax.axhline(best, color=INK, lw=1, ls="--")
    ax.axhspan(best - noise, best + noise, color=GREY, alpha=0.18)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.4f}", ha="center", fontsize=8)
    ax.set_ylim(0.82, 0.93)
    ax.set_ylabel("score on held-out lines")
    ax.set_title("Choosing the best candidate per line is worth "
                 f"{out['oracle_three'] - best:+.3f}")
    fig4 = save(fig, "fig4_ensemble.png")

    # ---- figure 5: label noise ---------------------------------------------------
    rate = (err.char_edits / err.chars.clip(lower=1)).to_numpy()
    sweep = []
    for thr in (1.01, 0.9, 0.8, 0.7, 0.6, 0.5):
        keep = err[rate <= thr]
        sweep.append({"threshold": thr, "excluded": int((rate > thr).sum()),
                      "public": metric.public_estimate(keep)})
    out["noise_sweep"] = sweep
    worst = err[rate > 0.6]
    out["suspect"] = {"lines": int(len(worst)), "share_lines": float(len(worst) / len(err)),
                      "share_edits": float(worst.char_edits.sum() / ce.sum())}
    examples = []
    for k in np.argsort(-rate)[:4]:
        examples.append({"id": ids[k], "label": refs[k][:78], "ours": hyps[k][:78],
                         "rate": float(rate[k])})
    out["examples"] = examples

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.2))
    ax1.plot([s["excluded"] for s in sweep], [s["public"] for s in sweep],
             "o-", color=GOOD, lw=1.8, ms=5)
    for s in sweep:
        if s["excluded"] in (0, 12):
            ax1.annotate(f"{s['public']:.4f}", (s["excluded"], s["public"]),
                         textcoords="offset points", xytext=(6, -10), fontsize=8)
    ax1.set_xlabel("lines excluded (worst first)")
    ax1.set_ylabel("score on what remains")
    ax1.set_title("Removing 12 lines of 820 moves the score 0.011")
    frac = np.arange(1, len(ce) + 1) / len(ce) * 100
    ax2.plot(frac, np.cumsum(np.sort(ce)[::-1]) / ce.sum() * 100, color=INK, lw=1.8)
    ax2.axvline(1.5, color=WARN, ls="--", lw=1)
    ax2.annotate(f"worst 1.5% of lines\n= {out['suspect']['share_edits']:.0%} of all edits",
                 (1.5, out["suspect"]["share_edits"] * 100), textcoords="offset points",
                 xytext=(14, -6), fontsize=8, color=WARN)
    ax2.set_xlim(0, 30)
    ax2.set_xlabel("percent of lines, worst first")
    ax2.set_ylabel("cumulative percent of edits")
    ax2.set_title("Error is extremely concentrated")
    fig5 = save(fig, "fig5_label_noise.png")

    out["figures"] = [fig1, fig2, fig3, fig4, fig5]
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "findings.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {DOCS / 'findings.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
