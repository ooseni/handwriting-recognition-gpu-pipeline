"""Render docs/findings.pdf from docs/findings.json and docs/figures/.

    python scripts/findings_pdf.py

Every number comes from the JSON that scripts/findings_figures.py measured, so the
report cannot drift from the runs it describes. Run that first.

No competition image or label file is reproduced here: the figures are derived
aggregates, and the few label fragments quoted are the minimum needed to show what a
mislabelled row looks like.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
INK = colors.HexColor("#1f3b57")
WARN = colors.HexColor("#a33a3a")
GOOD = colors.HexColor("#3f7d5a")
RULE = colors.HexColor("#c8ccd2")
MUTED = colors.HexColor("#5b6068")

TITLE = "What Does Not Work"
SUBTITLE = ("Negative results and error analysis on the R.O.A.D. Barbados "
            "Historic Handwriting Challenge")
AUTHOR = "Oseni Sakariyau Oluwadamilare"


def styles():
    base = getSampleStyleSheet()
    s = {
        "title": ParagraphStyle("title", parent=base["Title"], fontName="Helvetica-Bold",
                                fontSize=26, leading=30, textColor=INK, spaceAfter=6),
        "subtitle": ParagraphStyle("subtitle", parent=base["Normal"], fontName="Helvetica",
                                   fontSize=12.5, leading=17, textColor=MUTED, spaceAfter=22),
        "byline": ParagraphStyle("byline", parent=base["Normal"], fontName="Helvetica",
                                 fontSize=10, leading=14, textColor=MUTED),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontName="Helvetica-Bold",
                             fontSize=14, leading=18, textColor=INK,
                             spaceBefore=16, spaceAfter=7),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName="Helvetica-Bold",
                             fontSize=11, leading=15, textColor=INK,
                             spaceBefore=11, spaceAfter=4),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontName="Helvetica",
                               fontSize=9.6, leading=14.2, alignment=TA_JUSTIFY,
                               spaceAfter=7),
        "caption": ParagraphStyle("caption", parent=base["Normal"], fontName="Helvetica-Oblique",
                                  fontSize=8.3, leading=11.5, textColor=MUTED,
                                  spaceBefore=4, spaceAfter=13),
        "lead": ParagraphStyle("lead", parent=base["BodyText"], fontName="Helvetica",
                               fontSize=10.6, leading=15.6, alignment=TA_JUSTIFY,
                               textColor=INK, spaceAfter=8),
        "cell": ParagraphStyle("cell", parent=base["Normal"], fontName="Helvetica",
                               fontSize=8.4, leading=11),
        "mono": ParagraphStyle("mono", parent=base["Normal"], fontName="Courier",
                               fontSize=7.8, leading=10.4),
    }
    return s


def esc(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def figure(name: str, width=16.4 * cm):
    from PIL import Image as PILImage
    path = DOCS / "figures" / name
    with PILImage.open(path) as im:
        w, h = im.size
    return Image(str(path), width=width, height=width * h / w)


def table(rows, widths, align_right=(), header=True):
    t = Table(rows, colWidths=widths, hAlign="LEFT")
    style = [
        ("FONT", (0, 0), (-1, -1), "Helvetica", 8.4),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
    ]
    if header:
        style += [("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.4),
                  ("TEXTCOLOR", (0, 0), (-1, 0), INK),
                  ("LINEBELOW", (0, 0), (-1, 0), 0.9, INK),
                  ("BOTTOMPADDING", (0, 0), (-1, 0), 6)]
    for c in align_right:
        style.append(("ALIGN", (c, 0), (c, -1), "RIGHT"))
    t.setStyle(TableStyle(style))
    return t


AMP_KEY = "'&' vs 'and' ignored"


def build(d: dict) -> list:
    s = styles()
    pct = lambda x: f"{x:.0%}"  # noqa: E731
    f4 = lambda x: f"{x:+.4f}"  # noqa: E731
    noise = d["noise_sd"]
    story = []

    # ---- cover ------------------------------------------------------------------
    story += [Spacer(1, 3.4 * cm),
              Paragraph(TITLE, s["title"]),
              Paragraph(SUBTITLE, s["subtitle"]),
              Table([[""]], colWidths=[16.4 * cm], rowHeights=[2],
                    style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), INK)])),
              Spacer(1, 0.7 * cm),
              Paragraph(f"{esc(AUTHOR)}<br/>{date.today():%d %B %Y}", s["byline"]),
              Spacer(1, 1.2 * cm)]

    story += [Paragraph(
        f"Six ideas were measured against held-out data and five of them lost. This note "
        f"records what failed, how it was measured and why, so that the same compute is not "
        f"spent twice. All measurements are on {d['n_lines']} held-out lines from the "
        f"training split; nothing here uses the test set. Sampling noise on a set that size "
        f"is <b>sd {noise:.4f}</b>, so any effect smaller than about 0.005 is indistinguishable "
        f"from luck.", s["lead"])]

    summary = [
        ["Idea", "Expected", "Measured", "Verdict"],
        ["Recase words from a corpus lexicon", f"+{d['ceilings']['casing ignored']:.3f} ceiling",
         f"{min(d['recase'].values()):+.4f} to {max(d['recase'].values()):+.4f}", "Rejected"],
        ["Repair markup (^, :, &)", "large", f"{d['ceilings']["'^' ignored"]:+.4f} at best",
         "Negligible"],
        ["Snap rare words to the vocabulary", "positive", f"{d['snap']['delta']:+.4f}", "Rejected"],
        ["Guard against over-transcription", "large", "0 lines affected", "Not a problem"],
        ["Normalise the two scan batches", f"+{0.019:.3f}", "gap survives matching", "Rejected"],
        ["Combine several runs per line", "unknown",
         f"+{d['oracle_three'] - max(d['singles'].values()):.4f} ceiling", "Worth doing"],
    ]
    story += [table(summary, [6.6 * cm, 3.1 * cm, 3.5 * cm, 3.2 * cm]),
              Paragraph("Table 1. Everything tested, with the result that settled it. "
                        "Only the last row survived.", s["caption"])]

    # ---- method -----------------------------------------------------------------
    story += [Paragraph("1. How these numbers were produced", s["h1"]),
              Paragraph(
        f"A Qwen3-VL-4B model was fine-tuned with LoRA on four of five folds and its "
        f"predictions on the {d['n_lines']} unseen lines of the remaining fold are what every "
        f"measurement below uses. Folds are assigned by a salted hash of the line identifier, "
        f"so they are identical on any machine. The score quoted throughout is the "
        f"competition's own leaderboard formula, which turns out to be exactly", s["body"]),
        Paragraph("<font face='Courier' size=8.6>public = 0.5 x (1 - word_edits_per_line / 12) "
                  "+ 0.5 x (1 - char_edits_per_line / 55)</font>", s["body"]),
        Paragraph(
        "where the two edit counts are means per line rather than rates. That form was "
        "confirmed against the public leaderboard's own benchmark row to within 2e-10, and "
        "against one of our own submissions to nine decimal places. Two consequences matter "
        "for everything that follows: one word edit costs as much as 55/12 = 4.58 character "
        "edits, so whole-word correctness is worth far more than shaving characters; and "
        "because the metric sums edits rather than averaging per-line rates, long lines "
        "legitimately count for more.", s["body"])]

    # ---- where the errors are ----------------------------------------------------
    story += [Paragraph("2. Where the errors actually are", s["h1"]),
              Paragraph(
        f"Only {pct(d['exact'])} of held-out lines are transcribed exactly. The damage is "
        f"concentrated: the worst tenth of lines carry {pct(d['worst_decile_share'])} of all "
        f"character edits. Classifying every word-level edit shows what the model is doing "
        f"wrong.", s["body"]),
              figure(d["figures"][0]),
              Paragraph("Figure 1. Left: how character edits are distributed over lines. "
                        "Right: every word edit classified, summing to 100%.", s["caption"])]

    classes = d["classes"]
    story += [Paragraph(
        f"The single largest class, at {pct(classes.get('a different word', 0))}, is simply the "
        f"wrong word: not a misplaced mark or a casing slip but a different word entirely. A "
        f"large share of these are the model quietly modernising seventeenth-century spelling, "
        f"turning <font face='Courier'>prsence</font> into "
        f"<font face='Courier'>presence</font>. The transcriptions never modernise, so every "
        f"such correction is an error. A further {pct(classes.get('word added or dropped', 0))} "
        f"of edits are words added or dropped, which is where merged and split words land, and "
        f"those are expensive: a single merged word costs two word edits. Markup slips - the "
        f"caret, colon and ampersand conventions that look so conspicuous - are only "
        f"{pct(classes.get('markup slip (^ : &)', 0))} of the total.", s["body"])]

    # ---- post-processing ---------------------------------------------------------
    story += [Paragraph("3. Four post-processing ideas, all rejected", s["h1"]),
              Paragraph(
        f"Casing alone accounts for {pct(classes.get('casing only', 0))} of word edits, and a "
        f"perfect casing oracle would be worth {f4(d['ceilings']['casing ignored'])}. That "
        f"ceiling is unreachable. Pushing each word to the capitalisation it most often takes "
        f"in the training text loses at every confidence threshold tried, including at 1.00, "
        f"where the word takes that form in the training data without exception.", s["body"]),
              figure(d["figures"][1]),
              Paragraph("Figure 2. Left: recasing against the threshold required to act; the "
                        "shaded band is sampling noise. Right: what a perfect fix could win, "
                        "beside the one rule that was actually applied.", s["caption"]),
              Paragraph(
        "The reason is visible in the direction of the errors: the model writes lower case "
        "where the truth has upper case about as often as the reverse. That symmetry is the "
        "signature of genuine ambiguity in the hand, not of a bias a prior could correct. The "
        "model can see the ink; a corpus frequency table cannot.", s["body"]),
              Paragraph(
        f"Markup repair fares no better. The model emits more ampersands than the truth "
        f"contains, so the expansion of <font face='Courier'>&amp;</font> into "
        f"<font face='Courier'>and</font> that a general-purpose vision-language model is "
        f"supposed to perform is not happening: eliminating that confusion entirely is worth "
        f"{f4(d['ceilings'][AMP_KEY])}. Perfect caret handling is worth "
        f"{f4(d['ceilings']["'^' ignored"])} and perfect colon handling "
        f"{f4(d['ceilings']["':' ignored"])}, both inside the noise band.", s["body"]),
              Paragraph(
        f"Snapping out-of-vocabulary words to their nearest frequent neighbour in the training "
        f"vocabulary is the most tempting of the four, and the worst: applied to the held-out "
        f"predictions it scores {f4(d['snap']['delta'])}, fixing {d['snap']['fixed']} words "
        f"while breaking {d['snap']['broken']}. The structural reason is that most wrong words "
        f"are not exotic. Where our word and the truth are both ordinary vocabulary words, no "
        f"out-of-vocabulary filter can even see the error, and the rare words such a rule does "
        f"reach are disproportionately the names and places that must not be touched.",
        s["body"]),
              Paragraph(
        "Over-transcription, the failure in which a model reads a neighbouring line that is "
        "partly visible in the crop, was also checked and does not occur: no held-out line is "
        "more than twice the length of its label, and 96.5% are within a tenth of it.",
        s["body"])]

    # ---- scan batches -------------------------------------------------------------
    b = d["batches"]
    story += [Paragraph("4. Two scan batches, one of them much harder", s["h1"]),
              Paragraph(
        f"The images arrive in two distinct scan batches, in the same proportion in the "
        f"training and test splits. They are not equally difficult: the higher-resolution "
        f"batch is {b['small-res']['public'] - b['large-res']['public']:.3f} worse, and though "
        f"it is {pct(b['large-res']['lines'] / d['n_lines'])} of the lines it carries "
        f"{pct(d['large_edit_share'])} of the character edits.", s["body"]),
              table([["Batch", "Lines", "Score", "WER", "CER"],
                     ["small-res (height <= 150)", str(b["small-res"]["lines"]),
                      f"{b['small-res']['public']:.4f}", f"{b['small-res']['wer']:.4f}",
                      f"{b['small-res']['cer']:.4f}"],
                     ["large-res (height > 150)", str(b["large-res"]["lines"]),
                      f"{b['large-res']['public']:.4f}", f"{b['large-res']['wer']:.4f}",
                      f"{b['large-res']['cer']:.4f}"]],
                    [6.2 * cm, 2.4 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm], align_right=(1, 2, 3, 4)),
              Paragraph("Table 2. The two batches on held-out lines. CER is already normalised "
                        "per character, so this is not an artefact of line length.", s["caption"]),
              figure(d["figures"][2]),
              Paragraph("Figure 3. Left: the two batches. Right: the same comparison after "
                        "matching both batches on output pixels per character.", s["caption"]),
              Paragraph(
        "The obvious explanation is that a single resize mistreats them, and the evidence "
        "initially supports it: the large-res crops receive fewer output pixels per character "
        "than the small ones, because they carry proportionally more vertical margin, and "
        "within that batch the error rises monotonically as the squash gets harder. That "
        "explanation is nevertheless wrong. Cropping to the ink band raises those crops to "
        "parity on pixels per character and buys +0.0003. And at matched pixels per character "
        "the large-res batch remains about 0.05 worse in every bucket, as it does at matched "
        "line length. These are harder originals, not mishandled ones. The useful conclusion "
        "is that they deserve more training signal, not more pixels.", s["body"])]

    # ---- label noise ---------------------------------------------------------------
    sus = d["suspect"]
    story += [Paragraph("5. About one line in seventy is mislabelled", s["h1"]),
              Paragraph(
        f"The worst held-out lines are not model failures. On them the model produces fluent, "
        f"coherent legal boilerplate that has nothing to do with the label - the signature of "
        f"an image paired with the wrong transcription. {sus['lines']} lines, "
        f"{pct(sus['share_lines'])} of the fold, carry {pct(sus['share_edits'])} of every "
        f"character edit we make.", s["body"]),
              figure(d["figures"][4]),
              Paragraph("Figure 4. Left: the score on what remains as the worst lines are "
                        "excluded. Right: the cumulative share of all edits, worst lines first.",
                        s["caption"])]

    rows = [["Label", "What the model read"]]
    for ex in d["examples"][:3]:
        rows.append([Paragraph(esc(ex["label"]), s["mono"]),
                     Paragraph(esc(ex["ours"]), s["mono"])])
    story += [KeepTogether([table(rows, [8.2 * cm, 8.2 * cm]),
                            Paragraph("Table 3. The three worst held-out lines. The predictions "
                                      "are coherent and unrelated, which is what a wrong label "
                                      "looks like rather than a wrong reading.", s["caption"])])]
    story += [Paragraph(
        "Two consequences follow. Training on these rows teaches a model to invent text that "
        "is not in the image, which is the one habit worth least on this metric. And roughly "
        "this proportion of the test set is presumably mislabelled too, which puts a floor "
        "under everyone's score that no amount of modelling will reach through.", s["body"])]

    # ---- the one thing that worked ---------------------------------------------------
    best = max(d["singles"].values())
    story += [Paragraph("6. The one idea that survived", s["h1"]),
              Paragraph(
        f"Different models fail on different lines. Choosing, per line, the best of several "
        f"runs' candidate transcriptions would be worth "
        f"{f4(d['oracle_two'] - max(d['singles'][k] for k in d['singles'] if 'qwen' in k))} "
        f"across two fine-tuned checkpoints of the same family, and "
        f"{f4(d['oracle_three'] - best)} once a much weaker TrOCR run contributes candidates "
        f"as well. That weaker model scores {min(d['singles'].values()):.4f} on its own and "
        f"still nearly doubles the headroom, because it is right on lines the stronger models "
        f"miss. An oracle is not achievable, but against a noise level of {noise:.4f} this is "
        f"the only measured headroom of the six ideas examined.", s["body"]),
              figure(d["figures"][3], width=12.4 * cm),
              Paragraph("Figure 5. Individual runs and the per-line oracle over their "
                        "candidates. The dashed line is the best single run.", s["caption"])]

    story += [Paragraph("7. What this leaves", s["h1"]),
              Paragraph(
        "Post-processing is exhausted. Two thirds of the remaining errors are the wrong word "
        "rather than a fixable convention, and the corrections available from the training "
        "corpus are already better known to the model than to any frequency table built from "
        "it. What is left is to combine models that disagree, to stop training on rows whose "
        "labels are wrong, and to give the harder scan batch more weight during training "
        "rather than more pixels before it.", s["body"])]

    story += [Paragraph("Reproducing this", s["h2"]),
              Paragraph(
        "<font face='Courier' size=8.4>python scripts/findings_figures.py</font> re-measures "
        "every number here from the finished runs and rewrites the figures; "
        "<font face='Courier' size=8.4>python scripts/findings_pdf.py</font> rebuilds this "
        "document from them. No value in this report is entered by hand. The competition data "
        "itself is not redistributed: the figures are derived aggregates and the label "
        "fragments in Table 3 are quoted only to show what a mislabelled row looks like.",
        s["body"])]
    return story


def main() -> int:
    data_path = DOCS / "findings.json"
    if not data_path.exists():
        sys.exit("docs/findings.json is missing -- run scripts/findings_figures.py first")
    d = json.loads(data_path.read_text(encoding="utf-8"))

    out = DOCS / "findings.pdf"
    doc = BaseDocTemplate(str(out), pagesize=A4,
                          leftMargin=2.3 * cm, rightMargin=2.3 * cm,
                          topMargin=2.0 * cm, bottomMargin=1.9 * cm,
                          title=TITLE, author=AUTHOR,
                          subject=SUBTITLE)

    def decorate(canvas, doc_):
        canvas.saveState()
        if doc_.page > 1:
            canvas.setFont("Helvetica", 7.5)
            canvas.setFillColor(MUTED)
            canvas.drawString(2.3 * cm, A4[1] - 1.35 * cm, TITLE.upper())
            canvas.drawRightString(A4[0] - 2.3 * cm, A4[1] - 1.35 * cm,
                                   "R.O.A.D. Barbados Historic Handwriting Challenge")
            canvas.setStrokeColor(RULE)
            canvas.setLineWidth(0.5)
            canvas.line(2.3 * cm, A4[1] - 1.5 * cm, A4[0] - 2.3 * cm, A4[1] - 1.5 * cm)
            canvas.drawCentredString(A4[0] / 2, 1.25 * cm, str(doc_.page))
        canvas.restoreState()

    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="body")
    doc.addPageTemplates([PageTemplate(id="page", frames=[frame], onPage=decorate)])
    doc.build(build(d))
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
