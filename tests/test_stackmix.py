"""StackMix: word cuts that can be trusted, and synthetic lines whose labels are right.

The failure this guards against is the quiet one: a merge and a split that cancel, so
the word count still matches and every label after them is shifted. That was three of
four accepted lines before the width check, and it would have filled training with lines
whose text does not match their image.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageDraw

from road import paths, stackmix

needs_data = pytest.mark.skipif(not (paths.data_dir() / "Train.csv").exists(),
                                reason="dataset not present")


def blocks(widths, gap=20, height=64):
    """A line of solid ink blocks separated by clear gaps: easy, known cuts."""
    total = sum(widths) + gap * (len(widths) + 1)
    img = Image.new("L", (total, height), 235)
    d = ImageDraw.Draw(img)
    x = gap
    for w in widths:
        d.rectangle([x, 18, x + w, 50], fill=30)
        x += w + gap
    return img.convert("RGB")


def test_a_clean_line_cuts_into_its_words():
    img = blocks([60, 40, 80])
    cut = stackmix.segment(img, ["alpha", "and", "omega"])
    assert cut is not None and len(cut) == 3
    assert all(a < b for a, b in cut)


def test_the_wrong_word_count_is_refused():
    assert stackmix.segment(blocks([60, 40, 80]), ["only", "two"]) is None
    assert stackmix.segment(blocks([60, 40]), []) is None


def test_a_merge_that_keeps_the_count_is_refused_by_width():
    """Three pieces for three words, but one piece is far too wide for its label: the
    signature of a merge cancelled by a split elsewhere."""
    segs = [(0, 40), (60, 300), (320, 360)]            # middle piece holds two words
    assert stackmix.consistent(segs, ["the", "said", "and"]) is False
    assert stackmix.consistent([(0, 40), (60, 110), (130, 170)],
                               ["the", "said", "and"]) is True


def test_synthetic_lines_keep_their_words_and_are_deterministic():
    crops = [stackmix.Crop(Image.new("RGB", (30 + 5 * i, stackmix.OUT_H), (220, 210, 190)),
                           w, f"src{i // 4}")
             for i, w in enumerate(["the", "said", "land", "and", "to", "have", "and", "hold"])]
    a = stackmix.synthesise(crops, 20, seed=7)
    b = stackmix.synthesise(crops, 20, seed=7)
    assert [t for _, t in a] == [t for _, t in b]                 # the rules want it reproducible
    vocab = {c.word for c in crops}
    for img, text in a:
        assert img.height == stackmix.OUT_H
        assert set(text.split()) <= vocab                         # only words that were cut


def test_synthetic_ids_live_in_their_own_folder():
    """The competition's images are never mixed with anything we made."""
    assert paths.image_path(f"{paths.SYNTH_PREFIX}000001").parent == paths.synth_dir()
    assert paths.image_path("Zz9TestImage0001").parent == paths.images_dir()


def test_recolouring_matches_the_paper():
    img = Image.fromarray(np.full((20, 20, 3), (200, 180, 150), dtype=np.uint8))
    out = stackmix.recolour(img, np.array([230.0, 220.0, 200.0]))
    assert np.allclose(stackmix.paper(out), [230, 220, 200], atol=2)


@needs_data
def test_held_out_fold_is_never_cut(tmp_path, monkeypatch):
    """A fold-0 experiment must not train on crops of fold-0 lines."""
    from road import cv
    train = cv.load_train()
    folds = cv.make_folds(train.ID)
    held = set(train.ID[train.ID.map(folds) == 0])
    kept = train[~train.ID.isin(held)].head(80)
    crops = stackmix.harvest(kept)
    assert crops and not {c.source for c in crops} & held
