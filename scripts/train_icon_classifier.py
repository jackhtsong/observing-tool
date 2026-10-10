"""
train_icon_classifier.py: teach the dot detector what a player icon looks like, from your clicks.

Steps:
  1. For every frame you labelled (data/labels/dots.json), run the circle
     finder and keep every candidate on the floor. A candidate within 6 px
     of one of your clicks is a "player" example; every other candidate is
     a "not a player" example (ability icons, walls, X's, tinted floor...).
  2. Check: for each video in turn, train on the other videos only, then
     run the whole detector with that model on the left-out video and
     grade it against your clicks (same grading as scripts/score_dots.py).
     Also grade the old rules-only detector, for comparison.
  3. Train on all videos and save perception/models/icon_classifier.npz,
     which the detector then uses automatically.

Run it from the project root (after labelling with scripts.label_dots):

    .venv/bin/python -m scripts.train_icon_classifier

Re-run it whenever you've labelled more frames.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from capture.video_source import VideoSource
from perception.dot_labels import load_labels, score_frame
from perception.icon_classifier import DEFAULT_PATH, IconClassifier, crop_features
from perception.map_background import floor_mask, load_or_build_background
from perception.minimap_dots import DotDetector
from perception.regions import MINIMAP


def load_frames() -> list[tuple]:
    """Every finished labelled frame as (video, labelled frame, minimap crop, background)."""
    frames, backgrounds = [], {}
    for label in (f for f in load_labels() if f.done):
        with VideoSource(label.video) as video:
            if label.video not in backgrounds:
                backgrounds[label.video] = load_or_build_background(video, MINIMAP)
            minimap = MINIMAP.crop(video.frame_at(label.time)).copy()
        frames.append((label.video, label, minimap, backgrounds[label.video]))
    return frames


def examples(frames: list[tuple]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Step 1: (features, labels, video) for every candidate circle."""
    rows, labels, videos = [], [], []
    detectors = {}
    for video, label, minimap, background in frames:
        if video not in detectors:
            detectors[video] = DotDetector(background, floor_mask(background), classifier=None)
        det = detectors[video]
        for x, y, _ in det._find_circles(minimap):
            if not det._inside(x, y) or not det.allowed_area[int(y), int(x)]:
                continue
            near_click = any(np.hypot(p.x - x, p.y - y) <= 6 for p in label.players)
            rows.append(crop_features(minimap, background, x, y))
            labels.append(1.0 if near_click else 0.0)
            videos.append(video)
    return np.array(rows), np.array(labels), np.array(videos)


def grade(frames: list[tuple], classifier: IconClassifier | None) -> Counter:
    """Run the full detector on these frames and grade it against the clicks."""
    totals: Counter = Counter()
    detectors = {}
    for video, label, minimap, background in frames:
        if video not in detectors:
            detectors[video] = DotDetector(background, floor_mask(background), classifier=classifier)
        s = score_frame(label.players, detectors[video].detect(minimap))
        for key in ("found", "missed", "extra", "wrong_team"):
            totals[key] += getattr(s, key)
    return totals


def describe(c: Counter) -> str:
    labelled = c["found"] + c["missed"] + c["wrong_team"]
    detected = c["found"] + c["extra"] + c["wrong_team"]
    return (f"found {c['found'] / max(labelled, 1):4.0%}  precision {c['found'] / max(detected, 1):4.0%}  "
            f"({c['extra']:3d} false dots, {c['wrong_team']} wrong team)")


def main() -> None:
    print("Loading your labelled frames...")
    frames = load_frames()
    features, labels, videos = examples(frames)
    print(f"{len(labels)} candidate circles, {int(labels.sum())} of them near a click\n")

    print("Checked on each video with a model trained only on the others:")
    old_total, new_total = Counter(), Counter()
    for video in sorted(set(videos)):
        train = videos != video
        model = IconClassifier.fit(features[train], labels[train])
        held_out = [f for f in frames if f[0] == video]
        old, new = grade(held_out, None), grade(held_out, model)
        old_total.update(old)
        new_total.update(new)
        name = Path(video).stem
        print(f"  {name:22s} rules   {describe(old)}")
        print(f"  {'':22s} learned {describe(new)}")
    print(f"\n  {'ALL':22s} rules   {describe(old_total)}")
    print(f"  {'':22s} learned {describe(new_total)}")

    IconClassifier.fit(features, labels).save(DEFAULT_PATH)
    print(f"\nSaved {DEFAULT_PATH} (trained on all {len(frames)} frames). The detector now uses it.")


if __name__ == "__main__":
    main()
