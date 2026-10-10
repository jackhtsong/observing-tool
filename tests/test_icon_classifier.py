"""
test_icon_classifier.py: checks for the learned "is this a player icon?" model.

Uses made-up crops: "players" are a dark disc inside a coloured ring, "clutter"
is a coloured straight line. The model should learn to tell them apart, and
survive saving and loading.

Run from the project root:

    .venv/bin/python -m pytest tests/
"""

import cv2
import numpy as np

from perception.icon_classifier import IconClassifier, crop_features


def scene(kind: str, rng) -> np.ndarray:
    """A 64x64 grey patch with either an icon or a line in the middle."""
    img = np.full((64, 64, 3), 130, np.uint8)
    colour = (70, 60, 220) if rng.random() < 0.5 else (200, 220, 90)
    if kind == "icon":
        cv2.circle(img, (32, 32), 8, colour, -1)
        cv2.circle(img, (32, 32), 6, (int(rng.integers(40, 90)),) * 3, -1)
    else:
        angle = rng.random() * np.pi
        dx, dy = int(20 * np.cos(angle)), int(20 * np.sin(angle))
        cv2.line(img, (32 - dx, 32 - dy), (32 + dx, 32 + dy), colour, 3)
    return img


def make_examples(n: int, seed: int):
    rng = np.random.default_rng(seed)
    background = np.full((64, 64, 3), 130, np.uint8)
    rows, labels = [], []
    for i in range(n):
        kind = "icon" if i % 2 else "line"
        rows.append(crop_features(scene(kind, rng), background, 32, 32))
        labels.append(1.0 if kind == "icon" else 0.0)
    return np.array(rows), np.array(labels)


def test_learns_icons_from_lines():
    train_x, train_y = make_examples(200, seed=1)
    test_x, test_y = make_examples(60, seed=2)
    model = IconClassifier.fit(train_x, train_y)
    accuracy = ((model.predict(test_x) > 0.5) == (test_y == 1)).mean()
    assert accuracy > 0.95


def test_save_and_load(tmp_path):
    x, y = make_examples(40, seed=3)
    model = IconClassifier.fit(x, y, epochs=20)
    path = tmp_path / "icons.npz"
    model.save(path)
    assert np.allclose(IconClassifier.load(path).predict(x), model.predict(x))
    assert IconClassifier.load(tmp_path / "missing.npz") is None
