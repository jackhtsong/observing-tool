"""
icon_classifier.py: a small learned model that says "is this a player icon?".

Why
---
The dot detector's colour and shape rules can't tell a player icon from
look-alike clutter well enough: ability icons, walls crossing a circle,
death X's, tinted floor. Every rule strict enough to drop the clutter also
drops real players with faint rings. So instead the detector's circle
finder proposes candidates as before, and this model, trained on the
players you clicked (scripts/label_dots.py), decides which are players.

What it looks at (for a 32x32 crop around each candidate)
---------------------------------------------------------
  shape    a "histogram of oriented gradients" (HOG): for each 8x8 block of
           the crop, how much edge there is in each of 9 directions. A
           portrait in a ring has a different edge pattern from a cross, a
           straight wall or an ability symbol.
  colour   how the crop's pixels spread over 12 hues x 4 saturation levels.
  change   how different the crop is from the empty map at that spot.

The model is a tiny neural network: one hidden layer of 32 units, trained
with plain NumPy (no extra libraries). It outputs a probability 0..1.
It is trained by scripts/train_icon_classifier.py and saved to
perception/models/icon_classifier.npz.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

DEFAULT_PATH = Path(__file__).parent / "models" / "icon_classifier.npz"
CROP = 32   # crop size in minimap pixels


def _hog(grey: np.ndarray) -> np.ndarray:
    """Histogram of oriented gradients of a 32x32 grey crop: 4x4 cells x 9 directions."""
    g = grey.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=1)
    strength = np.hypot(gx, gy)
    direction = np.degrees(np.arctan2(gy, gx)) % 180          # edges have no "front", so 0-180
    bins = (direction / 20).astype(int) % 9
    cells = []
    for cy in range(4):
        for cx in range(4):
            block = (slice(cy * 8, cy * 8 + 8), slice(cx * 8, cx * 8 + 8))
            h = np.bincount(bins[block].ravel(), weights=strength[block].ravel(), minlength=9)
            cells.append(h / (np.linalg.norm(h) + 1e-6))       # normalise per cell (lighting)
    return np.concatenate(cells)


def crop_features(minimap: np.ndarray, background: np.ndarray, x: float, y: float) -> np.ndarray:
    """The numbers the model looks at, for a candidate centred at (x, y)."""
    half = CROP // 2
    cx, cy = int(round(x)), int(round(y))
    padded = cv2.copyMakeBorder(minimap, half, half, half, half, cv2.BORDER_REPLICATE)
    padded_bg = cv2.copyMakeBorder(background, half, half, half, half, cv2.BORDER_REPLICATE)
    crop = padded[cy:cy + CROP, cx:cx + CROP]
    crop_bg = padded_bg[cy:cy + CROP, cx:cx + CROP]

    shape = _hog(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    colour = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256]).ravel()
    colour /= colour.sum() + 1e-6
    change = np.abs(crop.astype(np.int16) - crop_bg.astype(np.int16)).mean() / 50.0
    return np.concatenate([shape, colour, [change]])


class IconClassifier:
    """One-hidden-layer neural network: features -> chance it's a player icon."""

    def __init__(self, mean, scale, w1, b1, w2, b2):
        self.mean, self.scale = np.asarray(mean), np.asarray(scale)
        self.w1, self.b1, self.w2, self.b2 = np.asarray(w1), np.asarray(b1), np.asarray(w2), float(b2)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Chance (0..1) for each row of features."""
        z = (np.atleast_2d(features) - self.mean) / self.scale
        hidden = np.maximum(z @ self.w1 + self.b1, 0.0)          # ReLU
        return 1.0 / (1.0 + np.exp(-(hidden @ self.w2 + self.b2)))

    @classmethod
    def fit(cls, features: np.ndarray, labels: np.ndarray, hidden: int = 32, epochs: int = 300,
            rate: float = 0.01, l2: float = 1e-3, seed: int = 0) -> IconClassifier:
        """Train on examples (rows of features, labels 1 = player, 0 = not).

        Full-batch gradient descent with the Adam update rule, which adapts
        the step size per weight. `l2` keeps the weights small so the
        network doesn't just memorise the training crops.
        """
        rng = np.random.default_rng(seed)
        mean, scale = features.mean(axis=0), features.std(axis=0) + 1e-6
        z = (features - mean) / scale
        params = [rng.normal(0, np.sqrt(2 / z.shape[1]), (z.shape[1], hidden)), np.zeros(hidden),
                  rng.normal(0, np.sqrt(1 / hidden), hidden), np.zeros(1)]
        m1 = [np.zeros_like(p) for p in params]
        m2 = [np.zeros_like(p) for p in params]
        for step in range(1, epochs + 1):
            # Forward pass.
            h = np.maximum(z @ params[0] + params[1], 0.0)
            p = 1.0 / (1.0 + np.exp(-(h @ params[2] + params[3][0])))
            # Backward pass: gradients of the average log-loss.
            g = (p - labels) / len(labels)
            g_w2 = h.T @ g + l2 * params[2]
            g_h = np.outer(g, params[2]) * (h > 0)
            grads = [z.T @ g_h + l2 * params[0], g_h.sum(axis=0), g_w2, np.array([g.sum()])]
            # Adam update.
            for i, grad in enumerate(grads):
                m1[i] = 0.9 * m1[i] + 0.1 * grad
                m2[i] = 0.999 * m2[i] + 0.001 * grad ** 2
                params[i] -= rate * (m1[i] / (1 - 0.9 ** step)) / (np.sqrt(m2[i] / (1 - 0.999 ** step)) + 1e-8)
        return cls(mean, scale, params[0], params[1], params[2], params[3][0])

    def save(self, path: Path = DEFAULT_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, scale=self.scale, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)

    @classmethod
    def load(cls, path: Path = DEFAULT_PATH) -> IconClassifier | None:
        """The saved model, or None if it hasn't been trained yet."""
        if not Path(path).exists():
            return None
        d = np.load(path)
        return cls(d["mean"], d["scale"], d["w1"], d["b1"], d["w2"], d["b2"])
