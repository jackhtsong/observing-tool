"""
probability.py: the chance that a pair of players is in a kill within the next few seconds.

The other predictors give each pair a score that is only good for ranking:
"this pair is likelier than that one". To say "only suggest a pair when a
kill is more likely than not", we need a real probability: of all the
moments the model says 70%, a kill should follow about 70% of the time.

What exactly is predicted
-------------------------
For a red player R and a teal player T at time t:

    P( R or T dies within the next `horizon` seconds )

"R or T dies" is used rather than "R kills T" because the kill feed tells
us who died but not reliably which tracked player killed them. If either
of the two dies, an observer watching that pair sees the fight.

How it works
------------
Each pair is described by a few numbers (features), all from things we
already measure:

    nearness        exp(-straight-line distance / 60)
    walk_nearness   exp(-walking distance / 60)        (map model)
    closing         how fast the straight-line gap shrinks, scaled to -1..1
    angle_bonus     held-angle bonus (predict/angles.py): one of them is
                    watching an angle the other is about to walk into
    sight           1 if they can see each other now (map model)
    spike_near      exp(-distance of the nearer one to the planted spike / 60),
                    0 if not planted
    planted         1 if the spike is planted
    few_pairs       1 / number of red-teal pairs (late in a round, with
                    few players left, every remaining pair is likelier)

A logistic regression turns them into a probability:

    p = 1 / (1 + exp(-(b + w1*x1 + w2*x2 + ...)))

The weights are learned from recorded matches by scripts/train_fight_model.py
and saved as JSON (predict/models/fight_within_3s.json), so the live tool
just loads them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from maps.map_model import MapModel
from predict.angles import AngleHoldPredictor, _key
from predict.heuristic import PairScore, _enemy_pairs
from tracking.player_tracker import Track

FEATURES = ["nearness", "walk_nearness", "closing", "angle_bonus", "sight", "spike_near", "planted", "few_pairs"]
DEFAULT_MODEL = Path(__file__).parent / "models" / "fight_within_3s.json"


def pair_features(tracks: list[Track], model: MapModel,
                  spike: tuple[float, float] | None) -> tuple[list[tuple[Track, Track]], np.ndarray]:
    """The features (see the top of this file) of every red/teal pair.

    Returns (pairs, table) where table has one row per pair and one column
    per name in FEATURES.
    """
    pairs = _enemy_pairs(tracks)
    if not pairs:
        return [], np.zeros((0, len(FEATURES)))
    bonus = AngleHoldPredictor(model).angle_bonuses(tracks)
    rows = []
    for red, teal in pairs:
        straight = float(np.hypot(red.x - teal.x, red.y - teal.y))
        walk = model.walking_distance((red.x, red.y), (teal.x, teal.y))
        later = float(np.hypot(red.x + red.vx - teal.x - teal.vx, red.y + red.vy - teal.y - teal.vy))
        closing = float(np.clip((straight - later) / 20.0, -1.0, 1.0))   # gap change over 1 s
        if spike is not None:
            nearer = min(np.hypot(p.x - spike[0], p.y - spike[1]) for p in (red, teal))
            spike_near = float(np.exp(-nearer / 60.0))
        else:
            spike_near = 0.0
        rows.append([
            np.exp(-straight / 60.0),
            np.exp(-walk / 60.0) if np.isfinite(walk) else 0.0,
            closing,
            bonus.get(_key(red, teal), 0.0),
            float(model.line_of_sight((red.x, red.y), (teal.x, teal.y))),
            spike_near,
            float(spike is not None),
            1.0 / len(pairs),
        ])
    return pairs, np.array(rows, dtype=np.float64)


@dataclass
class LogisticModel:
    """A logistic regression on standardised features."""

    weights: np.ndarray     # one per feature
    bias: float
    mean: np.ndarray        # feature means, used to standardise
    scale: np.ndarray       # feature standard deviations

    def predict(self, table: np.ndarray) -> np.ndarray:
        """Probability for each row of a feature table."""
        z = (table - self.mean) / self.scale
        return 1.0 / (1.0 + np.exp(-(z @ self.weights + self.bias)))

    @classmethod
    def fit(cls, table: np.ndarray, labels: np.ndarray, l2: float = 1.0, steps: int = 50) -> LogisticModel:
        """Learn the weights from examples (rows of features, 0/1 labels).

        Uses Newton's method, the standard way to fit a logistic
        regression: each step solves a small linear system and moves
        straight to a better set of weights. `l2` gently pulls weights
        towards 0, which keeps them sensible when features overlap.
        """
        mean = table.mean(axis=0)
        scale = table.std(axis=0)
        scale[scale < 1e-9] = 1.0                       # a constant column: leave it unscaled
        x = np.hstack([(table - mean) / scale, np.ones((len(table), 1))])   # last column = bias
        w = np.zeros(x.shape[1])
        penalty = np.eye(x.shape[1]) * l2
        penalty[-1, -1] = 0.0                           # don't pull the bias towards 0
        for _ in range(steps):
            p = 1.0 / (1.0 + np.exp(-(x @ w)))
            gradient = x.T @ (p - labels) + penalty @ w
            hessian = (x * (p * (1 - p))[:, None]).T @ x + penalty
            step = np.linalg.solve(hessian, gradient)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        return cls(weights=w[:-1], bias=float(w[-1]), mean=mean, scale=scale)

    def save(self, path: Path, info: dict | None = None) -> None:
        """Write the model (and any notes in `info`) to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"features": FEATURES, "weights": self.weights.tolist(), "bias": self.bias,
                "mean": self.mean.tolist(), "scale": self.scale.tolist(), "info": info or {}}
        path.write_text(json.dumps(data, indent=1))

    @classmethod
    def load(cls, path: Path) -> LogisticModel:
        data = json.loads(Path(path).read_text())
        if data["features"] != FEATURES:
            raise ValueError(f"{path} was trained on different features; retrain with scripts.train_fight_model")
        return cls(weights=np.array(data["weights"]), bias=data["bias"],
                   mean=np.array(data["mean"]), scale=np.array(data["scale"]))


class FightProbabilityPredictor:
    """Ranks pairs by their probability of a kill within the horizon.

    Same rank_pairs() as the other predictors, but each PairScore.score is a
    real probability (0..1), so a threshold like "over 50%" means something.
    """

    def __init__(self, map_model: MapModel, model: LogisticModel | None = None, path: Path = DEFAULT_MODEL):
        self.map_model = map_model
        self.model = model or LogisticModel.load(path)

    def rank_pairs(self, tracks: list[Track], spike: tuple[float, float] | None = None) -> list[PairScore]:
        pairs, table = pair_features(tracks, self.map_model, spike)
        if not pairs:
            return []
        probabilities = self.model.predict(table)
        scored = [PairScore(red=r, teal=t, score=float(p)) for (r, t), p in zip(pairs, probabilities)]
        return sorted(scored, key=lambda s: s.score, reverse=True)
