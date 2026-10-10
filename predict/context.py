"""
context.py: closest pair, plus a pull towards the planted spike.

After the spike is planted, the defenders must reach it to defuse, and the
attackers guard it. So the fights of a post-plant round happen around the
spike. This predictor starts from plain closeness (the best signal so far)
and, once the spike is planted, adds a bonus to pairs near it:

    score = exp(-distance / 60) + w_spike * exp(-spike distance / spike_scale)

where "spike distance" is how far the nearer of the two players is from
the spike, in a straight line. Before the plant it ranks exactly like the
closest-pair baseline.

The spike's spot comes from round_state.csv (perception/round_state.py);
callers pass it to rank_pairs() as `spike`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from predict.heuristic import PairScore, _enemy_pairs
from tracking.player_tracker import Track


@dataclass
class ContextSettings:
    w_spike: float = 0.5        # how much being near the planted spike counts
    spike_scale: float = 60.0   # bonus drops to ~0.37 this many px from the spike


class SpikeContextPredictor:
    """Closest pair, with pairs near the planted spike moved up."""

    def __init__(self, settings: ContextSettings | None = None):
        self.settings = settings or ContextSettings()

    def rank_pairs(self, tracks: list[Track], spike: tuple[float, float] | None = None) -> list[PairScore]:
        """Score every red/teal pair and return them best first.

        spike: (x, y) of the planted spike on the minimap, or None if it is
        not planted (then this is the same as closest pair).
        """
        s = self.settings
        pairs = []
        for red, teal in _enemy_pairs(tracks):
            score = float(np.exp(-np.hypot(red.x - teal.x, red.y - teal.y) / 60.0))
            if spike is not None:
                nearer = min(np.hypot(p.x - spike[0], p.y - spike[1]) for p in (red, teal))
                score += s.w_spike * float(np.exp(-nearer / s.spike_scale))
            pairs.append(PairScore(red=red, teal=teal, score=score))
        return sorted(pairs, key=lambda p: p.score, reverse=True)
