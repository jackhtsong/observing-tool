"""
heuristic.py: score every red/teal pair for how likely they are to fight soon.

The idea
--------
Most fights are two enemies getting close along a path they can walk, while
moving towards each other, with a sightline between them. So for every pair
of one red and one teal player we measure three things:

  walking distance   how far apart they are along walkable paths (not through
                     walls). Closer = more likely to fight. Turned into a
                     0..1 "nearness" with exp(-distance / distance_scale):
                     1 when on top of each other, about 0.37 at
                     distance_scale pixels apart, near 0 when far.

  closing speed      how fast that walking distance is shrinking. We move
                     both players forward along their current speed for
                     `look_ahead` seconds and measure the walking distance
                     again. Positive = getting closer. Scaled to -1..1.

  line of sight      whether a straight line between them stays on the
                     walls (see maps/map_model.py). 1 or 0.

  score = w_near * nearness + w_closing * closing + w_sight * line_of_sight

The weights are hand-picked (that's what "heuristic" means here). Phase 2
replaces this with a model trained on many recorded fights.

The baseline
------------
ClosestPairBaseline ranks pairs by plain straight-line distance and nothing
else. The heuristic is only worth keeping if it beats this.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from maps.map_model import MapModel
from perception.teams import RED, TEAL
from tracking.player_tracker import Track


@dataclass
class PairScore:
    """How likely one red and one teal player are to fight soon."""

    red: Track
    teal: Track
    score: float                    # higher = more likely; only the order matters
    walking_distance: float = 0.0   # px along walkable paths
    closing_speed: float = 0.0      # px/s; positive = getting closer
    line_of_sight: bool = False

    def involves(self, track_id: str) -> bool:
        """True if this pair includes the track with this id."""
        return track_id in (self.red.id, self.teal.id)


@dataclass
class HeuristicSettings:
    """The hand-tuned numbers. Distances in minimap px, speeds in px/s."""

    distance_scale: float = 60.0   # nearness drops to ~0.37 at this walking distance
    closing_scale: float = 20.0    # closing speed that counts as "fully closing" (1.0)
    look_ahead: float = 1.0        # seconds to move players forward to measure closing speed
    w_near: float = 1.0
    w_closing: float = 0.5
    w_sight: float = 0.3


class HeuristicPredictor:
    """Ranks red/teal pairs using walking distance, closing speed and line of sight."""

    def __init__(self, geometry: MapModel, settings: HeuristicSettings | None = None):
        self.geometry = geometry
        self.settings = settings or HeuristicSettings()

    def rank_pairs(self, tracks: list[Track], spike: tuple[float, float] | None = None) -> list[PairScore]:
        """Score every red/teal pair and return them best first. (`spike` is ignored here.)"""
        pairs = [self.score_pair(r, t) for r, t in _enemy_pairs(tracks)]
        return sorted(pairs, key=lambda p: p.score, reverse=True)

    def score_pair(self, red: Track, teal: Track) -> PairScore:
        """Work out the three measurements for one pair and combine them."""
        s = self.settings
        g = self.geometry

        now = g.walking_distance((red.x, red.y), (teal.x, teal.y))

        # Where both will be after `look_ahead` seconds if they keep going.
        red_later = (red.x + red.vx * s.look_ahead, red.y + red.vy * s.look_ahead)
        teal_later = (teal.x + teal.vx * s.look_ahead, teal.y + teal.vy * s.look_ahead)
        later = g.walking_distance(red_later, teal_later)
        closing_speed = (now - later) / s.look_ahead

        sight = g.line_of_sight((red.x, red.y), (teal.x, teal.y))

        nearness = float(np.exp(-now / s.distance_scale))
        closing = float(np.clip(closing_speed / s.closing_scale, -1.0, 1.0))
        score = s.w_near * nearness + s.w_closing * closing + s.w_sight * float(sight)

        return PairScore(red=red, teal=teal, score=score, walking_distance=now,
                         closing_speed=closing_speed, line_of_sight=sight)


class ClosestPairBaseline:
    """The naive baseline: the closer two enemies are in a straight line, the likelier the fight."""

    def rank_pairs(self, tracks: list[Track], spike: tuple[float, float] | None = None) -> list[PairScore]:
        """Score every red/teal pair by minus their straight-line distance, best first.

        `spike` (the planted spike's spot) is accepted so every predictor can
        be called the same way, but ignored here.
        """
        pairs = []
        for red, teal in _enemy_pairs(tracks):
            distance = float(np.hypot(red.x - teal.x, red.y - teal.y))
            pairs.append(PairScore(red=red, teal=teal, score=-distance))
        return sorted(pairs, key=lambda p: p.score, reverse=True)


def _enemy_pairs(tracks: list[Track]) -> list[tuple[Track, Track]]:
    """Every (red, teal) combination of the given tracks."""
    reds = [t for t in tracks if t.team == RED]
    teals = [t for t in tracks if t.team == TEAL]
    return [(r, t) for r in reds for t in teals]
