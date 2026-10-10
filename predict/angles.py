"""
angles.py: pair players by who is holding an angle and who is about to walk into it.

The idea (from the observer's point of view)
-------------------------------------------
A player holding an angle is watching an area they can see, in the
direction they face. An enemy who is just behind the wall at the edge of
that area, close to stepping into it, is about to be in a fight with them.
So the pair to watch is the holder and the enemy nearest to entering the
held angle. Plain closeness counts for less.

How it is worked out, for each player A whose facing is known
-------------------------------------------------------------
1. A's *held angle*: every map spot (maps/map_model.py) that
     - A can see in a straight line (no wall in between),
     - lies within `half_angle` degrees either side of A's facing, and
     - is within `view_range` pixels.
2. For each enemy B: the walking distance from B to the nearest spot of
   A's held angle, the "entry distance". 0 means B is already in it.
3. Only the enemy with the smallest entry distance gets paired with A, and
   the pair gets a bonus of exp(-entry distance / entry_scale): 1 when B is
   already in the angle, about 0.37 at `entry_scale` pixels away.

Each pair's score is then

    score = w_close * exp(-straight distance / 60) + w_angle * bonus

where the bonus is the bigger of the two directions (red holding against
teal, or teal holding against red).

Status: on the 77 kills of the test clips this scores *below* plain closest
pair (see the README). The likely reasons: facing read from the minimap is
noisy, and the minimap's sightlines ignore smokes, boxes and height. It is
kept so it can be watched in the replay viewer and re-tested as those
inputs improve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from maps.map_model import MapModel
from predict.heuristic import PairScore, _enemy_pairs
from tracking.player_tracker import Track


@dataclass
class AngleSettings:
    """The hand-picked numbers. Distances in minimap pixels."""

    half_angle: float = 45.0     # degrees either side of the facing direction
    view_range: float = 150.0    # how far the held angle reaches
    entry_scale: float = 30.0    # bonus drops to ~0.37 at this entry distance
    w_close: float = 1.0         # weight of plain closeness
    w_angle: float = 1.0         # weight of the angle bonus


class AngleHoldPredictor:
    """Ranks red/teal pairs by held angles plus (a smaller share of) closeness."""

    def __init__(self, model: MapModel, settings: AngleSettings | None = None):
        self.model = model
        self.settings = settings or AngleSettings()
        # Pixel centre (x, y) of every map spot, used to measure angles and ranges.
        self._centres = (model.spots[:, ::-1] + 0.5) * model.cell

    def rank_pairs(self, tracks: list[Track], spike: tuple[float, float] | None = None) -> list[PairScore]:
        """Score every red/teal pair and return them best first. (`spike` is ignored here.)"""
        s = self.settings
        bonus = self.angle_bonuses(tracks)
        pairs = []
        for red, teal in _enemy_pairs(tracks):
            distance = float(np.hypot(red.x - teal.x, red.y - teal.y))
            score = s.w_close * np.exp(-distance / 60.0) + s.w_angle * bonus.get(_key(red, teal), 0.0)
            pairs.append(PairScore(red=red, teal=teal, score=float(score)))
        return sorted(pairs, key=lambda p: p.score, reverse=True)

    def angle_bonuses(self, tracks: list[Track]) -> dict[tuple[str, str], float]:
        """Steps 2-3: {(id, id): bonus} for each holder and their nearest-to-entering enemy."""
        bonus: dict[tuple[str, str], float] = {}
        for holder in tracks:
            held = self.held_angle(holder)
            enemies = [t for t in tracks if t.team != holder.team]
            if held is None or not enemies:
                continue
            entry = [self.entry_distance(enemy, held) for enemy in enemies]
            nearest = int(np.argmin(entry))
            value = float(np.exp(-entry[nearest] / self.settings.entry_scale))
            key = _key(holder, enemies[nearest])
            bonus[key] = max(bonus.get(key, 0.0), value)
        return bonus

    def held_angle(self, player: Track) -> np.ndarray | None:
        """Step 1: spot numbers inside the player's held angle, or None if facing is unknown."""
        if not (player.facing_x or player.facing_y):
            return None
        s = self.settings
        visible = self.model.sight_table[self.model._spot((player.x, player.y))]
        offset = self._centres - np.array([player.x, player.y])
        distance = np.hypot(offset[:, 0], offset[:, 1]) + 1e-6
        # cos(angle between facing and the direction to each spot)
        alignment = (offset[:, 0] * player.facing_x + offset[:, 1] * player.facing_y) / distance
        inside = visible & (alignment >= np.cos(np.radians(s.half_angle))) & (distance <= s.view_range)
        spots = np.nonzero(inside)[0]
        return spots if len(spots) else None

    def entry_distance(self, enemy: Track, held: np.ndarray) -> float:
        """Step 2: walking distance (px) from the enemy to the nearest spot of a held angle."""
        start = self.model._spot((enemy.x, enemy.y))
        steps = self.model.walk_table[start, held]
        steps = steps[steps >= 0]
        return float(steps.min()) * self.model.cell if len(steps) else float("inf")


def _key(a: Track, b: Track) -> tuple[str, str]:
    """Same key for a pair whichever player comes first."""
    return tuple(sorted((a.id, b.id)))
