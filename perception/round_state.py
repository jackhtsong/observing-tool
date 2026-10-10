"""
round_state.py: read what phase the round is in, from the broadcast HUD.

Phase 0 reads one thing: whether the spike is planted, and where.

  planted?   The top bar shows the round timer in white digits, until the
             spike is planted. Then the timer is replaced by a big red spike
             icon. So we count strongly red pixels in the timer box
             (regions.ROUND_TIMER): about 0% before the plant, about 30%
             after.

  where?     On the minimap the spike is a small yellow icon. Before the
             plant it travels with its carrier; once planted it stays put.
             We take the biggest yellow blob on the minimap.

Why it matters for predicting fights: after the plant, the defenders have
to come to the spike to defuse it, so the fights happen around it.
"""

from __future__ import annotations

import cv2
import numpy as np

from perception.regions import ROUND_TIMER

# Share of strongly red pixels in the timer box that means "spike icon".
PLANTED_RED_SHARE = 0.15
# Smallest yellow blob (pixels) accepted as the spike on the minimap.
MIN_SPIKE_PIXELS = 6


def spike_planted(frame: np.ndarray) -> bool:
    """True if the top bar shows the planted-spike icon instead of the timer."""
    hsv = cv2.cvtColor(ROUND_TIMER.crop(frame), cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = ((hue >= 165) | (hue <= 8)) & (sat > 120) & (val > 120)
    return bool(red.mean() > PLANTED_RED_SHARE)


def find_spike(minimap: np.ndarray) -> tuple[float, float] | None:
    """Centre (x, y) of the spike icon on the minimap, or None if not seen.

    Uses the same yellow as the dot detector uses to ignore the spike.
    """
    hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
    yellow = ((hsv[..., 0] >= 20) & (hsv[..., 0] <= 35) & (hsv[..., 1] >= 120) & (hsv[..., 2] >= 150)).astype(np.uint8)
    count, _, stats, centres = cv2.connectedComponentsWithStats(yellow)
    if count <= 1:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[biggest, cv2.CC_STAT_AREA] < MIN_SPIKE_PIXELS:
        return None
    x, y = centres[biggest]
    return float(x), float(y)


class SpikeTracker:
    """Follows the spike through one round: not planted, then planted at a fixed spot.

    The single-frame readings above flicker (the spike icon pulses near the
    end of the timer, and stray yellow appears on the minimap), but a plant
    can't be undone within a round. So:
      - the spike counts as planted once at least `confirm` of the last
        `window` frames said "planted", and then stays planted until reset();
      - its spot is the median of the minimap readings from the first
        `locate_for` seconds after the plant, which ignores stray readings.
    Call reset() at the start of every round.
    """

    def __init__(self, window: int = 5, confirm: int = 3, locate_for: float = 2.0):
        self.window, self.confirm, self.locate_for = window, confirm, locate_for
        self.reset()

    def reset(self) -> None:
        """Forget everything, e.g. at the start of a new round."""
        self.planted_at: float | None = None
        self.spot: tuple[float, float] | None = None
        self._recent: list[bool] = []
        self._spot_readings: list[tuple[float, float]] = []

    @property
    def planted(self) -> bool:
        return self.planted_at is not None

    def update(self, time: float, frame: np.ndarray, minimap: np.ndarray) -> None:
        """Feed one live frame (full broadcast frame and its minimap crop)."""
        if not self.planted:
            self._recent = (self._recent + [spike_planted(frame)])[-self.window:]
            if sum(self._recent) >= self.confirm:
                self.planted_at = time

        if self.planted and time - self.planted_at <= self.locate_for:
            reading = find_spike(minimap)
            if reading is not None:
                self._spot_readings.append(reading)
                xs, ys = zip(*self._spot_readings)
                self.spot = (float(np.median(xs)), float(np.median(ys)))
