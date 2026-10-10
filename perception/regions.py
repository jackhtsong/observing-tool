"""
regions.py: fixed screen regions for the broadcast HUD.

The broadcast layout never moves during a match, so instead of searching the
whole 1920x1080 frame for the minimap or the kill feed we cut out fixed
rectangles. These numbers were measured on the Masters London grand final
VOD (data/vods/ascent_test.mp4). A different broadcast or resolution will
need new numbers here and nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Region:
    """A rectangle on screen, in pixels, measured from the top-left corner."""

    x: int       # left edge
    y: int       # top edge
    width: int
    height: int

    def crop(self, frame: np.ndarray) -> np.ndarray:
        """Cut this rectangle out of a full frame.

        NumPy indexes images as [row, column], i.e. [y, x], which is why y
        comes first below. The result is a view into the original frame, so
        call .copy() on it if you plan to draw on it.
        """
        return frame[self.y : self.y + self.height, self.x : self.x + self.width]


# The minimap in the top-left corner (Ascent fits inside this box).
MINIMAP = Region(x=60, y=50, width=400, height=410)

# The kill feed in the top-right corner, where "killer [gun] victim" rows appear.
KILL_FEED = Region(x=1300, y=95, width=605, height=165)

# The middle of the top bar: shows the round timer, or a red spike icon
# once the spike is planted.
ROUND_TIMER = Region(x=925, y=25, width=70, height=70)
