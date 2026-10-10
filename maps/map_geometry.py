"""
map_geometry.py: how far apart two players really are, and whether they can see each other.

Why straight-line distance is not enough
----------------------------------------
Two players on opposite sides of a wall can be 20 px apart on the minimap
but a long walk from each other, and they can't shoot through the wall.
Fights happen between players who are close *along walkable paths* and who
have, or are about to have, line of sight. This file answers both
questions using the floor mask from map_background.py (white = walkable).

Walking distance
----------------
The minimap is shrunk to a coarse grid of `cell` x `cell` pixel squares.
A square is walkable if most of its pixels are floor. From a player's
square we spread outwards one square at a time, only through walkable
squares, like water filling the map ("breadth-first search"). The number of
steps it takes to reach the other player, times the square size, is the
walking distance in minimap pixels. Diagonal steps count the same as
straight ones, so it slightly underestimates diagonal walks. That is fine
for scoring.

Each spread-out ("distance field") is remembered, so asking about the same
starting square again is instant.

Line of sight
-------------
We sample points every 2 px along the straight line between the two
players. If almost all of them are on the floor, the line doesn't cross
the empty space between corridors, and we say they can see each other.
This ignores boxes and other cover drawn inside the floor, so it is an
optimistic guess.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np


class MapGeometry:
    """Walking distance and line of sight on one map's floor mask."""

    def __init__(self, floor: np.ndarray, cell: int = 4, los_floor_share: float = 0.95):
        """Prepare the coarse walkable grid.

        floor:            floor mask in minimap pixels (255 = walkable).
        cell:             grid square size in pixels. Smaller is more exact but slower.
        los_floor_share:  share of points on the line that must be floor for
                          line of sight.
        """
        self.cell = cell
        self.los_floor_share = los_floor_share
        self.floor = floor > 0
        h, w = floor.shape

        # Shrink: each grid square becomes the share of floor pixels it holds
        # (INTER_AREA averages), and counts as walkable if that is over half.
        grid_w, grid_h = (w + cell - 1) // cell, (h + cell - 1) // cell
        share = cv2.resize(self.floor.astype(np.float32), (grid_w, grid_h), interpolation=cv2.INTER_AREA)
        self.walkable = share > 0.5

        # Every walkable square's (row, col), used to snap positions that
        # land just off the floor (icons can overhang walls) onto the floor.
        self._walkable_cells = np.argwhere(self.walkable)

    def walking_distance(self, a: tuple[float, float], b: tuple[float, float]) -> float:
        """Walking distance in minimap pixels between points a and b (x, y).

        Returns infinity if b can't be reached from a (shouldn't happen on a
        connected map, but a mis-detected dot could sit outside it).
        """
        start, goal = self._to_cell(a), self._to_cell(b)
        steps = self._distance_field(start)[goal]
        return float("inf") if steps < 0 else float(steps * self.cell)

    def line_of_sight(self, a: tuple[float, float], b: tuple[float, float]) -> bool:
        """True if the straight line from a to b stays (almost) on the floor."""
        length = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        samples = max(int(length / 2), 2)
        xs = np.linspace(a[0], b[0], samples).round().astype(int)
        ys = np.linspace(a[1], b[1], samples).round().astype(int)
        h, w = self.floor.shape
        xs, ys = xs.clip(0, w - 1), ys.clip(0, h - 1)
        return bool(self.floor[ys, xs].mean() >= self.los_floor_share)

    # ----- helpers -----

    def _to_cell(self, point: tuple[float, float]) -> tuple[int, int]:
        """Grid square (row, col) for a pixel point (x, y), snapped onto the floor."""
        row = int(np.clip(point[1] // self.cell, 0, self.walkable.shape[0] - 1))
        col = int(np.clip(point[0] // self.cell, 0, self.walkable.shape[1] - 1))
        if self.walkable[row, col]:
            return row, col
        # Not walkable: use the nearest walkable square instead.
        gaps = np.abs(self._walkable_cells - (row, col)).sum(axis=1)
        nearest = self._walkable_cells[int(np.argmin(gaps))]
        return int(nearest[0]), int(nearest[1])

    @lru_cache(maxsize=4096)
    def _distance_field(self, start: tuple[int, int]) -> np.ndarray:
        """Steps from `start` to every grid square (-1 = unreachable).

        Breadth-first search done with whole-image operations: `reached`
        grows by one square in every direction (a 3x3 dilation) per step,
        but only into walkable squares. Each square's distance is the step
        at which it was first reached.
        """
        steps = np.full(self.walkable.shape, -1, np.int32)
        reached = np.zeros(self.walkable.shape, np.uint8)
        reached[start] = 1
        steps[start] = 0
        walkable = self.walkable.astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)

        step = 0
        while True:
            step += 1
            grown = cv2.dilate(reached, kernel) & walkable
            new = (grown > 0) & (reached == 0)
            if not new.any():
                return steps
            steps[new] = step
            reached = grown
