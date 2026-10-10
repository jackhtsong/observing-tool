"""
map_model.py: a model of the map built from the minimap picture, that you can fix by hand.

What the model knows
--------------------
  walkable   where players can stand and walk (the grey floor)
  walls      what blocks sight: everything off the floor, plus
             the thin white wall lines drawn *inside* the floor (inner walls,
             boxes, pillars)

From those two pictures it precomputes, for a coarse grid of spots on the
map (one spot every `cell` pixels):
  - the walking distance between every pair of spots, going around walls
  - whether every pair of spots can see each other in a straight line

After that, "how far is A from B?" and "can A see B?" are a table lookup,
fast enough to ask for every pair of players many times a second.

Fixing it by hand
-----------------
The two pictures are saved as plain black-and-white PNGs in maps/<map>/
(white = walkable / wall). They are a first guess made from the minimap. If
the model gets something wrong (a door drawn as a wall, a box missing), open
the PNG in any image editor, paint the fix in black or white, and save. The
tables are rebuilt automatically the next time the model loads, because
their cache is keyed on the pictures' content.

Limits of a minimap model: no height (high ground, stairs), no smokes or
other utility, and boxes only as far as the minimap draws them.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

CACHE_DIR = Path("data/cache")

# Bump this whenever the way the tables are built changes, so old cached
# tables are not reused by mistake.
TABLES_VERSION = "4"


def extract_walls_and_floor(background: np.ndarray, floor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First guess at the walkable and wall pictures from an empty minimap.

    background: empty minimap (map_background.build_background), BGR.
    floor:      floor mask (map_background.floor_mask), 255 = floor.

    The thin wall lines are brighter than the floor right around them. A
    "top-hat" filter keeps exactly that: each pixel minus the darkest-ish
    level of its 7x7 neighbourhood, so thin bright lines stand out and flat
    floor becomes ~0.

    Returns (walkable, walls) as 0/255 pictures the size of the minimap.
    """
    grey = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    tophat = cv2.morphologyEx(grey, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    near_floor = cv2.dilate(floor, np.ones((5, 5), np.uint8)) > 0
    # Wall lines are white-ish. Brightly coloured lines are ability marks
    # that stayed in one place long enough to end up in the empty-map
    # picture (e.g. a Viper wall), not walls.
    # The floor mask also left them out (they aren't grey), so thin coloured
    # strokes lying inside the map are put back as floor.
    coloured = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)[..., 1] >= 70
    lines = (tophat > 15) & near_floor & ~coloured
    inside_map = cv2.morphologyEx(floor, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)) > 0
    floor_ok = (floor > 0) | (coloured & inside_map)

    walls = (lines | ~floor_ok).astype(np.uint8) * 255
    walkable = (floor_ok & ~lines).astype(np.uint8) * 255
    return walkable, walls


class MapModel:
    """Walking distances and sightlines between spots on one map."""

    def __init__(self, walkable: np.ndarray, walls: np.ndarray, cell: int = 6, use_cache: bool = True):
        """Build (or load from cache) the distance and sight tables.

        walkable, walls: 0/255 pictures in minimap pixels (white = yes).
        cell:            spacing of the grid of spots, in pixels. 6 px is
                         about a third of an icon; smaller is more exact
                         but the tables grow with the square of the spot count.
        """
        self.cell = cell
        self.walls = walls > 0
        h, w = walkable.shape
        self.grid_shape = ((h + cell - 1) // cell, (w + cell - 1) // cell)

        # A grid square is a spot if most of it is walkable floor.
        #
        # Note: the thin inner wall lines block *sight* (sight table) but not
        # *walking* here. Using them for walking too cut real doorways shut
        # (the minimap draws doors barely wider than the line gaps), which
        # turned short walks into long detours and made predictions worse.
        size = self.grid_shape[::-1]
        floor_share = cv2.resize((walkable > 0).astype(np.float32), size, interpolation=cv2.INTER_AREA)
        spot_grid = floor_share > 0.5
        # Keep only the biggest connected area, so every spot can reach
        # every other. Points that land on a stray island are snapped onto
        # the main area instead.
        count, labels, stats, _ = cv2.connectedComponentsWithStats(spot_grid.astype(np.uint8), connectivity=8)
        main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) if count > 1 else 0
        self.spot_grid = labels == main
        self.spots = np.argwhere(self.spot_grid)               # (row, col) of every spot
        # Lookup from grid square to spot number (-1 = not a spot).
        self.spot_index = np.full(self.grid_shape, -1, np.int32)
        self.spot_index[self.spot_grid] = np.arange(len(self.spots))

        # The cache file name is a fingerprint of everything the tables
        # depend on, so editing a picture or changing this code's rules
        # (bump TABLES_VERSION) automatically means a fresh build.
        fingerprint = (TABLES_VERSION.encode() + self.spot_grid.tobytes() + walls.tobytes() + bytes([cell]))
        key = hashlib.sha1(fingerprint).hexdigest()[:12]
        cache = CACHE_DIR / f"map_model_{key}.npz"
        if use_cache and cache.exists():
            tables = np.load(cache)
            self.walk_table, self.sight_table = tables["walk"], tables["sight"]
        else:
            self.walk_table = self._build_walk_table()
            self.sight_table = self._build_sight_table()
            if use_cache:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache, walk=self.walk_table, sight=self.sight_table)

    # ----- loading and saving the editable pictures -----

    @classmethod
    def load(cls, folder: str | Path, **kwargs) -> MapModel:
        """Load maps/<map>/walkable.png and walls.png and build the model."""
        folder = Path(folder)
        walkable = cv2.imread(str(folder / "walkable.png"), cv2.IMREAD_GRAYSCALE)
        walls = cv2.imread(str(folder / "walls.png"), cv2.IMREAD_GRAYSCALE)
        if walkable is None or walls is None:
            raise FileNotFoundError(f"{folder}/walkable.png or walls.png missing; run scripts.build_map_model")
        # Treat any light pixel as white, in case an editor smoothed the edges.
        return cls((walkable > 127).astype(np.uint8) * 255, (walls > 127).astype(np.uint8) * 255, **kwargs)

    # ----- questions about two points -----

    def walking_distance(self, a: tuple[float, float], b: tuple[float, float]) -> float:
        """Walking distance in minimap pixels between points a and b (x, y)."""
        steps = self.walk_table[self._spot(a), self._spot(b)]
        return float("inf") if steps < 0 else float(steps) * self.cell

    def line_of_sight(self, a: tuple[float, float], b: tuple[float, float]) -> bool:
        """True if a straight line from a to b crosses no wall."""
        return bool(self.sight_table[self._spot(a), self._spot(b)])

    def visible_area(self, a: tuple[float, float]) -> np.ndarray:
        """True/False grid of every spot visible from a (for pictures and checks)."""
        visible = np.zeros(self.grid_shape, bool)
        rows, cols = self.spots[self.sight_table[self._spot(a)]].T
        visible[rows, cols] = True
        return visible

    # ----- helpers -----

    def _spot(self, point: tuple[float, float]) -> int:
        """Spot number nearest to a pixel point (x, y).

        Icons can overhang walls, so a point off the walkable floor is
        moved to the nearest spot.
        """
        row = int(np.clip(point[1] // self.cell, 0, self.grid_shape[0] - 1))
        col = int(np.clip(point[0] // self.cell, 0, self.grid_shape[1] - 1))
        index = self.spot_index[row, col]
        if index >= 0:
            return int(index)
        gaps = np.abs(self.spots - (row, col)).sum(axis=1)
        return int(np.argmin(gaps))

    def _build_walk_table(self) -> np.ndarray:
        """Steps between every pair of spots, by breadth-first search from each spot.

        From one spot, `reached` grows one grid square in every direction
        per step (a 3x3 dilation), only into spots; each spot's distance is
        the step it was first reached. -1 = unreachable. Stored as int16 to
        keep the table small.
        """
        n = len(self.spots)
        table = np.full((n, n), -1, np.int16)
        walkable = self.spot_grid.astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        for i, (r, c) in enumerate(self.spots):
            steps = np.full(self.grid_shape, -1, np.int16)
            reached = np.zeros(self.grid_shape, np.uint8)
            reached[r, c] = 1
            steps[r, c] = 0
            step = 0
            while True:
                step += 1
                grown = cv2.dilate(reached, kernel) & walkable
                new = (grown > 0) & (reached == 0)
                if not new.any():
                    break
                steps[new] = step
                reached = grown
            table[i] = steps[self.spot_grid]
        return table

    def _build_sight_table(self) -> np.ndarray:
        """Whether every pair of spots can see each other.

        For each spot we draw straight lines to all other spots at once
        (NumPy arrays of sample points, one every ~1.5 px) and check whether
        any sample lands on a wall. The first and last few pixels of each
        line are skipped, since a spot's centre can sit right beside a wall.
        """
        n = len(self.spots)
        centres = (self.spots[:, ::-1] + 0.5) * self.cell   # (x, y) pixel centre of each spot
        h, w = self.walls.shape
        table = np.zeros((n, n), bool)
        fractions = np.linspace(0.0, 1.0, int(np.hypot(h, w) / 1.5))[None, :]   # positions along each line
        skip = 4.0  # px at each end not checked

        for i in range(n):
            start = centres[i]
            ends = centres[i + 1:]                       # only j > i; the table is symmetric
            if len(ends) == 0:
                break
            length = np.hypot(*(ends - start).T)[:, None]
            xs = start[0] + (ends[:, 0:1] - start[0]) * fractions
            ys = start[1] + (ends[:, 1:2] - start[1]) * fractions
            along = fractions * length
            checked = (along > skip) & (along < length - skip)
            hit = self.walls[ys.round().astype(int).clip(0, h - 1), xs.round().astype(int).clip(0, w - 1)]
            clear = ~(hit & checked).any(axis=1)
            table[i, i + 1:] = clear
            table[i + 1:, i] = clear
        np.fill_diagonal(table, True)
        return table


# Maps the project knows about. A video's map is read from the start of its
# file name, e.g. data/vods/breeze_prx_lev.mp4 -> "breeze". Videos whose name
# starts with none of these are treated as Ascent (the first test clips are
# called ascent_test*.mp4).
KNOWN_MAPS = ("ascent", "bind", "breeze", "fracture", "haven", "icebox", "lotus",
              "pearl", "split", "sunset", "abyss", "corrode")
DEFAULT_MAP = "ascent"


def map_name_for(video_path: str | Path) -> str:
    """Which map a video shows, from its file name (see KNOWN_MAPS)."""
    stem = Path(video_path).stem.lower()
    return next((name for name in KNOWN_MAPS if stem.startswith(name)), DEFAULT_MAP)


def load_map_for(video_path: str | Path) -> MapModel:
    """The map model for a video's map, from maps/<map>/.

    Raises a clear error naming the command to run if that map's pictures
    haven't been made yet.
    """
    name = map_name_for(video_path)
    folder = Path("maps") / name
    if not (folder / "walls.png").exists():
        raise SystemExit(
            f"No map model for {name} yet. Track a {name} video first, then run:\n"
            f"    .venv/bin/python -m scripts.build_map_model --map {name}")
    return MapModel.load(folder)
