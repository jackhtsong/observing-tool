"""
build_map_model.py: make the editable map pictures for a map, and a preview to check them.

Takes the empty-minimap picture of a tracked video, extracts the walkable
floor and the walls (maps/map_model.py), and saves them as
maps/<map>/walkable.png and maps/<map>/walls.png. These two files are meant
to be committed and hand-corrected over time, so this script never
overwrites them unless you pass --force.

It then builds the model (walking distances and sightlines between all
spots, cached in data/cache/) and saves a preview picture,
data/out/map_model/preview.png, with three panels:
  1. the minimap
  2. what the model thinks: grey = walkable, white = walls
  3. what can be seen from a few sample spots, one colour per spot

Run it from the project root:

    .venv/bin/python -m scripts.build_map_model                       # Ascent, from the first tracked Ascent video
    .venv/bin/python -m scripts.build_map_model --map breeze          # another map (video name must start with "breeze")
    .venv/bin/python -m scripts.build_map_model --force               # regenerate the pictures (loses hand edits!)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from maps.map_model import MapModel, extract_walls_and_floor, map_name_for
from perception.map_background import floor_mask

# Sample spots for the visibility preview: (label, x, y) in minimap pixels.
# These are Ascent's sites; on other maps they are just three sample spots.
PREVIEW_SPOTS = [("A site", 330, 125), ("B site", 60, 110), ("mid", 190, 200)]
PREVIEW_COLOURS = [(0, 200, 255), (255, 120, 0), (120, 255, 120)]   # BGR


def main() -> None:
    p = argparse.ArgumentParser(description="Build the editable map model pictures and a preview.")
    p.add_argument("--map", default="ascent", help="map name; pictures go to maps/<map>/")
    p.add_argument("--background", default=None,
                   help="empty minimap picture to start from (default: first data/out/tracks/*/background.png)")
    p.add_argument("--force", action="store_true", help="overwrite existing walkable.png / walls.png")
    args = p.parse_args()

    folder = Path("maps") / args.map
    background_path = args.background or default_background(args.map)
    background = cv2.imread(background_path)

    if (folder / "walls.png").exists() and not args.force:
        print(f"{folder}/walkable.png and walls.png already exist (keeping your edits; --force to redo)")
    else:
        walkable, walls = extract_walls_and_floor(background, floor_mask(background))
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / "walkable.png"), walkable)
        cv2.imwrite(str(folder / "walls.png"), walls)
        print(f"Wrote {folder}/walkable.png and walls.png from {background_path}")

    started = time.time()
    model = MapModel.load(folder)
    print(f"Model ready: {len(model.spots)} spots, built or loaded in {time.time() - started:.1f}s")

    out = Path("data/out/map_model")
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "preview.png"), preview(background, model))
    print(f"Preview: {out}/preview.png")


def default_background(map_name: str) -> str:
    """The empty-minimap picture of the first tracked video of this map."""
    for path in sorted(Path("data/out/tracks").glob("*/background.png")):
        if map_name_for(path.parent.name) == map_name:
            return str(path)
    raise SystemExit(f"No tracked {map_name} video yet; run scripts.process_vods first "
                     f"(the video's file name must start with '{map_name}').")


def preview(background: np.ndarray, model: MapModel, scale: int = 2) -> np.ndarray:
    """Three panels side by side: minimap, walkable/walls, and visibility from sample spots."""
    h, w = model.walls.shape
    big = lambda img: cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

    # Panel 2: walkable grey, walls white, everything else black.
    layout = np.zeros((h, w, 3), np.uint8)
    walkable_pixels = cv2.resize(model.spot_grid.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    layout[walkable_pixels] = (90, 90, 90)
    # Walls are drawn only next to the floor; the empty space outside the
    # map is a "wall" too, but painting it all white would hide the shape.
    near_floor = cv2.dilate(walkable_pixels.astype(np.uint8), np.ones((2 * model.cell + 1,) * 2, np.uint8)) > 0
    layout[model.walls & near_floor] = (255, 255, 255)

    # Panel 3: tint what each sample spot can see.
    sight = (background * 0.5).astype(np.uint8)
    for (label, x, y), colour in zip(PREVIEW_SPOTS, PREVIEW_COLOURS):
        visible = cv2.resize(model.visible_area((x, y)).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        sight[visible] = (0.5 * sight[visible] + 0.5 * np.array(colour)).astype(np.uint8)
    sight = big(sight)
    for (label, x, y), colour in zip(PREVIEW_SPOTS, PREVIEW_COLOURS):
        cv2.circle(sight, (x * scale, y * scale), 7, colour, -1)
        cv2.circle(sight, (x * scale, y * scale), 8, (0, 0, 0), 1)
        cv2.putText(sight, label, (x * scale + 10, y * scale - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)

    return np.hstack([big(background), big(layout), sight])


if __name__ == "__main__":
    main()
