"""
map_background.py: build an "empty" picture of the minimap from the VOD.

Why we need it
--------------
Player icons move around, but the map itself (floor, walls, the "A" and "B"
letters, small decals) stays still. If we take many minimap crops spread
across the whole video and, for every pixel, keep the *median* colour, the
moving icons disappear and only the static map is left. We call that the
background.

The detector uses the background in two ways:
  1. To tell an icon apart from a map decal: an icon looks different from
     the background at that spot, a decal does not.
  2. To get a "floor mask": which pixels are walkable floor. The minimap is
     partly see-through, so outside the floor we see the game camera behind
     it, which changes all the time and must be ignored.

The background is slow to build (it decodes ~80 frames), so it is saved to
data/cache/ and reused on later runs.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from capture.video_source import VideoSource
from perception.regions import Region


def build_background(video: VideoSource, region: Region, samples: int = 80) -> np.ndarray:
    """Return the median minimap image over `samples` evenly spaced frames.

    More samples give a cleaner result but take longer. 80 frames over a
    5-minute clip means one every ~4 s, which is plenty for icons to have
    moved away from any given spot most of the time.
    """
    times = np.linspace(0, video.duration - 1, samples)
    crops = []
    for t in times:
        frame = video.frame_at(float(t))
        if frame is not None:
            crops.append(region.crop(frame))

    # Stack into one array of shape (samples, height, width, 3) and take the
    # median along the first axis, i.e. per pixel and per colour channel.
    stacked = np.stack(crops)
    return np.median(stacked, axis=0).astype(np.uint8)


def floor_mask(background: np.ndarray) -> np.ndarray:
    """Return a mask of the walkable floor: 255 on floor, 0 elsewhere.

    The floor is drawn in flat grey (low colour saturation, medium-bright),
    except the two bomb sites, which are a pale yellow-green. Anything else
    in the background is the see-through game camera.
    """
    hsv = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # OpenCV hue runs 0-179 (half of the usual 0-360 degrees).
    grey_floor = (sat < 45) & (val > 100)
    site_floor = (hue >= 20) & (hue <= 40) & (val > 120)  # yellow-green sites
    mask = (grey_floor | site_floor).astype(np.uint8) * 255

    # "Closing" fills tiny gaps, such as thin walls drawn inside the floor.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # The map is one connected shape. Keep only the largest blob so stray grey
    # patches in the camera view behind the minimap are thrown away.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count <= 1:
        return mask  # nothing found; return as is rather than crash
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))  # label 0 is "not floor"
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def load_or_build_background(
    video: VideoSource, region: Region, cache_dir: str = "data/cache"
) -> np.ndarray:
    """Load the cached background for this video, or build and cache it.

    The cache file is named after the video, e.g.
    data/cache/ascent_test_minimap_background.png. Delete it to force a
    rebuild (for example after changing the minimap region).
    """
    cache_path = Path(cache_dir) / f"{Path(video.path).stem}_minimap_background.png"
    if cache_path.exists():
        return cv2.imread(str(cache_path))

    background = build_background(video, region)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(cache_path), background)
    return background
