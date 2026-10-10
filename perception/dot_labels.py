"""
dot_labels.py: hand-made answers for "where are the players?", and grading the detector against them.

A person clicks every player on a set of minimap frames (scripts/label_dots.py).
Those clicks are the truth we grade the dot detector against
(scripts/score_dots.py). Without them we can only judge the detector by eye.

The labels live in one JSON file, data/labels/dots.json, shaped like:

    {"frames": [
        {"video": "data/vods/ascent_test.mp4", "time": 125.0, "done": true,
         "players": [{"x": 82.5, "y": 171.0, "team": "teal"}, ...]},
        ...
    ]}

x and y are minimap pixels, the same coordinates the detector uses.
"done" marks frames the person has finished, so an empty "players" list on a
done frame means "nobody is alive", not "not labelled yet".
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from perception.minimap_dots import Dot

LABELS_PATH = Path("data/labels/dots.json")


@dataclass
class LabelledPlayer:
    """One player a person clicked on."""

    x: float
    y: float
    team: str   # "red" or "teal"


@dataclass
class LabelledFrame:
    """One minimap frame and the players marked on it."""

    video: str
    time: float
    players: list[LabelledPlayer] = field(default_factory=list)
    done: bool = False


def load_labels(path: Path = LABELS_PATH) -> list[LabelledFrame]:
    """Read the labels file; returns an empty list if it doesn't exist yet."""
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return [
        LabelledFrame(video=f["video"], time=f["time"], done=f.get("done", False),
                      players=[LabelledPlayer(**p) for p in f["players"]])
        for f in data["frames"]
    ]


def save_labels(frames: list[LabelledFrame], path: Path = LABELS_PATH) -> None:
    """Write the labels file (creating data/labels/ if needed).

    Written to a temporary file first and then renamed, so a crash halfway
    through saving can't leave a half-written, unreadable file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"frames": [asdict(f) for f in frames]}, indent=1))
    temporary.replace(path)


@dataclass
class FrameScore:
    """How the detector did on one labelled frame."""

    found: int        # labelled players the detector found, with the right team
    missed: int       # labelled players with no matching detection
    extra: int        # detections that match no labelled player (false alarms)
    wrong_team: int   # detections on a labelled player but with the other team's colour


def score_frame(labels: list[LabelledPlayer], dots: list[Dot], max_distance: float = 6.0) -> FrameScore:
    """Compare one frame's detections with the labels.

    A detection matches a labelled player if their centres are within
    `max_distance` pixels (about one icon radius). Matching is one-to-one
    and closest pairs first, so two detections can't both claim the same
    player. A match with the wrong team counts as wrong_team, not found.
    """
    options = []
    for li, player in enumerate(labels):
        for di, dot in enumerate(dots):
            distance = float(np.hypot(player.x - dot.x, player.y - dot.y))
            if distance <= max_distance:
                options.append((distance, li, di))

    used_labels, used_dots = set(), set()
    found = wrong_team = 0
    for distance, li, di in sorted(options):
        if li in used_labels or di in used_dots:
            continue
        used_labels.add(li)
        used_dots.add(di)
        if labels[li].team == dots[di].team:
            found += 1
        else:
            wrong_team += 1

    return FrameScore(found=found,
                      missed=len(labels) - len(used_labels),
                      extra=len(dots) - len(used_dots),
                      wrong_team=wrong_team)
