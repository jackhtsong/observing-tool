"""
track_log.py: load saved tracks and deaths back from CSV.

scripts/track_players.py takes about a minute to run because it decodes
the video. Its outputs, tracks.csv and deaths.csv, hold everything the
predictor needs, so the evaluation and the replay viewer read those files
instead of re-running the video.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass

import pandas as pd

from tracking.player_tracker import Death, Track


@dataclass
class TrackLog:
    """All tracked frames of a video, looked up by time."""

    times: list[float]                 # every frame time, in order
    frames: dict[float, list[Track]]   # time -> the tracks at that time

    def tracks_at(self, time: float, tolerance: float = 0.15) -> list[Track] | None:
        """The tracks of the frame nearest to `time`.

        Returns None if no frame is within `tolerance` seconds, which
        happens during replays and between rounds (those frames were never
        tracked).
        """
        if not self.times:
            return None
        i = bisect_left(self.times, time)
        nearby = [self.times[j] for j in (i - 1, i) if 0 <= j < len(self.times)]
        nearest = min(nearby, key=lambda t: abs(t - time))
        return self.frames[nearest] if abs(nearest - time) <= tolerance else None


def load_tracks(path: str) -> TrackLog:
    """Read tracks.csv into a TrackLog."""
    table = pd.read_csv(path)
    frames: dict[float, list[Track]] = {}
    for time, rows in table.groupby("time"):
        frames[float(time)] = [
            Track(id=r.track, team=r.team, x=r.x, y=r.y, vx=r.vx, vy=r.vy,
                  last_seen=float(time) if r.seen else float("nan"), confirmed=True)
            for r in rows.itertuples()
        ]
    return TrackLog(times=sorted(frames), frames=frames)


def load_deaths(path: str) -> list[Death]:
    """Read deaths.csv into a list of Death, in time order."""
    table = pd.read_csv(path).sort_values("time")
    return [Death(time=r.time, track_id=r.track, team=r.team, x=r.x, y=r.y) for r in table.itertuples()]
