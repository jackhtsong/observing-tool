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
    # Files made before facing was added have no fx / fy columns.
    if "fx" not in table:
        table["fx"] = 0.0
        table["fy"] = 0.0
    # ...and files made before player numbers (slots) have no slot column.
    if "slot" not in table:
        table["slot"] = 0
    frames: dict[float, list[Track]] = {}
    for time, rows in table.groupby("time"):
        frames[float(time)] = [
            Track(id=r.track, team=r.team, x=r.x, y=r.y, vx=r.vx, vy=r.vy,
                  last_seen=float(time) if r.seen else float("nan"), confirmed=True,
                  facing_x=r.fx, facing_y=r.fy, slot=int(r.slot))
            for r in rows.itertuples()
        ]
    return TrackLog(times=sorted(frames), frames=frames)


def load_deaths(path: str) -> list[Death]:
    """Read deaths.csv into a list of Death, in time order."""
    table = pd.read_csv(path).sort_values("time")
    return [Death(time=r.time, track_id=r.track, team=r.team, x=r.x, y=r.y) for r in table.itertuples()]


@dataclass
class RoundLog:
    """Spike state per tracked frame, looked up by time (from round_state.csv)."""

    times: list[float]
    planted: dict[float, bool]
    spot: dict[float, tuple[float, float] | None]

    def spike_at(self, time: float, tolerance: float = 0.15) -> tuple[float, float] | None:
        """Where the planted spike is at `time`, or None if not planted (or spot unknown)."""
        if not self.times:
            return None
        i = bisect_left(self.times, time)
        nearby = [self.times[j] for j in (i - 1, i) if 0 <= j < len(self.times)]
        nearest = min(nearby, key=lambda t: abs(t - time))
        if abs(nearest - time) > tolerance or not self.planted[nearest]:
            return None
        return self.spot[nearest]


def load_round_state(path: str) -> RoundLog:
    """Read round_state.csv into a RoundLog (empty if the file doesn't exist)."""
    try:
        table = pd.read_csv(path)
    except FileNotFoundError:
        return RoundLog(times=[], planted={}, spot={})
    planted, spot = {}, {}
    for r in table.itertuples():
        planted[float(r.time)] = bool(r.planted)
        spot[float(r.time)] = None if pd.isna(r.spike_x) else (float(r.spike_x), float(r.spike_y))
    return RoundLog(times=sorted(planted), planted=planted, spot=spot)
