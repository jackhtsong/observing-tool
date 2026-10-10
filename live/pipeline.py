"""
pipeline.py: everything the tool does to one frame, in one place.

Each call to LivePipeline.step(time, frame) runs, in order:
  1. layout check   is this the normal live minimap? (not a replay / round break)
  2. kill feed      any new kills? (read even during replays, so their old
                    rows are known, but only recorded while live)
  3. round change   if the minimap has been away for over a second, it's a
                    new round: reset the tracker and the spike
  4. spike          planted yet, and where
  5. dots           every player icon on the minimap, with team and facing
  6. tracker        dots -> one smoothed track per player, deaths matched
  7. predictor      every red/teal pair ranked, best first
and returns a FrameResult with the outcome and how long each part took.

Both the offline tracking script (scripts/track_players.py) and the live
runner (scripts/run_live.py) use this, so they always behave the same.
"""

from __future__ import annotations

import time as clock
from dataclasses import dataclass, field

import numpy as np

from perception.kill_feed import Kill, KillFeedTracker
from perception.map_background import floor_mask
from perception.minimap_dots import DotDetector
from perception.regions import KILL_FEED, MINIMAP
from perception.round_state import SpikeTracker
from predict.heuristic import PairScore
from tracking.player_tracker import PlayerTracker, Track

# How long the minimap must be out of its live layout before we treat it as
# a round change and reset. Short dips (a flash on screen) are ignored.
RESET_AFTER = 1.0


@dataclass
class FrameResult:
    """What the pipeline worked out for one frame."""

    time: float
    live: bool                                   # False during replays / round breaks
    minimap: np.ndarray                          # the minimap crop (for drawing)
    tracks: list[Track] = field(default_factory=list)
    ranked: list[PairScore] = field(default_factory=list)      # best pair first; empty if no predictor
    new_kills: list[Kill] = field(default_factory=list)
    spike: tuple[float, float] | None = None     # planted spike's spot, if planted and found
    planted: bool = False
    round_reset: bool = False                    # True on the frame where a new round was detected
    timings_ms: dict[str, float] = field(default_factory=dict)   # step name -> milliseconds

    @property
    def total_ms(self) -> float:
        """Time spent on this frame by the whole pipeline."""
        return sum(self.timings_ms.values())


class LivePipeline:
    """Holds every per-match piece and runs them on each frame."""

    def __init__(self, background: np.ndarray, predictor=None, assist: bool = False):
        """
        background:  empty-minimap picture for this broadcast and map
                     (perception/map_background.py)
        predictor:   object with rank_pairs(tracks, spike=...), or None to skip ranking
        assist:      let the tracker tell the detector where to look harder
                     (experimental; see scripts/track_players.py)
        """
        self.detector = DotDetector(background, floor_mask(background))
        self.kill_feed = KillFeedTracker()
        self.tracker = PlayerTracker()
        self.spike = SpikeTracker()
        self.predictor = predictor
        self.assist = assist
        self._not_live_since: float | None = None

    def step(self, time: float, frame: np.ndarray) -> FrameResult:
        """Run steps 1-7 on one full broadcast frame taken at `time` seconds."""
        timings: dict[str, float] = {}
        started = clock.perf_counter()

        def lap(name: str) -> None:
            """Record how long the step that just finished took."""
            nonlocal started
            now = clock.perf_counter()
            timings[name] = (now - started) * 1000
            started = now

        # 1. Layout check.
        minimap = MINIMAP.crop(frame)
        live = self.detector.is_standard_layout(minimap)
        lap("layout")

        # 2. Kill feed.
        new_kills = self.kill_feed.update(time, KILL_FEED.crop(frame), record=live)
        for kill in new_kills:
            self.tracker.report_death(kill.time, kill.victim_team)
        lap("kill feed")

        result = FrameResult(time=time, live=live, minimap=minimap, new_kills=new_kills, timings_ms=timings)

        # 3. Round change.
        if not live:
            self._not_live_since = time if self._not_live_since is None else self._not_live_since
            if time - self._not_live_since >= RESET_AFTER and self.tracker.tracks:
                self.tracker.reset()
                self.spike.reset()
                result.round_reset = True
            return result
        self._not_live_since = None

        # 4. Spike.
        self.spike.update(time, frame, minimap)
        result.planted, result.spike = self.spike.planted, self.spike.spot
        lap("spike")

        # 5. Dots.
        dots = self.detector.detect(minimap)
        if self.assist:
            dots += self.detector.find_expected(minimap, self.tracker.expected_missing(time, dots), dots)
        lap("dots")

        # 6. Tracker.
        result.tracks = self.tracker.update(time, dots)
        lap("tracker")

        # 7. Predictor.
        if self.predictor is not None:
            result.ranked = self.predictor.rank_pairs(result.tracks, spike=result.spike)
        lap("predictor")
        return result
