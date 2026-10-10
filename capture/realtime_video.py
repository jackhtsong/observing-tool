"""
realtime_video.py: play a recorded VOD as if it were a live screen.

A live capture always hands you "the frame on screen right now"; if your
processing is slow, you simply miss the frames in between. This class gives
a recorded video the same behaviour, so the whole live tool (pipeline,
suggestion, observer page) can be built and tested before it ever sees a
real game. A real screen-capture source (phase 1 too) will have the same
latest() method and can be swapped in.
"""

from __future__ import annotations

import time as clock

import cv2
import numpy as np

from capture.video_source import VideoSource


class RealTimeVideo:
    """A VOD that advances with the wall clock from the moment it is started."""

    def __init__(self, video: VideoSource, start: float = 0.0, speed: float = 1.0):
        """
        video:  an open VideoSource
        start:  the second of the video to begin at
        speed:  1.0 = real time; 2.0 = twice as fast (handy for testing)
        """
        self.video = video
        self.start = start
        self.speed = speed
        self._started_at: float | None = None
        self._index = int(round(start * video.fps))   # next frame the decoder will hand out
        video._cap.set(cv2.CAP_PROP_POS_FRAMES, self._index)

    def latest(self) -> tuple[float, np.ndarray] | None:
        """The frame that would be on screen now, as (video time, frame).

        Frames we have fallen behind on are skipped with grab(), which is
        faster than decoding them. Returns None at the end of the video.
        """
        if self._started_at is None:
            self._started_at = clock.perf_counter()
        elapsed = (clock.perf_counter() - self._started_at) * self.speed
        target = int((self.start + elapsed) * self.video.fps)
        if target >= self.video.frame_count:
            return None

        cap = self.video._cap
        while self._index < target:          # skip frames we no longer need
            cap.grab()
            self._index += 1
        ok, frame = cap.read()
        if not ok:
            return None
        time = self._index / self.video.fps
        self._index += 1
        return time, frame
