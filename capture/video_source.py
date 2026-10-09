"""
video_source.py: read frames from a recorded match video.

This is the only file that talks to the video file directly. Everything else
asks it for frames by time (in seconds) instead of by frame number, because
times are easier to reason about ("the fight at 2:05") and do not depend on
the video's frame rate.

Typical use:

    with VideoSource("data/vods/ascent_test.mp4") as video:
        frame = video.frame_at(125.0)            # one frame at t = 125 s
        for t, frame in video.frames(step=0.5):  # every half second
            ...
"""

from __future__ import annotations

from collections.abc import Iterator

import cv2
import numpy as np


class VideoSource:
    """A recorded video that can hand out frames by timestamp.

    Frames are returned exactly as OpenCV reads them: NumPy arrays of shape
    (height, width, 3) in BGR colour order (blue, green, red), not RGB.
    """

    def __init__(self, path: str):
        """Open the video file and read its basic properties.

        Raises FileNotFoundError if OpenCV cannot open the file, which is
        usually a wrong path or a codec OpenCV does not support.
        """
        self.path = path
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {path}")

        # Basic facts about the video, read once up front.
        self.fps: float = self._cap.get(cv2.CAP_PROP_FPS)
        self.frame_count: int = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width: int = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height: int = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def duration(self) -> float:
        """Length of the video in seconds."""
        return self.frame_count / self.fps

    def frame_at(self, seconds: float) -> np.ndarray | None:
        """Return the frame shown at `seconds` into the video.

        Returns None if the time is past the end of the video or the frame
        cannot be decoded.
        """
        frame_index = int(round(seconds * self.fps))
        if frame_index < 0 or frame_index >= self.frame_count:
            return None
        # Jump straight to the frame instead of reading everything before it.
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self._cap.read()
        return frame if ok else None

    def frames(
        self, start: float = 0.0, end: float | None = None, step: float = 1.0
    ) -> Iterator[tuple[float, np.ndarray]]:
        """Yield (time_in_seconds, frame) pairs from `start` to `end`.

        `step` is the gap between frames in seconds. For example step=0.1
        gives 10 frames per second of video. If `end` is None we go to the
        end of the video.
        """
        end = self.duration if end is None else min(end, self.duration)

        # Jumping (seeking) in a compressed video is slow, so we jump once to
        # `start` and then read forward. Frames we don't need are skipped
        # with grab(), which is cheaper than read() because it doesn't hand
        # the picture back to Python.
        first = int(round(start * self.fps))
        last = min(int(round(end * self.fps)), self.frame_count)
        frames_per_step = max(int(round(step * self.fps)), 1)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, first)

        index = first
        while index < last:
            ok, frame = self._cap.read()
            if not ok:
                break
            yield index / self.fps, frame
            # Skip ahead to the next frame we want.
            for _ in range(frames_per_step - 1):
                self._cap.grab()
            index += frames_per_step

    def close(self) -> None:
        """Release the video file. Called automatically when used with `with`."""
        self._cap.release()

    # These two methods let us write `with VideoSource(...) as video:` so the
    # file is always closed, even if an error happens inside the block.
    def __enter__(self) -> VideoSource:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
