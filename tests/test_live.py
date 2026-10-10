"""
test_live.py: checks for the live parts: the real-time video source and the
observer page server.

Run from the project root:

    .venv/bin/python -m pytest tests/
"""

import json
import time
import urllib.request

import cv2
import numpy as np

from capture.realtime_video import RealTimeVideo
from capture.video_source import VideoSource
from ui.server import ObserverServer


def make_video(path, frames=40, fps=20):
    """A tiny video whose frame number is written into its brightness."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for i in range(frames):
        writer.write(np.full((48, 64, 3), i * 5, np.uint8))
    writer.release()


def test_realtime_video_follows_the_clock_and_skips_frames(tmp_path):
    """After ~0.5 s of waiting, the next frame is ~0.5 s further on, not the next one."""
    path = tmp_path / "clip.mp4"
    make_video(path)
    with VideoSource(str(path)) as video:
        source = RealTimeVideo(video)
        t0, _ = source.latest()
        time.sleep(0.5)
        t1, _ = source.latest()
    assert t0 < 0.1
    assert 0.4 <= t1 - t0 <= 0.8


def test_realtime_video_ends(tmp_path):
    path = tmp_path / "clip.mp4"
    make_video(path, frames=10)
    with VideoSource(str(path)) as video:
        source = RealTimeVideo(video, speed=100)   # 100x speed: past the end almost at once
        assert source.latest() is not None         # the first call starts the clock
        time.sleep(0.1)
        assert source.latest() is None


def test_server_serves_the_page_and_latest_state():
    server = ObserverServer(port=8799)
    server.start()
    try:
        server.publish({"live": True, "watch": {"red": "3"}}, b"\xff\xd8fake-jpeg")
        state = json.loads(urllib.request.urlopen("http://127.0.0.1:8799/state").read())
        page = urllib.request.urlopen("http://127.0.0.1:8799/").read().decode()
        jpeg = urllib.request.urlopen("http://127.0.0.1:8799/minimap.jpg").read()
    finally:
        server.stop()
    assert state["watch"]["red"] == "3"
    assert "Observer assist" in page
    assert jpeg.startswith(b"\xff\xd8")
