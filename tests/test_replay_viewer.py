"""
test_replay_viewer.py: checks for the replay viewer's logic and drawing.

The window itself can't be tested automatically, so these check the parts
it is built from: finding the next and recent kills, jumping to a kill, and
that render() produces a picture (including during replays, when nothing
was tracked).

Run from the project root:

    .venv/bin/python -m pytest tests/
"""

import numpy as np

from perception.teams import RED, TEAL
from predict.heuristic import ClosestPairBaseline
from record.track_log import TrackLog
from scripts.replay_viewer import PANEL_WIDTH, SCALE, ReplayRenderer, jump_to_kill
from tracking.player_tracker import Death, Track


def make_renderer() -> ReplayRenderer:
    """Two players tracked from 10.0 s to 12.0 s; the red one dies at 11.5 s and 30 s."""
    frames = {}
    for i in range(21):
        t = round(10.0 + i * 0.1, 1)
        frames[t] = [Track(id="red-1", team=RED, x=100, y=100, confirmed=True),
                     Track(id="teal-1", team=TEAL, x=130, y=100, confirmed=True)]
    log = TrackLog(times=sorted(frames), frames=frames)
    deaths = [Death(time=11.5, track_id="red-1", team=RED, x=100, y=100),
              Death(time=30.0, track_id="teal-1", team=TEAL, x=50, y=50)]
    return ReplayRenderer(log, deaths, {"closest pair": ClosestPairBaseline()})


def test_next_and_recent_kills():
    r = make_renderer()
    assert r.next_death(10.0).time == 11.5
    assert r.next_death(11.5).time == 30.0          # strictly after
    assert r.next_death(31.0) is None
    assert [d.time for d in r.recent_deaths(12.0)] == [11.5]   # shown for 2 s
    assert r.recent_deaths(14.0) == []


def test_jump_lands_three_seconds_before_each_kill():
    r = make_renderer()
    assert jump_to_kill(r, 0.0, forward=True, start=0, end=100) == 8.5
    assert jump_to_kill(r, 8.5, forward=True, start=0, end=100) == 27.0   # pressing again moves on
    assert jump_to_kill(r, 27.0, forward=False, start=0, end=100) == 8.5


def test_render_size_live_and_not_live():
    """A full frame in, minimap-plus-panel out; also fine when nothing was tracked."""
    r = make_renderer()
    frame = np.zeros((1080, 1920, 3), np.uint8)
    for t in (10.5, 50.0):   # 50 s is outside the tracked frames, like a replay
        picture = r.render(t, frame)
        assert picture.shape == (410 * SCALE, 400 * SCALE + PANEL_WIDTH, 3)
