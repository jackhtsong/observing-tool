"""
test_round_state.py: checks for reading the spike state and the spike predictor.

Uses made-up frames: a black broadcast frame with a red block painted in
the timer box stands in for the planted-spike icon, and a yellow blob on a
grey minimap stands in for the spike icon on the map.

Run from the project root:

    .venv/bin/python -m pytest tests/
"""

import numpy as np

from perception.regions import ROUND_TIMER
from perception.round_state import SpikeTracker, find_spike, spike_planted
from perception.teams import RED, TEAL
from predict.context import SpikeContextPredictor
from tracking.player_tracker import Track

RED_ICON = (40, 40, 230)       # BGR
YELLOW = (40, 220, 240)


def broadcast(planted: bool) -> np.ndarray:
    """A 1080p frame; with planted=True the timer box shows a red block."""
    frame = np.zeros((1080, 1920, 3), np.uint8)
    if planted:
        r = ROUND_TIMER
        frame[r.y + 15:r.y + 55, r.x + 15:r.x + 55] = RED_ICON
    return frame


def minimap_with_spike(x: int, y: int) -> np.ndarray:
    minimap = np.full((410, 400, 3), 130, np.uint8)
    minimap[y - 3:y + 4, x - 3:x + 4] = YELLOW
    return minimap


def test_reads_planted_from_the_timer_box():
    assert spike_planted(broadcast(planted=True))
    assert not spike_planted(broadcast(planted=False))


def test_finds_the_spike_on_the_minimap():
    x, y = find_spike(minimap_with_spike(300, 120))
    assert abs(x - 300) < 1 and abs(y - 120) < 1
    assert find_spike(np.full((410, 400, 3), 130, np.uint8)) is None


def test_spike_tracker_ignores_one_frame_flicker_and_stays_planted():
    spike = SpikeTracker()
    minimap = minimap_with_spike(300, 120)
    spike.update(0.0, broadcast(True), minimap)       # one stray "planted" frame
    for t in (0.1, 0.2, 0.3):
        spike.update(t, broadcast(False), minimap)
    assert not spike.planted

    for t in (1.0, 1.1, 1.2):
        spike.update(t, broadcast(True), minimap)
    assert spike.planted and spike.planted_at == 1.2
    spike.update(1.3, broadcast(False), minimap)      # icon pulses off: still planted
    assert spike.planted and abs(spike.spot[0] - 300) < 1

    spike.reset()
    assert not spike.planted and spike.spot is None


def test_spike_predictor_moves_pairs_near_the_spike_up():
    """Two pairs equally close together; only one is next to the planted spike."""
    tracks = [Track(id="red-1", team=RED, x=100, y=100), Track(id="teal-1", team=TEAL, x=120, y=100),
              Track(id="red-2", team=RED, x=300, y=300), Track(id="teal-2", team=TEAL, x=320, y=300)]
    predictor = SpikeContextPredictor()
    near_spike = predictor.rank_pairs(tracks, spike=(310, 300))[0]
    assert {near_spike.red.id, near_spike.teal.id} == {"red-2", "teal-2"}
    no_spike = predictor.rank_pairs(tracks, spike=None)
    assert no_spike[0].score == no_spike[1].score            # without a plant: plain closeness, a tie
