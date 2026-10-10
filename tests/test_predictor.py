"""
test_predictor.py: checks for map geometry and the fight predictor.

Uses a tiny made-up map: two rooms side by side, joined only by a door at
the bottom. That is enough to check that walking distance goes around
walls, line of sight is blocked by them, and the predictor prefers
enemies that are near and closing in.

Run from the project root:

    .venv/bin/python -m pytest tests/
"""

import numpy as np

from maps.map_geometry import MapGeometry
from perception.teams import RED, TEAL
from predict.heuristic import ClosestPairBaseline, HeuristicPredictor
from tracking.player_tracker import Track


def two_rooms() -> np.ndarray:
    """Floor mask: left room x 20-100, right room x 120-200, door at the bottom (y 170-190)."""
    floor = np.zeros((220, 220), np.uint8)
    floor[20:200, 20:100] = 255     # left room
    floor[20:200, 120:200] = 255    # right room
    floor[170:190, 100:120] = 255   # door joining them near the bottom
    return floor


def track(id: str, team: str, x: float, y: float, vx: float = 0.0, vy: float = 0.0) -> Track:
    return Track(id=id, team=team, x=x, y=y, vx=vx, vy=vy, confirmed=True)


def test_walking_goes_around_the_wall():
    """Across the wall at the top: 40 px straight, but a long walk via the door."""
    g = MapGeometry(two_rooms())
    a, b = (90, 40), (130, 40)
    assert g.walking_distance(a, b) > 200


def test_line_of_sight_inside_a_room_but_not_through_the_wall():
    g = MapGeometry(two_rooms())
    assert g.line_of_sight((30, 40), (90, 150))       # same room
    assert not g.line_of_sight((90, 40), (130, 40))   # through the wall


def test_heuristic_prefers_the_pair_you_can_actually_reach():
    """Same straight-line distance, but one pair is split by the wall."""
    g = MapGeometry(two_rooms())
    tracks = [
        track("red-1", RED, 90, 40), track("teal-1", TEAL, 130, 40),      # across the wall
        track("red-2", RED, 40, 120), track("teal-2", TEAL, 80, 120),     # same room
    ]
    best = HeuristicPredictor(g).rank_pairs(tracks)[0]
    assert {best.red.id, best.teal.id} == {"red-2", "teal-2"}


def test_heuristic_prefers_players_moving_towards_each_other():
    """Two pairs equally far apart; only one is closing in."""
    g = MapGeometry(two_rooms())
    tracks = [
        track("red-1", RED, 30, 50, vx=20), track("teal-1", TEAL, 90, 50, vx=-20),   # closing
        track("red-2", RED, 130, 150), track("teal-2", TEAL, 190, 150),              # standing still
    ]
    best = HeuristicPredictor(g).rank_pairs(tracks)[0]
    assert {best.red.id, best.teal.id} == {"red-1", "teal-1"}


def test_baseline_ranks_by_straight_distance_only():
    """The baseline ignores walls, so it picks the pair split by the wall."""
    tracks = [
        track("red-1", RED, 90, 40), track("teal-1", TEAL, 130, 40),      # 40 px, through the wall
        track("red-2", RED, 30, 120), track("teal-2", TEAL, 90, 120),     # 60 px, same room
    ]
    best = ClosestPairBaseline().rank_pairs(tracks)[0]
    assert {best.red.id, best.teal.id} == {"red-1", "teal-1"}


def test_every_enemy_pair_is_scored_once():
    g = MapGeometry(two_rooms())
    tracks = [track(f"red-{i}", RED, 30 + 10 * i, 50) for i in range(3)] + \
             [track(f"teal-{i}", TEAL, 150 + 10 * i, 50) for i in range(2)]
    assert len(HeuristicPredictor(g).rank_pairs(tracks)) == 3 * 2
