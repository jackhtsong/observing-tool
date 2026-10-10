"""
test_player_tracker.py: checks for the player tracker.

The tracker only sees Dot objects, so these tests make up dots directly (no
video or images needed) and check the tracker's rules: steady movement
gives one track with the right speed, short gaps don't break a track,
one-frame flukes never become players, a team never has more than 5
tracks, and a reported death removes the right track.

Run from the project root:

    .venv/bin/python -m pytest tests/
"""

from perception.minimap_dots import Dot
from perception.teams import RED, TEAL
from tracking.player_tracker import PlayerTracker

STEP = 0.1  # seconds between frames, like the real script


def dot(x: float, y: float, team: str = RED) -> Dot:
    """A detected dot at (x, y); the other fields don't matter to the tracker."""
    return Dot(x=x, y=y, radius=7, team=team, red_share=0.3, teal_share=0.0, change=30)


def test_steady_walk_gives_one_track_with_right_speed():
    """A player walking right at 20 px/s: one track, speed close to (20, 0)."""
    tracker = PlayerTracker()
    for i in range(30):
        tracks = tracker.update(i * STEP, [dot(100 + 20 * i * STEP, 200)])
    assert len(tracks) == 1
    assert abs(tracks[0].vx - 20) < 3 and abs(tracks[0].vy) < 3


def test_short_gap_keeps_the_same_track():
    """Hidden for 1 s, then seen again further along the path: same id."""
    tracker = PlayerTracker()
    for i in range(20):
        tracker.update(i * STEP, [dot(100 + 20 * i * STEP, 200)])
    first_id = tracker.confirmed_tracks()[0].id

    for i in range(20, 30):                 # 1 s with no dot
        tracker.update(i * STEP, [])
    for i in range(30, 35):                 # back, 20 px further on
        tracks = tracker.update(i * STEP, [dot(100 + 20 * i * STEP, 200)])

    assert [t.id for t in tracks] == [first_id]


def test_one_frame_fluke_is_never_a_player():
    tracker = PlayerTracker()
    tracker.update(0.0, [dot(50, 50)])
    for i in range(1, 15):
        tracks = tracker.update(i * STEP, [])
    assert tracks == []


def test_at_most_five_tracks_per_team():
    """Seven red dots (two must be false): still only 5 red tracks."""
    tracker = PlayerTracker()
    dots = [dot(40 * k + 20, 100) for k in range(7)]
    for i in range(10):
        tracks = tracker.update(i * STEP, dots)
    assert sum(t.team == RED for t in tracks) == 5


def test_teams_are_tracked_separately():
    """A red and a teal dot on top of each other are two different players."""
    tracker = PlayerTracker()
    for i in range(10):
        tracks = tracker.update(i * STEP, [dot(100, 100, RED), dot(101, 100, TEAL)])
    assert sorted(t.team for t in tracks) == [RED, TEAL]


def test_reported_death_removes_the_vanished_track():
    """Two red players; one dot vanishes as the kill feed reports a red death."""
    tracker = PlayerTracker()
    for i in range(10):
        tracker.update(i * STEP, [dot(100, 100), dot(300, 300)])

    tracker.report_death(1.0, RED)
    for i in range(10, 20):                 # the player at (300, 300) is gone
        tracks = tracker.update(i * STEP, [dot(100, 100)])

    assert len(tracks) == 1 and abs(tracks[0].x - 100) < 1
    assert len(tracker.deaths) == 1
    death = tracker.deaths[0]
    assert death.team == RED and abs(death.x - 300) < 1 and abs(death.y - 300) < 1
