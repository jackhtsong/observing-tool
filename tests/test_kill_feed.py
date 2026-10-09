"""
test_kill_feed.py: checks for the kill-feed reader.

Like the minimap tests, these draw a fake kill feed instead of using the
VOD: a dark background with rows painted in team colours and white blocks
standing in for the player names. They check that rows are read with the
right teams, that a row is reported as a kill only once (even while it
slides up), and that look-alike backgrounds are ignored.

Run from the project root:

    .venv/bin/python -m pytest tests/
"""

import cv2
import numpy as np

from perception.kill_feed import KillFeedSettings, KillFeedTracker, read_rows
from perception.teams import RED, TEAL

SETTINGS = KillFeedSettings()

# BGR colours close to the broadcast's.
RED_ROW = (80, 70, 225)
TEAL_ROW = (160, 205, 100)
WHITE = (240, 240, 240)
COLOURS = {RED: RED_ROW, TEAL: TEAL_ROW}


def empty_feed() -> np.ndarray:
    """A blank kill-feed crop: dark grey game scene, no rows."""
    return np.full((165, 605, 3), 50, np.uint8)


def draw_row(feed: np.ndarray, top: int, killer: str, victim: str, name_seed: int = 0) -> None:
    """Paint one kill row with its top edge at `top`.

    `name_seed` changes the white "name" blocks, so two rows with the same
    teams can still look different, like two different player names.
    """
    bottom = top + SETTINGS.row_height
    cv2.rectangle(feed, (220, top), (440, bottom), COLOURS[killer], -1)   # killer part
    cv2.rectangle(feed, (440, top), (604, bottom), COLOURS[victim], -1)  # victim part
    rng = np.random.default_rng(name_seed)
    for x0 in (240, 350, 460):                                            # fake names / gun icon
        for _ in range(4):
            x = x0 + int(rng.integers(0, 60))
            cv2.rectangle(feed, (x, top + 12), (x + 5, top + 23), WHITE, -1)


def slot_top(slot: int) -> int:
    """Top edge (px) of a kill-feed slot."""
    return SETTINGS.first_row_top + slot * SETTINGS.row_pitch


def test_reads_teams_of_a_row():
    """A red-left / teal-right row means red killed teal."""
    feed = empty_feed()
    draw_row(feed, slot_top(0), RED, TEAL)
    rows = read_rows(feed)
    assert len(rows) == 1
    assert (rows[0].slot, rows[0].killer_team, rows[0].victim_team) == (0, RED, TEAL)


def test_empty_feed_has_no_rows():
    assert read_rows(empty_feed()) == []


def test_coloured_background_without_text_is_not_a_row():
    """A red-and-teal game scene behind the feed has no white names, so no row."""
    feed = empty_feed()
    feed[:, :440] = RED_ROW
    feed[:, 440:] = TEAL_ROW
    assert read_rows(feed) == []


def test_same_row_over_many_frames_is_one_kill():
    tracker = KillFeedTracker()
    feed = empty_feed()
    draw_row(feed, slot_top(0), TEAL, RED)
    kills = [k for i in range(20) for k in tracker.update(i * 0.1, feed)]
    assert len(kills) == 1
    assert kills[0].time == 0.0


def test_row_sliding_up_is_not_a_new_kill():
    """When the top row expires, the next row moves up a slot; no new kill."""
    tracker = KillFeedTracker()

    before = empty_feed()
    draw_row(before, slot_top(0), RED, TEAL, name_seed=1)
    draw_row(before, slot_top(1), TEAL, RED, name_seed=2)
    assert len(tracker.update(0.0, before)) == 2

    # Halfway through sliding up, then settled in slot 0.
    halfway = empty_feed()
    draw_row(halfway, slot_top(1) - 15, TEAL, RED, name_seed=2)
    after = empty_feed()
    draw_row(after, slot_top(0), TEAL, RED, name_seed=2)

    assert tracker.update(0.1, halfway) == []
    assert tracker.update(0.2, after) == []


def test_second_kill_with_same_teams_is_counted():
    """Two PRX-kills-LEV rows with different names are two kills."""
    tracker = KillFeedTracker()
    feed = empty_feed()
    draw_row(feed, slot_top(0), RED, TEAL, name_seed=1)
    assert len(tracker.update(0.0, feed)) == 1

    draw_row(feed, slot_top(1), RED, TEAL, name_seed=2)
    new = tracker.update(0.1, feed)
    assert len(new) == 1 and new[0].slot == 1


def test_replay_rows_are_remembered_but_not_reported():
    """With record=False (replays) rows are learned silently."""
    tracker = KillFeedTracker()
    feed = empty_feed()
    draw_row(feed, slot_top(0), RED, TEAL)
    assert tracker.update(0.0, feed, record=False) == []
    assert tracker.update(0.1, feed) == []
