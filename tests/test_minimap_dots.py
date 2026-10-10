"""
test_minimap_dots.py: checks for the minimap player-dot detector.

These tests do not need the VOD. They draw a fake minimap (grey floor with
a few walls), paint fake player icons and distractors onto it, and check the
detector finds exactly what it should. If a later change to the detector
breaks one of the basic rules, one of these fails.

Run from the project root:

    .venv/bin/python -m pytest tests/
"""

import cv2
import numpy as np

from perception.map_background import floor_mask
from perception.minimap_dots import RED, TEAL, DotDetector

# Colours in BGR order (OpenCV's order), picked to match the broadcast.
FLOOR_GREY = (130, 130, 130)
WALL_WHITE = (225, 225, 225)
RED_RING = (70, 60, 220)      # PRX ring
TEAL_RING = (200, 220, 90)    # LEV ring
PORTRAIT = (60, 75, 110)      # a dark, skin-ish agent portrait


def make_empty_map() -> np.ndarray:
    """Draw a fake empty minimap: a big grey floor with a few white walls."""
    img = np.full((410, 400, 3), (40, 60, 120), np.uint8)   # brownish "game camera" outside the map
    cv2.rectangle(img, (40, 40), (360, 370), FLOOR_GREY, -1)
    for x in (120, 200, 280):                              # some walls so the map has outlines
        cv2.line(img, (x, 60), (x, 160), WALL_WHITE, 2)
    cv2.line(img, (60, 250), (340, 250), WALL_WHITE, 2)
    return img


def draw_icon(img: np.ndarray, x: int, y: int, ring_colour: tuple) -> None:
    """Paint a player icon: a portrait disc wrapped in a 2 px team ring."""
    cv2.circle(img, (x, y), 8, ring_colour, -1)
    cv2.circle(img, (x, y), 6, PORTRAIT, -1)


def draw_death_x(img: np.ndarray, x: int, y: int, colour: tuple) -> None:
    """Paint an X marker like the ones left where a player died."""
    cv2.line(img, (x - 5, y - 5), (x + 5, y + 5), colour, 3)
    cv2.line(img, (x - 5, y + 5), (x + 5, y - 5), colour, 3)


def make_detector(background: np.ndarray) -> DotDetector:
    """Build a detector for the fake map, the same way the real script does."""
    return DotDetector(background, floor_mask(background))


def test_empty_map_has_no_dots():
    """Nothing on the map means nothing should be detected."""
    background = make_empty_map()
    assert make_detector(background).detect(background.copy()) == []


def test_finds_red_and_teal_icons_in_the_right_place():
    """One icon per team: both found, right team, centre within 2 px."""
    background = make_empty_map()
    frame = background.copy()
    draw_icon(frame, 90, 200, RED_RING)
    draw_icon(frame, 300, 310, TEAL_RING)

    dots = make_detector(background).detect(frame)

    assert len(dots) == 2
    by_team = {d.team: d for d in dots}
    assert abs(by_team[RED].x - 90) <= 2 and abs(by_team[RED].y - 200) <= 2
    assert abs(by_team[TEAL].x - 300) <= 2 and abs(by_team[TEAL].y - 310) <= 2


def test_death_markers_are_ignored():
    """A red X is red but not a round icon, so it must not be reported."""
    background = make_empty_map()
    frame = background.copy()
    draw_death_x(frame, 160, 300, RED_RING)
    assert make_detector(background).detect(frame) == []


def test_two_teammates_side_by_side_stay_separate():
    """Two touching icons of the same team are two players, not one."""
    background = make_empty_map()
    frame = background.copy()
    draw_icon(frame, 150, 200, TEAL_RING)
    draw_icon(frame, 167, 200, TEAL_RING)
    dots = make_detector(background).detect(frame)
    assert [d.team for d in dots] == [TEAL, TEAL]


def test_rotated_minimap_is_not_standard_layout():
    """A rotated minimap (as in some replays) must fail the layout check."""
    background = make_empty_map()
    detector = make_detector(background)
    rotation = cv2.getRotationMatrix2D((200, 205), 30, 1.0)
    rotated = cv2.warpAffine(background, rotation, (400, 410), borderValue=(40, 60, 120))

    assert detector.is_standard_layout(background)
    assert not detector.is_standard_layout(rotated)


def test_facing_points_at_the_pointer():
    """An icon with its pointer on the right faces right; on top, faces up."""
    background = make_empty_map()
    frame = background.copy()
    draw_icon(frame, 100, 200, RED_RING)
    cv2.fillPoly(frame, [np.array([(107, 195), (107, 205), (114, 200)])], RED_RING)   # pointer, right
    draw_icon(frame, 300, 310, TEAL_RING)
    cv2.fillPoly(frame, [np.array([(295, 303), (305, 303), (300, 296)])], TEAL_RING)  # pointer, up

    by_team = {d.team: d for d in make_detector(background).detect(frame)}
    assert by_team[RED].facing_x > 0.8          # right
    assert by_team[TEAL].facing_y < -0.8        # up (y grows downwards)


def test_icon_without_pointer_has_unknown_facing():
    background = make_empty_map()
    frame = background.copy()
    draw_icon(frame, 100, 200, RED_RING)
    dot = make_detector(background).detect(frame)[0]
    assert (dot.facing_x, dot.facing_y) == (0.0, 0.0)
