"""
test_dot_labels.py: checks for the hand labels, the detector grading, and
the click tool's logic.

The click tool's window can't be tested automatically, so its behaviour
lives in LabelSession (clicks, undo, next frame), which is tested here
directly without opening a window.

Run from the project root:

    .venv/bin/python -m pytest tests/
"""

from perception.dot_labels import LabelledFrame, LabelledPlayer, load_labels, save_labels, score_frame
from perception.minimap_dots import Dot
from perception.teams import RED, TEAL
from scripts.label_dots import LabelSession


def dot(x: float, y: float, team: str) -> Dot:
    return Dot(x=x, y=y, radius=7, team=team, red_share=0.3, teal_share=0.3, change=30)


def player(x: float, y: float, team: str) -> LabelledPlayer:
    return LabelledPlayer(x=x, y=y, team=team)


# ----- grading -----

def test_perfect_detection():
    score = score_frame([player(10, 10, RED), player(50, 50, TEAL)], [dot(11, 10, RED), dot(50, 52, TEAL)])
    assert (score.found, score.missed, score.extra, score.wrong_team) == (2, 0, 0, 0)


def test_missed_and_extra():
    """One player missed, one dot where nobody is."""
    score = score_frame([player(10, 10, RED), player(50, 50, TEAL)], [dot(10, 10, RED), dot(200, 200, TEAL)])
    assert (score.found, score.missed, score.extra) == (1, 1, 1)


def test_right_place_wrong_team():
    score = score_frame([player(10, 10, RED)], [dot(10, 10, TEAL)])
    assert (score.found, score.missed, score.wrong_team) == (0, 0, 1)


def test_one_dot_cannot_count_for_two_players():
    """Two stacked players, one dot: one found, one missed."""
    score = score_frame([player(10, 10, RED), player(14, 10, RED)], [dot(12, 10, RED)])
    assert (score.found, score.missed) == (1, 1)


# ----- saving and loading -----

def test_labels_survive_save_and_load(tmp_path):
    path = tmp_path / "dots.json"
    frames = [LabelledFrame(video="v.mp4", time=12.5, done=True, players=[player(1.5, 2.0, TEAL)])]
    save_labels(frames, path)
    assert load_labels(path) == frames


# ----- the click tool's logic -----

def test_click_adds_and_clicking_again_removes():
    session = LabelSession([LabelledFrame(video="v.mp4", time=1.0)])
    session.click(100, 100, RED)
    session.click(200, 50, TEAL)
    assert [(p.team, p.x) for p in session.frame.players] == [(RED, 100), (TEAL, 200)]

    session.click(102, 101, TEAL)   # near the red mark: removes it, team doesn't matter
    assert [p.team for p in session.frame.players] == [TEAL]


def test_undo_and_next_frame():
    session = LabelSession([LabelledFrame(video="v.mp4", time=1.0), LabelledFrame(video="v.mp4", time=2.0)])
    session.click(10, 10, RED)
    session.click(20, 20, RED)
    session.undo()
    assert len(session.frame.players) == 1

    assert session.next() is True       # first frame done, now on the second
    assert session.frames[0].done and session.index == 1
    assert session.next() is False      # no more frames


def test_session_resumes_at_first_unfinished_frame():
    frames = [LabelledFrame(video="v.mp4", time=t, done=t < 3) for t in (1.0, 2.0, 3.0, 4.0)]
    assert LabelSession(frames).index == 2
