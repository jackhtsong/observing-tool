"""
test_suggestion_policy.py: checks for the hysteresis rules of the suggestion.

Run from the project root:

    .venv/bin/python -m pytest tests/
"""

from perception.teams import RED, TEAL
from policy.suggestion import SuggestionPolicy
from predict.heuristic import PairScore
from tracking.player_tracker import Track


def pair(red: str, teal: str, score: float) -> PairScore:
    return PairScore(red=Track(id=red, team=RED, x=0, y=0), teal=Track(id=teal, team=TEAL, x=0, y=0), score=score)


def ranked(*pairs: PairScore) -> list[PairScore]:
    return sorted(pairs, key=lambda p: p.score, reverse=True)


def test_first_frame_suggests_the_best_pair():
    s = SuggestionPolicy().update(0.0, ranked(pair("red-1", "teal-1", 0.9), pair("red-2", "teal-2", 0.5)))
    assert (s.pair.red.id, s.also.red.id, s.changed) == ("red-1", "red-2", True)


def test_small_swaps_do_not_flicker():
    """The two pairs keep swapping by a hair: the suggestion stays put."""
    switches = 0
    policy = SuggestionPolicy()
    for i in range(20):
        a, b = (0.80, 0.82) if i % 2 else (0.82, 0.80)
        switches += policy.update(i * 0.1, ranked(pair("red-1", "teal-1", a), pair("red-2", "teal-2", b))).changed
    assert switches == 1   # only the very first suggestion


def test_clear_challenger_takes_over_after_the_hold_time():
    policy = SuggestionPolicy()
    policy.update(0.0, ranked(pair("red-1", "teal-1", 0.9), pair("red-2", "teal-2", 0.5)))
    # From 1.0 s the second pair is clearly ahead.
    times = [1.0, 1.1, 1.2, 1.3]
    picks = [policy.update(t, ranked(pair("red-1", "teal-1", 0.5), pair("red-2", "teal-2", 0.9))).pair.red.id
             for t in times]
    assert picks == ["red-1", "red-1", "red-1", "red-2"]   # switches once it has led for 0.3 s


def test_switches_at_once_when_the_pair_disappears():
    policy = SuggestionPolicy()
    policy.update(0.0, ranked(pair("red-1", "teal-1", 0.9), pair("red-2", "teal-2", 0.5)))
    s = policy.update(0.1, ranked(pair("red-2", "teal-2", 0.5)))   # teal-1 died
    assert s.pair.red.id == "red-2" and s.changed


def test_min_score_hides_unlikely_pairs_and_keeps_a_band():
    """With a 50% bar: nothing at 40%; shown at 55%; kept at 47%; dropped at 40%."""
    from policy.suggestion import PolicySettings
    policy = SuggestionPolicy(PolicySettings(min_score=0.5, keep_band=0.05))
    assert policy.update(0.0, ranked(pair("red-1", "teal-1", 0.40))).pair is None
    assert policy.update(0.1, ranked(pair("red-1", "teal-1", 0.55))).pair.red.id == "red-1"
    assert policy.update(0.2, ranked(pair("red-1", "teal-1", 0.47))).pair.red.id == "red-1"
    assert policy.update(0.3, ranked(pair("red-1", "teal-1", 0.40))).pair is None


def test_min_score_also_applies_to_the_also_watch_pair():
    from policy.suggestion import PolicySettings
    policy = SuggestionPolicy(PolicySettings(min_score=0.5))
    s = policy.update(0.0, ranked(pair("red-1", "teal-1", 0.7), pair("red-2", "teal-2", 0.3)))
    assert s.pair.red.id == "red-1" and s.also is None
