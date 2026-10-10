"""
suggestion.py: decide which pair the observer should watch right now.

The predictor re-ranks every pair ten times a second, and the top pair can
swap back and forth when two pairs score about the same. Shown raw, the
suggestion would flicker, and an observer stops trusting a tool that
flickers. So, as the design doc says, the suggestion only changes when:

  - a different pair beats the current one by at least `margin`, and has
    kept doing so for at least `hold_for` seconds (hysteresis), or
  - the current pair no longer exists (one of the two players died or was
    lost), in which case we switch straight away.

Besides the main suggestion it also gives an "also watch" pair: the best
other pair, for the busy moments when two things happen at once.

Optionally, a minimum score (`min_score`): with a probability predictor
(predict/probability.py) this means "only suggest a pair whose chance of a
kill is over, say, 50%". Below it there is no suggestion at all. To avoid
flicker right at the line, a pair needs `min_score` to be suggested but may
stay suggested down to `min_score - keep_band`.
"""

from __future__ import annotations

from dataclasses import dataclass

from predict.heuristic import PairScore


@dataclass
class Suggestion:
    """What the observer is told at one moment."""

    pair: PairScore | None          # the pair to watch, or None if there is none
    also: PairScore | None          # the runner-up pair, if any
    since: float | None             # when this pair became the suggestion
    changed: bool = False           # True on the frame the suggestion switched


@dataclass
class PolicySettings:
    margin: float = 0.1     # how much higher a challenger must score to take over
    hold_for: float = 0.3   # seconds the challenger must stay ahead (design doc: 300 ms)
    min_score: float | None = None   # no suggestion unless a pair scores at least this
    keep_band: float = 0.05          # a suggested pair may dip this far below min_score and stay


def pair_key(pair: PairScore) -> tuple[str, str]:
    """A pair's identity: its two track ids."""
    return pair.red.id, pair.teal.id


class SuggestionPolicy:
    """Keeps the current suggestion and decides when to change it."""

    def __init__(self, settings: PolicySettings | None = None):
        self.settings = settings or PolicySettings()
        self.reset()

    def reset(self) -> None:
        """Forget the current suggestion, e.g. at a new round."""
        self._current: tuple[str, str] | None = None
        self._since: float | None = None
        self._challenger: tuple[str, str] | None = None
        self._challenger_since: float | None = None

    def update(self, time: float, ranked: list[PairScore]) -> Suggestion:
        """Take this frame's ranking (best first) and return the suggestion to show."""
        s = self.settings
        if s.min_score is not None:
            # Pairs good enough to be suggested: over the bar, or the current
            # pair still within the keep band below it.
            ranked = [p for p in ranked
                      if p.score >= s.min_score
                      or (pair_key(p) == self._current and p.score >= s.min_score - s.keep_band)]
        if not ranked:
            self.reset()
            return Suggestion(pair=None, also=None, since=None)

        by_key = {pair_key(p): p for p in ranked}
        best = ranked[0]
        changed = False

        if self._current not in by_key:
            # No suggestion yet, or the suggested pair is gone (a death): switch now.
            self._switch_to(pair_key(best), time)
            changed = True
        elif pair_key(best) != self._current:
            current_score = by_key[self._current].score
            if best.score >= current_score + s.margin:
                # A challenger is clearly ahead. Start (or keep) its clock.
                if self._challenger != pair_key(best):
                    self._challenger, self._challenger_since = pair_key(best), time
                if time - self._challenger_since >= s.hold_for:
                    self._switch_to(pair_key(best), time)
                    changed = True
            else:
                self._challenger = self._challenger_since = None
        else:
            self._challenger = self._challenger_since = None

        pair = by_key[self._current]
        also = next((p for p in ranked if pair_key(p) != self._current), None)
        return Suggestion(pair=pair, also=also, since=self._since, changed=changed)

    def _switch_to(self, key: tuple[str, str], time: float) -> None:
        self._current, self._since = key, time
        self._challenger = self._challenger_since = None
