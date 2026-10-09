r"""
kill_feed.py: read the kill feed and report each new kill as it appears.

What the kill feed looks like
-----------------------------
In the top-right corner, every kill adds one row:

    [killer portrait] TEAM killer-name [gun] | TEAM victim-name [victim portrait]
    \_______ killer's team colour _______/   \______ victim's team colour ______/

The left part of the row is painted in the killer's team colour and the
right part in the victim's team colour (red = PRX, teal = LEV here). Rows
are right-aligned, about 35 px tall, and sit in fixed "slots" 39 px apart.
A new kill is added in the next free slot below the others; after a few
seconds the oldest row disappears and the rest slide up one slot.

What we read
------------
For phase 0 we only need *when* each kill happened and *which team* killed
which. Player names would need text recognition (OCR), which we don't have
yet; instead we save a picture of every row so a person can check it.

How it works
------------
1. read_rows():  for each slot, look at two fixed windows: one inside the
                 killer part and one inside the victim part. Each window
                 votes red or teal by counting coloured pixels. A slot holds
                 a kill row only if both windows have a clear colour, the
                 two colours differ, and there is white text in the row.
                 Those rules keep the game scene behind the feed (which can
                 be any colour) from being mistaken for a row.
2. KillFeedTracker.update():  compare this frame's rows with the rows seen
                 just before. A row that matches none of them (same teams,
                 similar-looking picture) is a new kill. Matching by picture
                 rather than by slot is what lets rows slide up without
                 being counted twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from perception.teams import RED, TEAL, UNKNOWN


@dataclass
class FeedRow:
    """One kill row currently visible in the kill feed."""

    slot: int              # 0 = top row, 1 = the one below, ...
    killer_team: str       # RED or TEAL
    victim_team: str       # RED or TEAL
    image: np.ndarray      # the row cut out of the frame (BGR), for matching and checking
    surroundings: np.ndarray  # the row plus `match_slack` px above and below, for matching


@dataclass
class Kill:
    """A kill event: the moment a new row first appeared in the kill feed."""

    time: float            # seconds into the video
    killer_team: str
    victim_team: str
    slot: int              # which slot the row appeared in
    image: np.ndarray      # picture of the row, so the names can be read by a person


@dataclass
class KillFeedSettings:
    """All the tunable numbers in one place.

    Positions are in pixels inside the KILL_FEED crop (see regions.py).
    Colours use OpenCV's HSV scale: hue 0-179, saturation and value 0-255.
    """

    # Slot layout.
    first_row_top: int = 4       # top edge of the first row
    row_pitch: int = 39          # distance from one row's top to the next row's top
    row_height: int = 35
    slot_count: int = 4          # how many rows fit in the crop

    # Sampling windows (x from, x to). The victim window sits just left of
    # the victim portrait; the killer window just left of where the victim
    # part can start, so it is inside the killer part even for short names.
    killer_window: tuple[int, int] = (300, 415)
    victim_window: tuple[int, int] = (470, 530)
    # Only the middle of the row's height is sampled, to stay off its edges.
    window_margin: int = 8

    # Picture used to match rows between frames: everything from the killer
    # window to the right edge, which is always part of the row.
    match_from_x: int = 300

    # Which pixels count as coloured, and which colour they are.
    min_saturation: int = 40
    min_value: int = 120
    red_hue_low: int = 160       # red wraps around 0: hue >= 160 or hue <= 10
    red_hue_high: int = 10
    teal_hue_low: int = 55
    teal_hue_high: int = 100

    # A window votes for a team when at least `min_coloured` of its pixels
    # are coloured and at least `min_agreement` of those are that team.
    min_coloured: float = 0.35
    min_agreement: float = 0.75

    # White text check: share of near-white pixels a real row must have.
    white_max_saturation: int = 60
    white_min_value: int = 190
    min_white_text: float = 0.03

    # Rows match between frames when their pictures correlate at least this
    # much (1.0 = identical). Lower than 1 because the game scene behind
    # the semi-transparent rows keeps changing a little.
    min_match: float = 0.75

    # How far (px) a row may have moved up or down between two frames and
    # still match. Half a slot covers rows mid-way through sliding up.
    match_slack: int = 20

    # A row must be gone for this long before it is forgotten. This stops a
    # row that flickers for one frame (e.g. during its slide-in animation)
    # from being counted as a second kill when it reappears.
    forget_after: float = 1.5


def read_rows(feed: np.ndarray, settings: KillFeedSettings | None = None) -> list[FeedRow]:
    """Return every kill row visible in one kill-feed crop (step 1)."""
    s = settings or KillFeedSettings()
    hsv = cv2.cvtColor(feed, cv2.COLOR_BGR2HSV)

    rows = []
    for slot in range(s.slot_count):
        top = s.first_row_top + slot * s.row_pitch
        if top + s.row_height > feed.shape[0]:
            break  # this slot would hang off the bottom of the crop

        # Middle band of the row's height.
        band = slice(top + s.window_margin, top + s.row_height - s.window_margin)
        killer = _window_team(hsv[band, s.killer_window[0]:s.killer_window[1]], s)
        victim = _window_team(hsv[band, s.victim_window[0]:s.victim_window[1]], s)

        # Both parts must be a clear team colour, and the colours must differ.
        if killer == UNKNOWN or victim == UNKNOWN or killer == victim:
            continue

        # Real rows carry white text (names, gun icon). A red-and-teal
        # pattern in the game scene behind the feed does not.
        text_area = hsv[band, s.match_from_x:]
        white = (text_area[..., 1] < s.white_max_saturation) & (text_area[..., 2] > s.white_min_value)
        if white.mean() < s.min_white_text:
            continue

        image = feed[top:top + s.row_height].copy()
        surroundings = feed[max(top - s.match_slack, 0):top + s.row_height + s.match_slack].copy()
        rows.append(FeedRow(slot=slot, killer_team=killer, victim_team=victim,
                            image=image, surroundings=surroundings))
    return rows


def _window_team(window_hsv: np.ndarray, s: KillFeedSettings) -> str:
    """Decide which team colour fills a small window of a row, if any.

    White text and icons are not coloured (low saturation), so they are
    simply ignored, and the vote is among the remaining coloured pixels.
    """
    hue = window_hsv[..., 0].astype(int)
    sat = window_hsv[..., 1]
    val = window_hsv[..., 2]

    coloured = (sat >= s.min_saturation) & (val >= s.min_value)
    if coloured.mean() < s.min_coloured:
        return UNKNOWN

    red = coloured & ((hue >= s.red_hue_low) | (hue <= s.red_hue_high))
    teal = coloured & (hue >= s.teal_hue_low) & (hue <= s.teal_hue_high)
    total = coloured.sum()
    if red.sum() >= s.min_agreement * total:
        return RED
    if teal.sum() >= s.min_agreement * total:
        return TEAL
    return UNKNOWN


@dataclass
class _SeenRow:
    """A row the tracker has seen recently (internal bookkeeping)."""

    row: FeedRow
    last_seen: float


@dataclass
class KillFeedTracker:
    """Turns per-frame rows into kill events (step 2).

    Feed it frames in time order with update(); it returns the kills whose
    rows appeared for the first time in that frame.
    """

    settings: KillFeedSettings = field(default_factory=KillFeedSettings)
    _recent: list[_SeenRow] = field(default_factory=list)
    _previous_time: float | None = None   # time of the frame before this one

    def update(self, time: float, feed: np.ndarray, record: bool = True) -> list[Kill]:
        """Process one kill-feed crop taken at `time` seconds.

        With record=False the rows are still remembered (so they won't be
        reported later) but no kills are returned. The script uses this
        during replays, which re-show old kill rows.
        """
        s = self.settings
        rows = read_rows(feed, s)
        matches = self._match_rows(rows)

        new_kills = []
        for i, row in enumerate(rows):
            if i in matches:
                # Already known; refresh it with the latest look and time, so
                # a row that slowly fades out still matches frame to frame.
                matches[i].row = row
                matches[i].last_seen = time
                continue
            self._recent.append(_SeenRow(row=row, last_seen=time))
            if record:
                new_kills.append(Kill(time=time, killer_team=row.killer_team,
                                      victim_team=row.victim_team, slot=row.slot, image=row.image))

        # Forget rows that have not been seen for a while.
        self._recent = [r for r in self._recent if time - r.last_seen <= s.forget_after]
        self._previous_time = time
        return new_kills

    def _match_rows(self, rows: list[FeedRow]) -> dict[int, _SeenRow]:
        """Pair this frame's rows with recently seen rows.

        Returns {index in `rows`: the seen row it is}. Rows left out of the
        result are new. A pair needs the same killer and victim teams and
        similar pictures. When two kills have the same teams (e.g. PRX kills
        two LEV players), the picture is what tells them apart, since names
        and portraits differ.

        Each seen row can be paired only once, and the most similar pairs are
        made first. Without that rule, a brand-new row could "steal" the
        match of an older row with the same teams.
        """
        candidates = []
        for i, row in enumerate(rows):
            for seen in self._recent:
                if (seen.row.killer_team, seen.row.victim_team) != (row.killer_team, row.victim_team):
                    continue
                score = _similarity(seen.row, row, self.settings)
                if score >= self.settings.min_match:
                    candidates.append((score, i, seen))

        matches: dict[int, _SeenRow] = {}
        used: list[_SeenRow] = []
        for score, i, seen in sorted(candidates, key=lambda c: c[0], reverse=True):
            if i in matches or any(seen is u for u in used):
                continue
            matches[i] = seen
            used.append(seen)

        # Fallback for rows whose look changed suddenly (for example a bright
        # flash in the game scene behind the half-transparent row). If a row
        # was in this exact slot, with the same teams, in the previous frame,
        # it is the same row. New kills never appear in place of an old row;
        # they are added below the others.
        for i, row in enumerate(rows):
            if i in matches:
                continue
            for seen in self._recent:
                if (any(seen is u for u in used)
                        or seen.last_seen != self._previous_time
                        or seen.row.slot != row.slot
                        or (seen.row.killer_team, seen.row.victim_team) != (row.killer_team, row.victim_team)):
                    continue
                matches[i] = seen
                used.append(seen)
                break
        return matches


def _similarity(old: FeedRow, new: FeedRow, s: KillFeedSettings) -> float:
    """How alike two rows look, from -1 to 1 (normalised correlation).

    We look for the middle of the old row anywhere in the new row's
    surroundings, i.e. allowing it to sit up to `match_slack` px higher or
    lower. That matters while rows slide up one slot: for a few frames a
    row sits between slots, and comparing it at exactly the slot position
    would make it look like a different row.

    Only the part right of `match_from_x` is compared, since that is always
    row content; left of it may be the game scene, which keeps changing.
    """
    m = s.window_margin
    middle = old.image[m:s.row_height - m, s.match_from_x:]
    search = new.surroundings[:, s.match_from_x:]
    grey_middle = cv2.cvtColor(middle, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grey_search = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # matchTemplate slides the middle strip down the search area and scores
    # every position; the best position's score is the answer.
    return float(cv2.matchTemplate(grey_search, grey_middle, cv2.TM_CCOEFF_NORMED).max())
