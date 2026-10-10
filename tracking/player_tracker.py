"""
player_tracker.py: follow each player across frames, with position and speed.

Why a tracker
-------------
The dot detector works one frame at a time and makes mistakes: it misses a
player for a few frames (icons overlap, a vision cone covers the ring) and
sometimes reports a dot that isn't a player. It also doesn't know which dot
in this frame is which dot from the last frame. The predictor needs more
than that: it needs "this player is here, moving this fast, this way".

The tracker keeps one *track* per player. Every frame it:
  1. predicts where each track should be now, from its last position and
     speed,
  2. pairs each track with the nearest detected dot of the same team (if
     one is close enough),
  3. nudges each paired track towards its dot and updates its speed,
  4. lets unpaired tracks keep gliding for a short while (the player is
     probably just hidden) and deletes them if they stay unseen too long,
  5. starts a new track for a dot no track explains, but only treats it as
     a real player once it has been seen in several frames.

Deaths
------
When the kill feed reports that a player of a team died, the tracker looks
for a track of that team that just disappeared from the minimap, and marks
it dead at its last position. That tells later steps *where* the fight was
and which track was the victim.

What it does not know
---------------------
Track names like "red-3" are just numbers, not agents or player names.
A player gets a new track id each round (call reset() between rounds),
and also whenever their dot is lost for too long.

Units: positions are minimap pixels, speeds are minimap pixels per second.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from perception.minimap_dots import Dot
from perception.teams import RED, TEAL


@dataclass
class Track:
    """One player being followed."""

    id: str                 # e.g. "red-3"; unique within a round
    team: str               # RED or TEAL
    x: float
    y: float
    vx: float = 0.0         # speed in px/s, left-right (positive = right)
    vy: float = 0.0         # speed in px/s, up-down (positive = down)
    created: float = 0.0    # time the track was started
    last_seen: float = 0.0  # time a dot was last paired with it
    hits: int = 1           # how many frames a dot was paired with it
    confirmed: bool = False # True once seen often enough to trust

    @property
    def speed(self) -> float:
        """How fast the player moves, in px/s."""
        return float(np.hypot(self.vx, self.vy))


@dataclass
class Death:
    """A player death matched to a track (from the kill feed)."""

    time: float             # kill-feed time of the kill
    track_id: str
    team: str               # the team of the player who died
    x: float                # last known position of the victim
    y: float


@dataclass
class TrackerSettings:
    """All the tunable numbers in one place (pixels and seconds)."""

    # Matching: a dot may pair with a track if it is within
    #   gate_base + max_speed * (time since the track was last seen)
    # pixels of where the track is predicted to be. The longer a player
    # has been hidden, the further they may have moved.
    gate_base: float = 8.0
    max_speed: float = 40.0      # players never move faster than this (px/s)
    gate_max: float = 50.0       # ...but never further than this, however long they were hidden

    # Smoothing (an "alpha-beta filter"): how far to move towards the new
    # dot (alpha) and how much to trust the implied change of speed (beta).
    # Smaller numbers = smoother but slower to react.
    alpha: float = 0.6
    beta: float = 0.15

    # While unseen, a track's speed shrinks by this factor every second,
    # so a hidden player doesn't glide off through walls.
    coast_slowdown: float = 0.3

    # Confirmation: a new track counts as a real player after this many
    # sightings within `confirm_within` seconds; otherwise it is dropped.
    confirm_hits: int = 3
    confirm_within: float = 0.8

    # A confirmed track that stays unseen this long is deleted.
    max_unseen: float = 4.0

    players_per_team: int = 5

    # Deaths: after a kill, a track of the victim's team that has been
    # unseen for at least `death_unseen` seconds, and went unseen no more
    # than `death_window` seconds before or after the kill, is the victim.
    death_unseen: float = 0.2
    death_window: float = 1.5


@dataclass
class _PendingDeath:
    """A kill-feed death not yet matched to a track (internal bookkeeping)."""

    time: float
    team: str


@dataclass
class PlayerTracker:
    """Keeps up to 5 tracks per team, updated once per frame."""

    settings: TrackerSettings = field(default_factory=TrackerSettings)
    tracks: list[Track] = field(default_factory=list)
    deaths: list[Death] = field(default_factory=list)
    _pending_deaths: list[_PendingDeath] = field(default_factory=list)
    _next_number: dict[str, int] = field(default_factory=lambda: {RED: 1, TEAL: 1})
    _time: float | None = None

    def reset(self) -> None:
        """Forget all tracks, e.g. at the start of a new round.

        Track numbers keep counting up, so an id like "red-7" is never
        reused within one run of the tracker.
        """
        self.tracks.clear()
        self._pending_deaths.clear()
        self._time = None

    def update(self, time: float, dots: list[Dot]) -> list[Track]:
        """Feed one frame's dots (taken at `time` seconds); return confirmed tracks."""
        dt = 0.0 if self._time is None else time - self._time
        self._time = time

        self._predict(dt)
        for team in (RED, TEAL):
            team_dots = [d for d in dots if d.team == team]
            team_tracks = [t for t in self.tracks if t.team == team]
            leftover_dots = self._pair_and_correct(time, dt, team_tracks, team_dots)
            self._start_new_tracks(time, team, leftover_dots)

        self._match_deaths(time)
        self._drop_old_tracks(time)
        return self.confirmed_tracks()

    def report_death(self, time: float, team: str) -> None:
        """Tell the tracker the kill feed showed a player of `team` dying."""
        self._pending_deaths.append(_PendingDeath(time=time, team=team))

    def confirmed_tracks(self) -> list[Track]:
        """Tracks we believe are real players, alive right now."""
        return [t for t in self.tracks if t.confirmed]

    # ----- the steps of update(), in order -----

    def _predict(self, dt: float) -> None:
        """Step 1: move every track along its speed by `dt` seconds."""
        if dt <= 0:
            return
        for t in self.tracks:
            t.x += t.vx * dt
            t.y += t.vy * dt

    def _pair_and_correct(self, time: float, dt: float, tracks: list[Track], dots: list[Dot]) -> list[Dot]:
        """Steps 2-4: pair tracks with dots, correct the paired tracks.

        Pairs are made closest-first: work out every track-to-dot distance,
        then repeatedly take the shortest one whose track and dot are both
        still free. (With at most 5 tracks and a handful of dots this simple
        method is as good as anything fancier.) Unpaired tracks just coast,
        and their speed slowly dies down.

        Returns the dots no track took.
        """
        s = self.settings
        options = []
        for ti, track in enumerate(tracks):
            gate = min(s.gate_base + s.max_speed * (time - track.last_seen), s.gate_max)
            for di, dot in enumerate(dots):
                distance = float(np.hypot(dot.x - track.x, dot.y - track.y))
                if distance <= gate:
                    options.append((distance, ti, di))

        paired_tracks, paired_dots = set(), set()
        for distance, ti, di in sorted(options):
            if ti in paired_tracks or di in paired_dots:
                continue
            paired_tracks.add(ti)
            paired_dots.add(di)
            self._correct(tracks[ti], dots[di], time)

        # Unseen this frame: keep gliding, but slow down. Raising to the
        # power dt makes the slowdown per *second*, whatever the frame rate.
        factor = s.coast_slowdown ** dt
        for ti, track in enumerate(tracks):
            if ti not in paired_tracks:
                track.vx *= factor
                track.vy *= factor

        return [dot for di, dot in enumerate(dots) if di not in paired_dots]

    def _correct(self, track: Track, dot: Dot, time: float) -> None:
        """Step 3: move a track towards its paired dot and update its speed.

        `miss` is how far the dot is from where we predicted. We move part of
        the way (alpha) towards the dot, and change the speed by part (beta)
        of the speed that miss implies. This is a classic alpha-beta filter,
        a simple cousin of the Kalman filter.
        """
        s = self.settings
        elapsed = max(time - track.last_seen, 1e-3)
        miss_x, miss_y = dot.x - track.x, dot.y - track.y
        track.x += s.alpha * miss_x
        track.y += s.alpha * miss_y
        track.vx += s.beta * miss_x / elapsed
        track.vy += s.beta * miss_y / elapsed

        # Never faster than a player can actually run.
        speed = track.speed
        if speed > s.max_speed:
            track.vx *= s.max_speed / speed
            track.vy *= s.max_speed / speed

        track.last_seen = time
        track.hits += 1
        if not track.confirmed and track.hits >= s.confirm_hits:
            track.confirmed = True

    def _start_new_tracks(self, time: float, team: str, dots: list[Dot]) -> None:
        """Step 5: start a tentative track for each unexplained dot.

        Skipped when the team already has 5 tracks: an extra dot then has
        to be a false detection, or a hidden player who reappeared far away
        (their old track will pick them up once its gate has grown).
        """
        for dot in dots:
            if sum(t.team == team for t in self.tracks) >= self.settings.players_per_team:
                return
            number = self._next_number[team]
            self._next_number[team] += 1
            self.tracks.append(Track(id=f"{team}-{number}", team=team, x=dot.x, y=dot.y,
                                     created=time, last_seen=time))

    def _match_deaths(self, time: float) -> None:
        """Turn kill-feed deaths into dead tracks (see "Deaths" at the top).

        For each pending death, pick the track of that team that has been
        unseen for a while and disappeared closest in time to the kill.
        Matched tracks are removed and recorded in self.deaths. Deaths that
        find no track within `death_window` seconds are dropped.
        """
        s = self.settings
        still_pending = []
        for death in self._pending_deaths:
            candidates = [
                t for t in self.tracks
                if t.team == death.team and t.confirmed
                and time - t.last_seen >= s.death_unseen
                and abs(t.last_seen - death.time) <= s.death_window
            ]
            if candidates:
                victim = min(candidates, key=lambda t: abs(t.last_seen - death.time))
                self.deaths.append(Death(time=death.time, track_id=victim.id, team=victim.team,
                                         x=victim.x, y=victim.y))
                self.tracks.remove(victim)
            elif time - death.time <= s.death_window:
                still_pending.append(death)  # keep waiting for the dot to vanish
        self._pending_deaths = still_pending

    def _drop_old_tracks(self, time: float) -> None:
        """Delete tracks that are unseen too long, or never got confirmed."""
        s = self.settings
        kept = []
        for t in self.tracks:
            if t.confirmed and time - t.last_seen > s.max_unseen:
                continue
            if not t.confirmed and time - t.created > s.confirm_within:
                continue
            kept.append(t)
        self.tracks = kept
