"""
minimap_dots.py: find every player icon on the minimap and say which team it is.

What a player icon looks like
-----------------------------
On the observer minimap each living player is a small round agent portrait,
about 14-18 px across at 1080p, wrapped in a thin coloured ring with a little
pointer showing where they are looking. The ring colour is the team colour:

    PRX  -> red / pink
    LEV  -> teal / mint green

Things that look a bit like icons but are NOT players:
    - X markers where players died (red or blue crosses)
    - ability markers (blue squares, small blue crosses)
    - smokes (big dark circles), the yellow spike icon
    - map decals like the "A" and "B" letters

How the detector works (one frame)
----------------------------------
0. Right layout?        Check the frame shows the normal minimap (not a
                        replay or a rotated map). If not, skip the frame.
1. Find round shapes.   A circle finder (Hough transform) proposes every
                        roughly icon-sized circle on the minimap.
2. On the floor?        Drop circles whose centre is not on the map floor.
3. Which team?          Count red and teal pixels in a thin ring around the
                        circle. Whichever colour is common enough wins. If
                        neither is, the team is "unknown" (smokes, faded
                        ability icons, or a player whose ring is hidden).
4. Something there?     Compare the inside of the circle with the empty-map
                        background. An icon changes those pixels; a map
                        decal or a wall corner does not. Unknown circles are
                        dropped by default (see keep_unknown).
5. Clean up.            Merge double circles on the same icon, and keep at
                        most 5 dots per team (a team only has 5 players).

Known weak spots: players stacked on top of each other, rings covered by
a vision cone, and team-coloured smokes can still fool it. The tracker
(next phase 0 step) smooths these out over time.

Everything is pure colour and shape rules, no machine learning. That is the
phase 0 plan; a trained detector replaces this in phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from perception.teams import RED, TEAL, UNKNOWN


@dataclass
class Dot:
    """One detected player icon, in minimap pixel coordinates."""

    x: float           # centre, pixels from the left edge of the minimap crop
    y: float           # centre, pixels from the top edge of the minimap crop
    radius: float      # icon radius in pixels
    team: str          # RED, TEAL or UNKNOWN
    red_share: float   # fraction of ring pixels that are team red (0..1)
    teal_share: float  # fraction of ring pixels that are team teal (0..1)
    change: float      # how different the icon area is from the empty map (0..255)
    facing_x: float = 0.0   # which way the player looks, as a direction (x, y);
    facing_y: float = 0.0   #   (0, 0) when the pointer couldn't be found


@dataclass
class DotDetectorSettings:
    """All the tunable numbers in one place.

    Sizes are in minimap pixels at 1080p. Colour thresholds use OpenCV's HSV
    scale: hue 0-179 (half of 0-360 degrees), saturation and value 0-255.
    """

    # Before anything: skip frames whose map outlines do not match the empty
    # map (replays, round-end screens, a zoomed or rotated minimap).
    min_layout_score: float = 0.4

    # Step 1: circle finder.
    upscale: int = 2              # icons are tiny; doubling the size helps the circle finder
    min_radius: float = 5.5       # smallest icon radius we accept
    max_radius: float = 9.5       # largest icon radius we accept
    min_distance: float = 7.0     # two circle centres must be at least this far apart
    edge_strength: float = 80.0   # how sharp an edge must be to count (Hough param1)
    circle_votes: float = 11.0    # how "complete" a circle must be (Hough param2); lower finds more

    # Step 2: allow centres slightly off the floor, since icons near walls
    # can overhang the edge of the floor.
    floor_margin: int = 4

    # Step 3: team ring colours.
    red_hue_low: int = 158        # red wraps around hue 0, so it is two ranges:
    red_hue_high: int = 8         #   hue >= 158 (pinkish red) or hue <= 8
    teal_hue_low: int = 50
    teal_hue_high: int = 95
    min_saturation: int = 60      # ring pixels must be clearly coloured...
    min_value: int = 100          # ...and not too dark
    min_ring_share: float = 0.08  # at least 8% of ring pixels must be the team colour
    max_inside_share: float = 0.5   # more team colour than this INSIDE means it is background, not a ring
                                    # (not lower: red agents' skin and hair are reddish too)

    # The spike icon (yellow) next to its carrier often gets a circle of its
    # own. Circles that are this yellow inside are dropped.
    max_yellow_inside: float = 0.3

    # Step 4: how much more the inside of the circle has changed (versus the
    # empty map) than the floor just around it. Comparing with the
    # surroundings matters because the whole minimap is see-through: a bright
    # or coloured game scene behind it tints every floor pixel at once, and
    # that tint must not count as "something is there".
    min_change_team: float = 3.0      # for circles with a team-coloured ring
    min_change_unknown: float = 30.0  # for circles without one

    # Facing (see _facing): the band outside the ring where the pointer is,
    # in px beyond the radius, and how clear the pointer must be.
    pointer_inner: float = 1.5
    pointer_outer: float = 6.0
    pointer_min_pixels: int = 4
    pointer_min_strength: float = 0.35   # 1.0 = every band pixel on one side

    # Two same-team dots closer than this many radii are one icon (see
    # _remove_duplicates).
    merge_distance: float = 1.3

    # find_expected(): how far around the tracker's predicted spot to search,
    # and the lower bars a dot must pass there.
    search_radius: int = 4
    expected_ring_share: float = 0.04
    expected_change: float = 8.0
    expected_keep_away: float = 9.0   # px from any dot already found

    # A team has 5 players, so never report more than 5 dots of one colour.
    max_per_team: int = 5

    # Circles with no team ring are mostly not players (smokes, faded ability
    # icons, floor marks), so they are thrown away unless this is switched on.
    keep_unknown: bool = False


@dataclass
class _ColourMaps:
    """Per-pixel facts about one frame, worked out once and shared by all candidates."""

    red: np.ndarray      # True where the pixel is team red
    teal: np.ndarray     # True where the pixel is team teal
    yellow: np.ndarray   # True where the pixel is spike yellow
    change: np.ndarray   # how different each pixel is from the empty map (0..255)


class DotDetector:
    """Finds player icons on minimap crops of one particular video.

    It needs the empty-map background for that video (see map_background.py)
    because step 3 compares each frame against it.
    """

    def __init__(self, background: np.ndarray, floor: np.ndarray, settings: DotDetectorSettings | None = None):
        """Store the background and pre-compute the area where icons may appear.

        background: empty minimap, BGR, same size as the crops we will detect on.
        floor:      floor mask from map_background.floor_mask (255 = floor).
        """
        self.settings = settings or DotDetectorSettings()
        self.background = background.astype(np.int16)  # signed, so subtraction can go negative

        self.floor = floor > 0

        # Outlines of the empty map (walls and floor edges), used by
        # layout_score() to check a frame still shows the usual minimap.
        self.background_edges = self._edges(background)

        # Grow the floor mask by `floor_margin` pixels in every direction.
        grow = 2 * self.settings.floor_margin + 1
        self.allowed_area = cv2.dilate(floor, np.ones((grow, grow), np.uint8)) > 0

    def layout_score(self, minimap: np.ndarray) -> float:
        """How well this frame's map outlines line up with the empty map (0..1).

        Between rounds the broadcast shows replays, round-end banners and
        sometimes a zoomed or rotated minimap. Every pixel position is then
        wrong, so detecting dots would be meaningless. We find the edges
        (outlines) in this frame and measure what share of them fall on an
        outline of the empty map. Normal live-round frames score about
        0.7-0.9; replays and transitions score below 0.25.
        """
        edges = self._edges(minimap)
        # Allow 1 px of wiggle, since video compression blurs edges slightly.
        near_background_edge = cv2.dilate(self.background_edges.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        total = int(edges.sum())
        if total == 0:
            return 0.0
        return float((edges & near_background_edge).sum()) / total

    def is_standard_layout(self, minimap: np.ndarray) -> bool:
        """True if the minimap looks like the usual live-round minimap."""
        return self.layout_score(minimap) >= self.settings.min_layout_score

    def detect(self, minimap: np.ndarray) -> list[Dot]:
        """Run steps 1-4 on one minimap crop and return the icons found.

        This assumes the frame shows the standard minimap; call
        is_standard_layout() first and skip frames where it is False.
        """
        hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
        red_pixels, teal_pixels = self._team_colour_masks(hsv)
        yellow_pixels = (hsv[..., 0] >= 20) & (hsv[..., 0] <= 35) & (hsv[..., 1] >= 120) & (hsv[..., 2] >= 150)
        change_map = np.abs(minimap.astype(np.int16) - self.background).mean(axis=2)

        maps = _ColourMaps(red=red_pixels, teal=teal_pixels, yellow=yellow_pixels, change=change_map)

        # Step 1: find circles; steps 2-4: check each one (see _check).
        dots = [d for d in (self._check(x, y, r, maps) for x, y, r in self._find_circles(minimap)) if d]
        # Step 5: clean up.
        dots = self._remove_duplicates(dots)

        return self._limit_per_team(dots)

    def find_expected(self, minimap: np.ndarray, expected: list[tuple[float, float, str]],
                      existing: list[Dot]) -> list[Dot]:
        """Look harder for players the tracker expects but detect() didn't find.

        expected: (x, y, team) of where the tracker predicts each missing
        player to be. Around each spot we try every centre within
        `search_radius` px, keep the one whose ring has the most of that
        team's colour, and accept it with lower bars than detect() uses: a
        faint ring right where a player should be is very likely that player.

        existing: the dots detect() already found. A find that lands on one
        of them (another player's icon next to the expected spot) is not a
        new player and is dropped, and each team stays at 5 dots at most.
        """
        s = self.settings
        if not expected:
            return []
        hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
        red_pixels, teal_pixels = self._team_colour_masks(hsv)
        yellow_pixels = (hsv[..., 0] >= 20) & (hsv[..., 0] <= 35) & (hsv[..., 1] >= 120) & (hsv[..., 2] >= 150)
        change_map = np.abs(minimap.astype(np.int16) - self.background).mean(axis=2)
        maps = _ColourMaps(red=red_pixels, teal=teal_pixels, yellow=yellow_pixels, change=change_map)

        found = []
        r = (s.min_radius + s.max_radius) / 2
        offsets = range(-s.search_radius, s.search_radius + 1)
        for ex, ey, team in expected:
            tries = [self._check(ex + dx, ey + dy, r, maps, s.expected_ring_share, s.expected_change)
                     for dx in offsets for dy in offsets]
            tries = [d for d in tries if d is not None and d.team == team
                     and not any(np.hypot(d.x - e.x, d.y - e.y) < s.expected_keep_away for e in existing + found)]
            if tries:
                found.append(max(tries, key=lambda d: d.red_share if team == RED else d.teal_share))

        # Respect the 5-per-team limit, counting the dots already found.
        kept = []
        for d in found:
            if sum(e.team == d.team for e in existing + kept) < s.max_per_team:
                kept.append(d)
        return kept

    def _check(self, x: float, y: float, r: float, maps: _ColourMaps,
               min_ring_share: float | None = None, min_change: float | None = None) -> Dot | None:
        """Steps 2-4 for one candidate circle: return a Dot, or None if it fails.

        min_ring_share and min_change override the settings' usual bars
        (find_expected uses lower ones).
        """
        min_ring_share = self.settings.min_ring_share if min_ring_share is None else min_ring_share
        min_change = self.settings.min_change_team if min_change is None else min_change
        shape = maps.change.shape
        # Step 2: centre must be on (or right next to) the floor.
        if not self._inside(x, y) or not self.allowed_area[int(y), int(x)]:
            return None

        # Step 3: share of red and teal pixels in a ring around the circle.
        # The ring goes from just inside the edge to a few pixels outside,
        # which also covers the little "facing" pointer.
        ring = self._disc_mask(shape, x, y, r + 3) & ~self._disc_mask(shape, x, y, r - 2)
        red_share = float(maps.red[ring].mean())
        teal_share = float(maps.teal[ring].mean())
        team = self._pick_team(red_share, teal_share, min_ring_share)

        # A real icon has a portrait inside the ring, so its inside is not
        # mostly team colour. When the inside IS mostly team colour, the
        # "ring" is really a red or teal game scene showing through a gap
        # in the see-through minimap, so we ignore the colour.
        inside = self._disc_mask(shape, x, y, r - 1)
        if team == RED and maps.red[inside].mean() > self.settings.max_inside_share:
            team = UNKNOWN
        elif team == TEAL and maps.teal[inside].mean() > self.settings.max_inside_share:
            team = UNKNOWN

        # The spike icon is not a player.
        if maps.yellow[inside].mean() > self.settings.max_yellow_inside:
            return None

        # Step 4: change inside the circle minus the typical change of the
        # floor just around it (see the settings for why we subtract).
        around = (self._disc_mask(shape, x, y, r + 8)
                  & ~self._disc_mask(shape, x, y, r + 3)
                  & self.floor)
        local_tint = float(np.median(maps.change[around])) if around.any() else 0.0
        change = float(maps.change[inside].mean()) - local_tint

        # A coloured ring is strong evidence on its own, so team dots only
        # need a little change. Circles without a team ring need a lot of
        # change, and are dropped entirely unless keep_unknown is on.
        if team == UNKNOWN:
            if not self.settings.keep_unknown or change < self.settings.min_change_unknown:
                return None
        elif change < min_change:
            return None

        facing_x, facing_y = self._facing(x, y, r, maps.red if team == RED else maps.teal) \
            if team != UNKNOWN else (0.0, 0.0)
        return Dot(x=x, y=y, radius=r, team=team, red_share=red_share, teal_share=teal_share,
                   change=change, facing_x=facing_x, facing_y=facing_y)

    def _facing(self, x: float, y: float, r: float, team_pixels: np.ndarray) -> tuple[float, float]:
        """Which way the icon's pointer sticks out: a direction (x, y), or (0, 0).

        The team-coloured ring goes all the way round the portrait, except
        for the little pointer (a teardrop tip) on the side the player is
        looking. So we take the team-coloured pixels in a band just outside
        the ring, and add up the directions from the icon centre to each of
        them. Pixels all round cancel out; the pointer's pixels don't, so the
        total points at the pointer.

        If the total is weak compared to the number of pixels (no clear
        pointer, e.g. it is hidden under another icon), we return (0, 0) for
        "unknown" rather than a guess. Note y grows downwards on screen.
        """
        s = self.settings
        h, w = team_pixels.shape
        reach = int(r + s.pointer_outer) + 1
        x0, x1 = max(int(x) - reach, 0), min(int(x) + reach + 1, w)
        y0, y1 = max(int(y) - reach, 0), min(int(y) + reach + 1, h)
        ys, xs = np.nonzero(team_pixels[y0:y1, x0:x1])
        dx, dy = xs + x0 - x, ys + y0 - y
        distance = np.hypot(dx, dy)
        band = (distance >= r + s.pointer_inner) & (distance <= r + s.pointer_outer)
        if band.sum() < s.pointer_min_pixels:
            return 0.0, 0.0
        sum_x = float((dx[band] / distance[band]).sum())
        sum_y = float((dy[band] / distance[band]).sum())
        length = float(np.hypot(sum_x, sum_y))
        if length / band.sum() < s.pointer_min_strength:
            return 0.0, 0.0
        return sum_x / length, sum_y / length

    # ----- helpers used by detect() -----

    def _limit_per_team(self, dots: list[Dot]) -> list[Dot]:
        """Keep at most `max_per_team` dots per team.

        A team never has more than 5 players, so extra dots must be false
        detections (team-coloured smokes, abilities). We keep the ones with
        the strongest team-coloured ring. Unknown dots are left alone.
        """
        limit = self.settings.max_per_team
        kept = [d for d in dots if d.team == UNKNOWN]
        for team, share in ((RED, lambda d: d.red_share), (TEAL, lambda d: d.teal_share)):
            team_dots = sorted((d for d in dots if d.team == team), key=share, reverse=True)
            kept.extend(team_dots[:limit])
        return kept

    @staticmethod
    def _edges(image: np.ndarray) -> np.ndarray:
        """Return a True/False image marking sharp outlines (Canny edge detector)."""
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(grey, 50, 150) > 0

    def _find_circles(self, minimap: np.ndarray) -> list[tuple[float, float, float]]:
        """Step 1: return (x, y, radius) of every icon-sized circle.

        The image is enlarged first because the circle finder struggles with
        circles only a dozen pixels wide. Results are scaled back down to
        minimap pixels.
        """
        s = self.settings
        k = s.upscale
        big = cv2.resize(minimap, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
        grey = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
        grey = cv2.GaussianBlur(grey, (5, 5), 1.2)  # soften video compression noise

        circles = cv2.HoughCircles(
            grey, cv2.HOUGH_GRADIENT, dp=1,
            minDist=s.min_distance * k,
            param1=s.edge_strength, param2=s.circle_votes,
            minRadius=int(s.min_radius * k), maxRadius=int(s.max_radius * k),
        )
        if circles is None:
            return []
        return [(float(x) / k, float(y) / k, float(r) / k) for x, y, r in circles[0]]

    def _remove_duplicates(self, dots: list[Dot]) -> list[Dot]:
        """Merge circles that sit on the same icon.

        The teardrop shape (portrait plus pointer) sometimes gets two circle
        fits a few pixels apart. If two dots of the same team are closer than
        `merge_distance` times the bigger radius, we keep only the one whose
        inside changed most, since that one is centred on the portrait.
        Two different players standing on top of each other are a full icon
        width apart (about 2 radii), so they are not merged.
        """
        kept: list[Dot] = []
        # Strongest first, so the best circle on each icon is the one kept.
        for dot in sorted(dots, key=lambda d: d.change, reverse=True):
            is_duplicate = any(
                other.team == dot.team
                and np.hypot(other.x - dot.x, other.y - dot.y)
                < self.settings.merge_distance * max(other.radius, dot.radius)
                for other in kept
            )
            if not is_duplicate:
                kept.append(dot)
        return kept

    def _team_colour_masks(self, hsv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return two True/False images: pixels that look team-red, and team-teal."""
        s = self.settings
        hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        coloured = (sat >= s.min_saturation) & (val >= s.min_value)
        red = coloured & ((hue >= s.red_hue_low) | (hue <= s.red_hue_high))
        teal = coloured & (hue >= s.teal_hue_low) & (hue <= s.teal_hue_high)
        return red, teal

    def _pick_team(self, red_share: float, teal_share: float, min_share: float) -> str:
        """Choose the team whose colour fills more of the ring, if either reaches min_share."""
        best = max(red_share, teal_share)
        if best < min_share:
            return UNKNOWN
        return RED if red_share >= teal_share else TEAL

    def _inside(self, x: float, y: float) -> bool:
        """True if (x, y) lies inside the minimap crop."""
        h, w = self.allowed_area.shape
        return 0 <= x < w and 0 <= y < h

    @staticmethod
    def _disc_mask(shape: tuple[int, ...], x: float, y: float, r: float) -> np.ndarray:
        """Return a True/False image that is True inside a filled circle."""
        mask = np.zeros(shape[:2], np.uint8)
        cv2.circle(mask, (int(round(x)), int(round(y))), max(int(round(r)), 1), 1, thickness=-1)
        return mask.astype(bool)


def draw_dots(minimap: np.ndarray, dots: list[Dot], scale: int = 2) -> np.ndarray:
    """Return an enlarged copy of the minimap with each detection circled.

    Red circles are team red, cyan circles team teal, white circles unknown.
    The label above each circle is "R" / "T" / "?" followed by the ring
    shares, which helps when tuning the colour thresholds.
    """
    colours = {RED: (0, 0, 255), TEAL: (255, 255, 0), UNKNOWN: (255, 255, 255)}  # BGR
    letters = {RED: "R", TEAL: "T", UNKNOWN: "?"}
    out = cv2.resize(minimap, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    for d in dots:
        centre = (int(d.x * scale), int(d.y * scale))
        colour = colours[d.team]
        cv2.circle(out, centre, int((d.radius + 2) * scale), colour, 1)
        label = f"{letters[d.team]} {d.red_share:.2f}/{d.teal_share:.2f}"
        cv2.putText(out, label, (centre[0] - 22, centre[1] - int((d.radius + 4) * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, colour, 1, cv2.LINE_AA)
    return out
