"""
replay_viewer.py: watch a VOD with the predictor's suggestions drawn next to the real kills.

This is the phase 0 demo, and the quickest way to see *why* a prediction
missed. For every moment of the video it shows:

  left    the minimap, enlarged, with
            - every tracked player (red / teal circles with their track number)
            - the predictor's #1 pair (thick yellow line) and #2 pair (thin line)
            - each kill as a big X on the victim's last spot, for 2 s
  right   a side panel with
            - the broadcast picture (what the observer is showing)
            - the top pairs and their scores
            - a countdown to the next kill, and whether the victim is in the
              top 2 right now (green = the tool would have the observer there)

It reads the results of scripts/track_players.py (or process_vods.py), so
run that first for the video.

Controls (interactive window)
-----------------------------
    space          pause / play
    d / a          forward / back 5 seconds
    . / ,          one step forward / back (0.1 s), handy while paused
    n / p          jump to 3 s before the next / previous kill
    1 / 2          predictor: 1 = heuristic, 2 = closest pair
    q / esc        quit

Run it from the project root:

    .venv/bin/python -m scripts.replay_viewer                                   # first tracked video
    .venv/bin/python -m scripts.replay_viewer --video data/vods/ascent_test_extended.mp4
    .venv/bin/python -m scripts.replay_viewer --start 240 --end 275 --save      # write an mp4 instead of a window

With --save the result goes to data/out/replay/<video>_<start>-<end>.mp4.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from pathlib import Path

import cv2
import numpy as np

from capture.video_source import VideoSource
from maps.map_model import MapModel
from perception.regions import MINIMAP
from perception.teams import RED, TEAM_NAMES
from predict.heuristic import ClosestPairBaseline, HeuristicPredictor, PairScore
from record.track_log import TrackLog, load_deaths, load_tracks
from tracking.player_tracker import Death, Track

STEP = 0.1                 # seconds between shown frames (the tracks were made at 10 per second)
SCALE = 2                  # minimap enlargement
PANEL_WIDTH = 440          # width of the side panel in pixels
KILL_SHOW_FOR = 2.0        # seconds a kill's X stays on the minimap
LEAD = 2.0                 # the "2 seconds before the kill" the README's exit test uses

# BGR colours
TEAM_COLOURS = {RED: (60, 60, 255), "teal": (230, 220, 40)}
PAIR_COLOUR = (0, 230, 255)        # yellow
GOOD = (80, 210, 80)               # green
BAD = (80, 80, 230)                # red
TEXT = (235, 235, 235)
DIM = (150, 150, 150)
PANEL_BG = (32, 32, 32)


class ReplayRenderer:
    """Draws one replay screen for a given time. Holds no window, so it can be tested."""

    def __init__(self, log: TrackLog, deaths: list[Death], predictors: dict[str, object]):
        """
        log:         the tracked frames (record.track_log.load_tracks)
        deaths:      kills tied to tracks (record.track_log.load_deaths)
        predictors:  name -> object with rank_pairs(tracks), e.g. HeuristicPredictor
        """
        self.log = log
        self.deaths = deaths
        self.death_times = [d.time for d in deaths]
        self.predictors = predictors
        self.predictor_name = next(iter(predictors))

    # ----- facts for one moment -----

    def ranked_pairs(self, tracks: list[Track]) -> list[PairScore]:
        """The current predictor's pairs for these tracks, best first."""
        return self.predictors[self.predictor_name].rank_pairs(tracks)

    def next_death(self, time: float) -> Death | None:
        """The first kill strictly after `time`, if any."""
        i = bisect_right(self.death_times, time)
        return self.deaths[i] if i < len(self.deaths) else None

    def recent_deaths(self, time: float) -> list[Death]:
        """Kills in the last KILL_SHOW_FOR seconds (drawn as X's)."""
        lo = bisect_left(self.death_times, time - KILL_SHOW_FOR)
        hi = bisect_right(self.death_times, time)
        return self.deaths[lo:hi]

    # ----- drawing -----

    def render(self, time: float, frame: np.ndarray) -> np.ndarray:
        """The full screen for `time`: enlarged minimap on the left, panel on the right."""
        tracks = self.log.tracks_at(time)
        ranked = self.ranked_pairs(tracks) if tracks else []
        minimap = self._draw_minimap(MINIMAP.crop(frame), tracks, ranked[:2], time)
        panel = self._draw_panel(frame, tracks, ranked, time, height=minimap.shape[0])
        return np.hstack([minimap, panel])

    def _draw_minimap(self, minimap: np.ndarray, tracks: list[Track] | None,
                      top: list[PairScore], time: float) -> np.ndarray:
        """Enlarged minimap with players, the top 2 pairs and recent kills."""
        out = cv2.resize(minimap, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
        at = lambda x, y: (int(x * SCALE), int(y * SCALE))

        if tracks is None:
            # A replay or round break: nothing was tracked here.
            out = (out * 0.4).astype(np.uint8)
            cv2.putText(out, "not live (replay / round break)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT, 2, cv2.LINE_AA)
            return out

        # Top pairs first, so the player circles are drawn on top of the lines.
        for rank, pair in reversed(list(enumerate(top))):
            thickness = 4 if rank == 0 else 2
            cv2.line(out, at(pair.red.x, pair.red.y), at(pair.teal.x, pair.teal.y),
                     PAIR_COLOUR, thickness, cv2.LINE_AA)

        for t in tracks:
            colour = TEAM_COLOURS[t.team]
            centre = at(t.x, t.y)
            cv2.circle(out, centre, 9 * SCALE // 2, colour, 2, cv2.LINE_AA)
            cv2.putText(out, t.id.split("-")[-1], (centre[0] + 11, centre[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)

        for d in self.recent_deaths(time):
            centre = at(d.x, d.y)
            size = 12
            for sign in (1, -1):
                cv2.line(out, (centre[0] - size, centre[1] - sign * size),
                         (centre[0] + size, centre[1] + sign * size), (255, 255, 255), 5, cv2.LINE_AA)
                cv2.line(out, (centre[0] - size, centre[1] - sign * size),
                         (centre[0] + size, centre[1] + sign * size), TEAM_COLOURS[d.team], 3, cv2.LINE_AA)
            cv2.putText(out, "KILL", (centre[0] + 14, centre[1] + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        return out

    def _draw_panel(self, frame: np.ndarray, tracks: list[Track] | None, ranked: list[PairScore],
                    time: float, height: int) -> np.ndarray:
        """Side panel: broadcast thumbnail, time, top pairs, next-kill countdown."""
        panel = np.full((height, PANEL_WIDTH, 3), PANEL_BG, np.uint8)
        pad = 12
        y = pad

        # Broadcast thumbnail (16:9) at the top.
        thumb_w = PANEL_WIDTH - 2 * pad
        thumb_h = thumb_w * 9 // 16
        panel[y:y + thumb_h, pad:pad + thumb_w] = cv2.resize(frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        y += thumb_h + 28

        def write(text: str, colour=TEXT, size: float = 0.55, bold: int = 1, gap: int = 26):
            nonlocal y
            cv2.putText(panel, text, (pad, y), cv2.FONT_HERSHEY_SIMPLEX, size, colour, bold, cv2.LINE_AA)
            y += gap

        minutes, seconds = divmod(time, 60)
        write(f"{int(minutes)}:{seconds:04.1f}", size=0.8, bold=2, gap=30)
        write(f"predictor: {self.predictor_name}   (1 / 2 to switch)", DIM, 0.45)
        y += 6

        # Next kill and whether the victim is in the top 2 right now.
        upcoming = self.next_death(time)
        if upcoming is not None:
            wait = upcoming.time - time
            in_top2 = any(p.involves(upcoming.track_id) for p in ranked[:2])
            victim_team = TEAM_NAMES.get(upcoming.team, upcoming.team)
            write(f"next kill in {wait:4.1f}s: {victim_team} {upcoming.track_id} dies",
                  TEXT, 0.55, 1)
            if tracks is None:
                write("  (not live now)", DIM, 0.5)
            elif in_top2:
                write("  victim IS in the top 2 now", GOOD, 0.55, 2)
            else:
                write("  victim is NOT in the top 2 now", BAD, 0.55, 2)
            if wait <= LEAD:
                write(f"  within {LEAD:.0f}s of the kill: this is what the test scores", DIM, 0.45)
        else:
            write("no more kills in this video", DIM)
        y += 8

        # The top pairs, best first.
        write("top pairs (most likely to fight next):", DIM, 0.5)
        for rank, pair in enumerate(ranked[:5], start=1):
            colour = PAIR_COLOUR if rank <= 2 else TEXT
            extra = ""
            if self.predictor_name == "heuristic":
                extra = f"  walk {pair.walking_distance:3.0f}px  closing {pair.closing_speed:+4.0f}"
            write(f"{rank}. {pair.red.id} vs {pair.teal.id}   {pair.score:+.2f}{extra}", colour, 0.45, 1, 22)
        if not ranked:
            write("(no pairs)", DIM, 0.5)

        # Controls reminder at the bottom.
        cv2.putText(panel, "space pause  a/d -/+5s  ,/. step  n/p next/prev kill  q quit",
                    (pad, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, DIM, 1, cv2.LINE_AA)
        return panel


def build_renderer(video_path: str) -> ReplayRenderer:
    """Load the tracks, deaths and map model for a video and set up both predictors."""
    folder = Path("data/out/tracks") / Path(video_path).stem
    if not (folder / "tracks.csv").exists():
        raise SystemExit(f"No tracks for {video_path}. Run: .venv/bin/python -m scripts.track_players --video {video_path}")
    log = load_tracks(str(folder / "tracks.csv"))
    deaths = load_deaths(str(folder / "deaths.csv"))
    predictors = {"heuristic": HeuristicPredictor(MapModel.load("maps/ascent")),
                  "closest pair": ClosestPairBaseline()}
    return ReplayRenderer(log, deaths, predictors)


def save_clip(video: VideoSource, renderer: ReplayRenderer, start: float, end: float) -> Path:
    """Render start..end to an mp4 at 10 frames per second and return its path."""
    out_dir = Path("data/out/replay")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{Path(video.path).stem}_{start:.0f}-{end:.0f}.mp4"
    writer = None
    for t, frame in video.frames(start, end, STEP):
        picture = renderer.render(t, frame)
        if writer is None:
            h, w = picture.shape[:2]
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 1 / STEP, (w, h))
        writer.write(picture)
    if writer is not None:
        writer.release()
    return path


def play(video: VideoSource, renderer: ReplayRenderer, start: float, end: float) -> None:
    """Interactive window: plays at real speed (10 shown frames per second) with the controls above."""
    window = "Replay viewer"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    names = list(renderer.predictors)
    t, paused = start, False
    cached_time, cached_frame = None, None

    while True:
        # Only decode a new frame when the time changed.
        if cached_time is None or abs(cached_time - t) > 1e-6:
            cached_frame = video.frame_at(t)
            cached_time = t
        if cached_frame is None:
            break
        cv2.imshow(window, renderer.render(t, cached_frame))

        key = cv2.waitKey(int(STEP * 1000)) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            paused = not paused
        elif key == ord("d"):
            t = min(t + 5, end)
        elif key == ord("a"):
            t = max(t - 5, start)
        elif key == ord("."):
            t = min(t + STEP, end)
        elif key == ord(","):
            t = max(t - STEP, start)
        elif key in (ord("n"), ord("p")):
            t = jump_to_kill(renderer, t, forward=key == ord("n"), start=start, end=end)
            paused = True   # stop there, so the moment before the kill can be studied
        elif key in (ord("1"), ord("2")):
            index = key - ord("1")
            if index < len(names):
                renderer.predictor_name = names[index]
        elif not paused:
            t += STEP
            if t > end:
                break
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break   # window closed with its red button
    cv2.destroyAllWindows()


def jump_to_kill(renderer: ReplayRenderer, t: float, forward: bool, start: float, end: float) -> float:
    """Time 3 s before the next (or previous) kill, or t unchanged if there is none."""
    # "Now" is treated as 3 s ahead, so pressing n twice moves on to the following kill.
    reference = t + 3.0
    if forward:
        later = [d.time for d in renderer.deaths if d.time > reference + 1e-6]
        target = later[0] if later else None
    else:
        earlier = [d.time for d in renderer.deaths if d.time < reference - 1e-6]
        target = earlier[-1] if earlier else None
    if target is None:
        return t
    return float(np.clip(target - 3.0, start, end))


def main() -> None:
    p = argparse.ArgumentParser(description="Watch predictions next to real kills.")
    p.add_argument("--video", default=None, help="VOD to replay (default: the first tracked one)")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float, default=None)
    p.add_argument("--predictor", choices=["heuristic", "closest pair"], default="heuristic")
    p.add_argument("--save", action="store_true", help="write an mp4 instead of opening a window")
    args = p.parse_args()

    if args.video is None:
        tracked = sorted(f.name for f in Path("data/out/tracks").iterdir() if (f / "tracks.csv").exists())
        args.video = f"data/vods/{tracked[0]}.mp4"

    renderer = build_renderer(args.video)
    renderer.predictor_name = args.predictor
    with VideoSource(args.video) as video:
        end = video.duration if args.end is None else min(args.end, video.duration)
        if args.save:
            print(f"Wrote {save_clip(video, renderer, args.start, end)}")
        else:
            play(video, renderer, args.start, end)


if __name__ == "__main__":
    main()
