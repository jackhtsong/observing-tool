"""
track_players.py: follow every player through the VOD and record their tracks.

This combines the pieces built so far, frame by frame:
  1. skip replays and round-end screens (minimap layout check),
  2. find the player dots on the minimap,
  3. read the kill feed for new kills,
  4. update the tracker with the dots, and tell it about each death.

The tracker is reset whenever the minimap leaves its live layout for more
than a second, which happens between every round, so each round starts
with fresh tracks.

Run it from the project root:

    .venv/bin/python -m scripts.track_players                    # whole clip
    .venv/bin/python -m scripts.track_players --start 108 --end 180 --save-video
    .venv/bin/python -m scripts.track_players --video data/vods/other_match.mp4

Outputs go to data/out/tracks/<video name>/, e.g. data/out/tracks/ascent_test/:
    tracks.csv    time, track, team, x, y, vx, vy, speed, seen   (one row per track per frame)
    deaths.csv    time, track, team, x, y                        (one row per death matched to a track)
    background.png  the empty minimap of this video (used by the evaluation)
    tracks.mp4    with --save-video: the minimap with tracks drawn on, to check by eye
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from capture.video_source import VideoSource
from perception.kill_feed import KillFeedTracker
from perception.map_background import floor_mask, load_or_build_background
from perception.minimap_dots import DotDetector
from perception.regions import KILL_FEED, MINIMAP
from perception.teams import RED, TEAM_NAMES
from tracking.player_tracker import PlayerTracker, Track

# How long the minimap must be out of its live layout before we treat it as
# a round change and reset the tracker. Short dips (a flash on screen) are
# ignored.
RESET_AFTER = 1.0


def parse_args() -> argparse.Namespace:
    """Read the command-line options (run with --help to see them)."""
    p = argparse.ArgumentParser(description="Track players on the minimap.")
    p.add_argument("--video", default="data/vods/ascent_test.mp4", help="path to the VOD")
    p.add_argument("--out", default="data/out/tracks", help="parent folder; results go in a subfolder named after the video")
    p.add_argument("--start", type=float, default=0.0, help="first second to process")
    p.add_argument("--end", type=float, default=None, help="last second to process (default: end of video)")
    p.add_argument("--step", type=float, default=0.1, help="seconds between frames (0.1 = 10 per second)")
    p.add_argument("--assist", action="store_true",
                   help="let the tracker help the detector look for expected players (experimental)")
    p.add_argument("--save-video", "--video-out", dest="save_video", action="store_true",
                   help="also save tracks.mp4 with the tracks drawn on the minimap")
    return p.parse_args()


def draw_tracks(minimap: np.ndarray, tracks: list[Track], trails: dict[str, list], now: float,
                scale: int = 2) -> np.ndarray:
    """Return an enlarged minimap with each track drawn on it.

    Each track gets a circle (solid outline if seen this frame, thin if it
    is coasting), its id, an arrow for its speed and direction (where it
    will be in one second), and a short trail of where it has been.
    """
    colours = {RED: (0, 0, 255), "teal": (255, 255, 0)}  # BGR
    out = cv2.resize(minimap, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    for t in tracks:
        colour = colours[t.team]
        trail = [(int(x * scale), int(y * scale)) for x, y in trails.get(t.id, [])]
        for a, b in zip(trail, trail[1:]):
            cv2.line(out, a, b, colour, 1, cv2.LINE_AA)
        centre = (int(t.x * scale), int(t.y * scale))
        seen = now - t.last_seen < 1e-6
        cv2.circle(out, centre, 9 * scale // 2, colour, 2 if seen else 1, cv2.LINE_AA)
        arrow_end = (int((t.x + t.vx) * scale), int((t.y + t.vy) * scale))
        cv2.arrowedLine(out, centre, arrow_end, colour, 1, cv2.LINE_AA, tipLength=0.3)
        cv2.putText(out, t.id.split("-")[1], (centre[0] + 10, centre[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    cv2.putText(out, f"{now:6.1f}s", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def track_video(video_path: str, out_dir: Path, start: float = 0.0, end: float | None = None,
                step: float = 0.1, save_video: bool = False, assist: bool = False) -> None:
    """Run detection, kill reading and tracking over one video and save the results.

    Also saves the video's empty-map background as background.png, so later
    steps (like the evaluation) have the map of this video without opening it.

    assist: let the tracker tell the detector where to look harder for
    players it expects but didn't find (DotDetector.find_expected). Off by
    default: on the labelled frames it found 5% more players but added more
    false ones, kept dead players' tracks alive on their death markers, and
    made the predictor evaluation slightly worse.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    with VideoSource(video_path) as video:
        background = load_or_build_background(video, MINIMAP)
        cv2.imwrite(str(out_dir / "background.png"), background)
        detector = DotDetector(background, floor_mask(background))
        kill_feed = KillFeedTracker()
        tracker = PlayerTracker()

        writer_video = None
        trails: dict[str, list] = {}   # track id -> recent positions, for drawing
        not_live_since = None          # when the minimap last left its live layout

        with open(out_dir / "tracks.csv", "w", newline="") as f:
            rows = csv.writer(f)
            rows.writerow(["time", "track", "team", "x", "y", "vx", "vy", "speed", "seen"])

            for t, frame in video.frames(start, end, step):
                minimap = MINIMAP.crop(frame)
                live = detector.is_standard_layout(minimap)

                # Kills: read every frame, but only record them while live.
                for kill in kill_feed.update(t, KILL_FEED.crop(frame), record=live):
                    tracker.report_death(kill.time, kill.victim_team)
                    print(f"{t:7.1f}s  kill: {TEAM_NAMES[kill.killer_team]} killed {TEAM_NAMES[kill.victim_team]}")

                if not live:
                    # Reset once the break has lasted long enough to be a round change.
                    not_live_since = t if not_live_since is None else not_live_since
                    if t - not_live_since >= RESET_AFTER and tracker.tracks:
                        tracker.reset()
                        trails.clear()
                        print(f"{t:7.1f}s  minimap left live layout: tracker reset")
                    continue
                not_live_since = None

                dots = detector.detect(minimap)
                if assist:
                    dots += detector.find_expected(minimap, tracker.expected_missing(t, dots), dots)
                tracks = tracker.update(t, dots)
                for tr in tracks:
                    seen = int(t - tr.last_seen < 1e-6)
                    rows.writerow([f"{t:.2f}", tr.id, tr.team, f"{tr.x:.1f}", f"{tr.y:.1f}",
                                   f"{tr.vx:.1f}", f"{tr.vy:.1f}", f"{tr.speed:.1f}", seen])
                    trails.setdefault(tr.id, []).append((tr.x, tr.y))
                    trails[tr.id] = trails[tr.id][-30:]   # keep the last 3 s at 10 fps

                if save_video:
                    picture = draw_tracks(minimap, tracks, trails, t)
                    if writer_video is None:
                        h, w = picture.shape[:2]
                        fps = 1.0 / step
                        writer_video = cv2.VideoWriter(str(out_dir / "tracks.mp4"),
                                                       cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    writer_video.write(picture)

        if writer_video is not None:
            writer_video.release()

    # Deaths the tracker managed to tie to a track.
    with open(out_dir / "deaths.csv", "w", newline="") as f:
        rows = csv.writer(f)
        rows.writerow(["time", "track", "team", "x", "y"])
        for d in tracker.deaths:
            rows.writerow([f"{d.time:.2f}", d.track_id, d.team, f"{d.x:.1f}", f"{d.y:.1f}"])

    print(f"{len(tracker.deaths)} deaths matched to tracks. Wrote results to {out_dir}/")


def main() -> None:
    """Read the command-line options and track one video."""
    args = parse_args()
    # Each video gets its own folder, named after the video file.
    out_dir = Path(args.out) / Path(args.video).stem
    track_video(args.video, out_dir, args.start, args.end, args.step, args.save_video, assist=args.assist)


if __name__ == "__main__":
    main()
