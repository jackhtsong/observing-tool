"""
detect_kills.py: list every kill in a VOD by watching the kill feed.

These kills are the "ground truth" for phase 0: the predictor is judged on
whether it saw each of these fights coming. For every kill we record the
time, which team killed which, and save a picture of the kill-feed row so a
person can read the player names and check the result.

Kills shown during replays and round-end screens are ignored: we only
record a kill while the minimap is in its normal live-round layout (the same
check the dot detector uses).

Run it from the project root:

    .venv/bin/python -m scripts.detect_kills                 # whole clip
    .venv/bin/python -m scripts.detect_kills --start 20 --end 80

Outputs go to data/out/kills/:
    kills.csv            time, killer_team, victim_team, killer, victim, slot, image
    kill_<time>.png      the kill-feed row for each kill
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from capture.video_source import VideoSource
from perception.kill_feed import KillFeedTracker
from perception.map_background import floor_mask, load_or_build_background
from perception.minimap_dots import DotDetector
from perception.regions import KILL_FEED, MINIMAP
from perception.teams import TEAM_NAMES


def parse_args() -> argparse.Namespace:
    """Read the command-line options (run with --help to see them)."""
    p = argparse.ArgumentParser(description="Find kills in the kill feed.")
    p.add_argument("--video", default="data/vods/ascent_test.mp4", help="path to the VOD")
    p.add_argument("--out", default="data/out/kills", help="folder for the CSV and row pictures")
    p.add_argument("--start", type=float, default=0.0, help="first second to process")
    p.add_argument("--end", type=float, default=None, help="last second to process (default: end of video)")
    p.add_argument("--step", type=float, default=0.1,
                   help="seconds between frames; 0.1 means kill times are accurate to 0.1 s")
    return p.parse_args()


def main() -> None:
    """Walk through the video frame by frame and write out every new kill."""
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with VideoSource(args.video) as video:
        # Only used for its "is this the normal minimap?" check.
        background = load_or_build_background(video, MINIMAP)
        minimap_check = DotDetector(background, floor_mask(background))
        tracker = KillFeedTracker()

        csv_path = out_dir / "kills.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "killer_team", "victim_team", "killer", "victim", "slot", "image"])

            kill_count = 0
            for t, frame in video.frames(args.start, args.end, args.step):
                live = minimap_check.is_standard_layout(MINIMAP.crop(frame))

                # During replays we still feed the tracker (so their rows are
                # known and not reported later) but don't record kills.
                for kill in tracker.update(t, KILL_FEED.crop(frame), record=live):
                    kill_count += 1
                    image_name = f"kill_{kill.time:07.2f}.png"
                    cv2.imwrite(str(out_dir / image_name), kill.image)
                    killer, victim = TEAM_NAMES[kill.killer_team], TEAM_NAMES[kill.victim_team]
                    writer.writerow([f"{kill.time:.2f}", kill.killer_team, kill.victim_team,
                                     killer, victim, kill.slot, image_name])
                    print(f"{kill.time:7.2f}s  {killer} killed {victim}  (slot {kill.slot})")

    print(f"{kill_count} kills. Wrote {csv_path}")


if __name__ == "__main__":
    main()
