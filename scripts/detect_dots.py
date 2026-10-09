"""
detect_dots.py: run the minimap player-dot detector on a VOD and save the results.

This is the first runnable step of phase 0. For each sampled moment of the
video it:
  1. cuts out the minimap,
  2. finds every player icon and its team (perception/minimap_dots.py),
  3. writes one CSV row per icon, and
  4. optionally saves a picture with the detections circled, so you can
     check by eye whether the detector got it right.

Run it from the project root:

    .venv/bin/python -m scripts.detect_dots                      # whole clip, 2 frames/s
    .venv/bin/python -m scripts.detect_dots --times 125 175 250  # just these seconds, with pictures
    .venv/bin/python -m scripts.detect_dots --start 120 --end 180 --step 0.5 --images

Outputs go to data/out/dots/ (data/ is git-ignored):
    dots.csv          time, x, y, radius, team, red_share, teal_share, change
    dots_<time>.png   annotated minimap, when pictures are switched on
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2

from capture.video_source import VideoSource
from perception.map_background import floor_mask, load_or_build_background
from perception.minimap_dots import DotDetector, draw_dots
from perception.regions import MINIMAP


def parse_args() -> argparse.Namespace:
    """Read the command-line options (run with --help to see them)."""
    p = argparse.ArgumentParser(description="Detect player dots on the minimap.")
    p.add_argument("--video", default="data/vods/ascent_test.mp4", help="path to the VOD")
    p.add_argument("--out", default="data/out/dots", help="folder for the CSV and pictures")
    p.add_argument("--times", type=float, nargs="*", help="exact seconds to check (turns on pictures)")
    p.add_argument("--start", type=float, default=0.0, help="first second to process")
    p.add_argument("--end", type=float, default=None, help="last second to process (default: end of video)")
    p.add_argument("--step", type=float, default=0.5, help="seconds between processed frames")
    p.add_argument("--images", action="store_true", help="also save an annotated picture per frame")
    return p.parse_args()


def main() -> None:
    """Load the video, build the detector, run it over the chosen frames."""
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with VideoSource(args.video) as video:
        # The empty-map background is built once per video and cached.
        background = load_or_build_background(video, MINIMAP)
        cv2.imwrite(str(out_dir / "background.png"), background)
        detector = DotDetector(background, floor_mask(background))

        # Either the exact --times given, or every --step seconds in a range.
        if args.times:
            moments = ((t, video.frame_at(t)) for t in args.times)
            save_images = True
        else:
            moments = video.frames(args.start, args.end, args.step)
            save_images = args.images

        csv_path = out_dir / "dots.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "x", "y", "radius", "team", "red_share", "teal_share", "change"])

            for t, frame in moments:
                if frame is None:
                    print(f"{t:7.1f}s  no frame (past end of video?)")
                    continue
                minimap = MINIMAP.crop(frame)

                # Replays, round-end screens and rotated minimaps are skipped:
                # positions on them do not mean the same thing.
                if not detector.is_standard_layout(minimap):
                    print(f"{t:7.1f}s  skipped (minimap not in its usual layout)")
                    continue

                dots = detector.detect(minimap)

                for d in dots:
                    writer.writerow([f"{t:.2f}", f"{d.x:.1f}", f"{d.y:.1f}", f"{d.radius:.1f}", d.team,
                                     f"{d.red_share:.3f}", f"{d.teal_share:.3f}", f"{d.change:.1f}"])

                # One summary line per frame, e.g. "125.0s  red 5  teal 4  unknown 1"
                counts = Counter(d.team for d in dots)
                print(f"{t:7.1f}s  red {counts['red']}  teal {counts['teal']}  unknown {counts['unknown']}")

                if save_images:
                    cv2.imwrite(str(out_dir / f"dots_{t:07.2f}.png"), draw_dots(minimap, dots))

    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
