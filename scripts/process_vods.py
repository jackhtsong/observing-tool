"""
process_vods.py: track every VOD in data/vods/ that hasn't been tracked yet.

Drop new match videos into data/vods/ and run this. For each video it runs
the same steps as scripts/track_players.py (dots, kill feed, tracker) and
writes the results to data/out/tracks/<video name>/. Videos that already
have a deaths.csv there are skipped, so re-running only does the new ones.
Then run scripts/evaluate_predictor.py to score the predictor on all of
them together.

Requirements for a video (the pipeline's fixed screen positions assume them):
  - 1920x1080, the same broadcast overlay as the first clip
    (minimap top-left, kill feed top-right)
  - map Ascent (the map geometry is built per video from its own minimap,
    but only Ascent has been checked so far)

Run it from the project root:

    .venv/bin/python -m scripts.process_vods
    .venv/bin/python -m scripts.process_vods --redo     # track every video again
"""

from __future__ import annotations

import argparse
from pathlib import Path

from capture.video_source import VideoSource
from scripts.track_players import track_video

VOD_DIR = Path("data/vods")
OUT_DIR = Path("data/out/tracks")
VIDEO_TYPES = {".mp4", ".mkv", ".mov"}


def main() -> None:
    """Find the videos, check their size, and track the ones not done yet."""
    p = argparse.ArgumentParser(description="Track every VOD in data/vods/.")
    p.add_argument("--redo", action="store_true", help="track videos again even if already done")
    args = p.parse_args()

    videos = sorted(f for f in VOD_DIR.iterdir() if f.suffix.lower() in VIDEO_TYPES)
    if not videos:
        print(f"No videos found in {VOD_DIR}/")
        return

    for path in videos:
        out_dir = OUT_DIR / path.stem
        if (out_dir / "deaths.csv").exists() and not args.redo:
            print(f"{path.name}: already tracked, skipping")
            continue

        # Screen positions are fixed for 1080p, so other sizes would read
        # the wrong parts of the screen.
        with VideoSource(str(path)) as video:
            size = (video.width, video.height)
        if size != (1920, 1080):
            print(f"{path.name}: {size[0]}x{size[1]}, not 1920x1080, skipping")
            continue

        print(f"{path.name}: tracking...")
        track_video(str(path), out_dir)


if __name__ == "__main__":
    main()
