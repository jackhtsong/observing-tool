"""
score_dots.py: grade the dot detector against the players you clicked.

Runs the detector on every frame marked done in data/labels/dots.json and
compares its dots with your marks (see perception/dot_labels.py for the
matching rule). It reports:

    found       share of your marked players the detector found, right team
                (this is "recall": higher = fewer missed players)
    precision   share of the detector's dots that are real players, right team
                (higher = fewer false alarms)
    wrong team  dots on a real player but with the other team's colour

Re-run it after every detector change: if found goes up without precision
dropping, the change helped.

Run it from the project root:

    .venv/bin/python -m scripts.score_dots
    .venv/bin/python -m scripts.score_dots --pictures    # also save a picture per frame

With --pictures, data/out/score/ gets one image per frame: your marks as
filled dots, detector dots as circles, and missed players ringed in white.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from capture.video_source import VideoSource
from perception.dot_labels import load_labels, score_frame
from perception.map_background import floor_mask, load_or_build_background
from perception.minimap_dots import DotDetector, draw_dots
from perception.regions import MINIMAP
from perception.teams import RED

OUT_DIR = Path("data/out/score")


def main() -> None:
    """Score the detector on every finished labelled frame and print the totals."""
    p = argparse.ArgumentParser(description="Grade the dot detector against hand labels.")
    p.add_argument("--pictures", action="store_true", help="save a comparison picture per frame")
    args = p.parse_args()

    frames = [f for f in load_labels() if f.done]
    if not frames:
        print("No finished labels yet. Run scripts.label_dots first.")
        return
    if args.pictures:
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    totals = {"found": 0, "missed": 0, "extra": 0, "wrong_team": 0}
    per_team_found = {RED: [0, 0], "teal": [0, 0]}   # team -> [found, labelled]
    videos: dict[str, tuple[VideoSource, DotDetector]] = {}

    try:
        for frame in frames:
            if frame.video not in videos:
                video = VideoSource(frame.video)
                background = load_or_build_background(video, MINIMAP)
                videos[frame.video] = (video, DotDetector(background, floor_mask(background)))
            video, detector = videos[frame.video]

            minimap = MINIMAP.crop(video.frame_at(frame.time))
            dots = detector.detect(minimap)
            score = score_frame(frame.players, dots)
            for key in totals:
                totals[key] += getattr(score, key)

            # Per-team recall: score each team's labels against that team's dots.
            for team in per_team_found:
                team_labels = [pl for pl in frame.players if pl.team == team]
                team_score = score_frame(team_labels, [d for d in dots if d.team == team])
                per_team_found[team][0] += team_score.found
                per_team_found[team][1] += len(team_labels)

            if args.pictures:
                save_picture(minimap, frame, dots)
    finally:
        for video, _ in videos.values():
            video.close()

    labelled = totals["found"] + totals["missed"] + totals["wrong_team"]
    detected = totals["found"] + totals["extra"] + totals["wrong_team"]
    print(f"{len(frames)} frames, {labelled} labelled players, {detected} detector dots\n")
    print(f"  found       {totals['found'] / max(labelled, 1):6.0%}   ({totals['found']} of {labelled} players)")
    print(f"  precision   {totals['found'] / max(detected, 1):6.0%}   ({totals['extra']} false alarms)")
    print(f"  wrong team  {totals['wrong_team']:6d}")
    for team, (found, total) in per_team_found.items():
        print(f"  found {team:5s} {found / max(total, 1):6.0%}   ({found} of {total})")


def save_picture(minimap: np.ndarray, frame, dots) -> None:
    """Save the detector's dots and the hand labels drawn on one picture."""
    picture = draw_dots(minimap, dots, scale=2)
    for pl in frame.players:
        centre = (int(pl.x * 2), int(pl.y * 2))
        colour = (40, 40, 255) if pl.team == RED else (255, 230, 0)
        cv2.circle(picture, centre, 4, colour, -1, cv2.LINE_AA)
        # Ring the label in white if no detector dot is within 6 px of it.
        if not any(np.hypot(d.x - pl.x, d.y - pl.y) <= 6 for d in dots):
            cv2.circle(picture, centre, 18, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(OUT_DIR / f"{Path(frame.video).stem}_{frame.time:07.2f}.png"), picture)


if __name__ == "__main__":
    main()
