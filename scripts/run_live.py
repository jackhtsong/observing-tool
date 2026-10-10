"""
run_live.py: run the observer tool live and show its suggestion in a browser.

Until screen capture is added, the "live" input is a VOD played back in real
time (capture/realtime_video.py): the tool only ever sees the frame that is
on screen now, exactly as it would with a real game. Every frame it:

  1. runs the pipeline (live/pipeline.py): dots, kill feed, spike, tracker,
     predictor,
  2. turns the ranking into one calm suggestion (policy/suggestion.py),
  3. draws the minimap with the suggested pair and publishes both to the
     observer page (ui/server.py).

Open the printed address (http://localhost:8765/) in a browser on a second
monitor, or add it to OBS as a browser source. Every 10 seconds the terminal
prints how long the pipeline takes per frame; the phase 1 target is under
100 ms.

Run it from the project root:

    .venv/bin/python -m scripts.run_live                                  # Lotus VOD from the start
    .venv/bin/python -m scripts.run_live --video data/vods/ascent_test.mp4 --start 120
    .venv/bin/python -m scripts.run_live --predictor "closest pair"
    .venv/bin/python -m scripts.run_live --speed 2                        # twice real time

Stop it with Ctrl+C.
"""

from __future__ import annotations

import argparse
import time as clock

import cv2
import numpy as np

from capture.realtime_video import RealTimeVideo
from capture.video_source import VideoSource
from live.pipeline import FrameResult, LivePipeline
from maps.map_model import load_map_for
from perception.map_background import load_or_build_background
from perception.regions import MINIMAP
from perception.teams import RED, TEAL, TEAM_NAMES
from policy.suggestion import Suggestion, SuggestionPolicy
from predict.angles import AngleHoldPredictor
from predict.heuristic import ClosestPairBaseline, HeuristicPredictor, PairScore
from ui.server import ObserverServer

FRAME_GAP = 0.1   # aim for 10 frames per second; more adds nothing at a 2-5 s horizon
SCALE = 2         # minimap enlargement on the page
COLOURS = {RED: (60, 60, 255), TEAL: (230, 220, 40)}   # BGR
PAIR = (0, 230, 255)                                     # yellow


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the observer tool live on a VOD played in real time.")
    p.add_argument("--video", default="data/vods/lotus_test.mp4")
    p.add_argument("--start", type=float, default=0.0, help="second of the video to start at")
    p.add_argument("--speed", type=float, default=1.0, help="playback speed (1 = real time)")
    p.add_argument("--predictor", choices=["angles", "closest pair", "heuristic"], default="angles")
    p.add_argument("--port", type=int, default=8765)
    return p.parse_args()


def make_predictor(name: str, video_path: str):
    """The chosen predictor; the ones that need the map get this video's map model."""
    if name == "closest pair":
        return ClosestPairBaseline()
    model = load_map_for(video_path)
    return AngleHoldPredictor(model) if name == "angles" else HeuristicPredictor(model)


def draw(result: FrameResult, suggestion: Suggestion) -> np.ndarray:
    """Enlarged minimap: all players, the suggested pair (thick) and the also-watch pair (thin)."""
    out = cv2.resize(result.minimap, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    at = lambda x, y: (int(x * SCALE), int(y * SCALE))
    if not result.live:
        return (out * 0.4).astype(np.uint8)
    for pair, thickness in ((suggestion.also, 2), (suggestion.pair, 5)):
        if pair is not None:
            cv2.line(out, at(pair.red.x, pair.red.y), at(pair.teal.x, pair.teal.y), PAIR, thickness, cv2.LINE_AA)
    for t in result.tracks:
        cv2.circle(out, at(t.x, t.y), 9, COLOURS[t.team], 2, cv2.LINE_AA)
        cv2.putText(out, t.id.split("-")[-1], (at(t.x, t.y)[0] + 11, at(t.x, t.y)[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOURS[t.team], 1, cv2.LINE_AA)
    if result.spike is not None:
        cx, cy = at(*result.spike)
        cv2.polylines(out, [np.array([(cx, cy - 12), (cx + 12, cy), (cx, cy + 12), (cx - 12, cy)])],
                      True, (0, 220, 255), 3, cv2.LINE_AA)
    return out


def pair_json(pair: PairScore | None) -> dict | None:
    """A pair as the page wants it: team names and track numbers."""
    if pair is None:
        return None
    return {"red": pair.red.id.split("-")[-1], "red_name": TEAM_NAMES[RED],
            "teal": pair.teal.id.split("-")[-1], "teal_name": TEAM_NAMES[TEAL]}


def main() -> None:
    args = parse_args()
    server = ObserverServer(args.port)
    server.start()
    print(f"Observer page: {server.url}   (Ctrl+C to stop)", flush=True)

    with VideoSource(args.video) as video:
        background = load_or_build_background(video, MINIMAP)
        pipeline = LivePipeline(background, predictor=make_predictor(args.predictor, args.video))
        policy = SuggestionPolicy()
        source = RealTimeVideo(video, start=args.start, speed=args.speed)

        timings: list[float] = []
        last_report = clock.perf_counter()
        try:
            while True:
                frame_started = clock.perf_counter()
                latest = source.latest()
                if latest is None:
                    print("End of video.")
                    break
                t, frame = latest

                result = pipeline.step(t, frame)
                if result.round_reset:
                    policy.reset()
                suggestion = policy.update(t, result.ranked) if result.live else Suggestion(None, None, None)
                timings.append(result.total_ms)

                minutes, seconds = divmod(t, 60)
                state = {
                    "live": result.live,
                    "watch": pair_json(suggestion.pair),
                    "also": pair_json(suggestion.also),
                    "held_for": (t - suggestion.since) if suggestion.since is not None else 0.0,
                    "planted": result.planted,
                    "alive_red": sum(tr.team == RED for tr in result.tracks),
                    "alive_teal": sum(tr.team == TEAL for tr in result.tracks),
                    "latency_ms": result.total_ms,
                    "clock": f"{int(minutes)}:{seconds:04.1f}",
                    "predictor": args.predictor,
                }
                ok, jpeg = cv2.imencode(".jpg", draw(result, suggestion), [cv2.IMWRITE_JPEG_QUALITY, 80])
                server.publish(state, jpeg.tobytes() if ok else None)

                # Every 10 s, report pipeline timings (the phase 1 target is < 100 ms).
                if clock.perf_counter() - last_report >= 10 and timings:
                    ordered = sorted(timings)
                    print(f"{state['clock']}  pipeline per frame: median {ordered[len(ordered) // 2]:.0f} ms, "
                          f"worst {ordered[-1]:.0f} ms  ({len(timings)} frames)", flush=True)
                    timings.clear()
                    last_report = clock.perf_counter()

                # Wait for the next frame slot (10 per second).
                spare = FRAME_GAP - (clock.perf_counter() - frame_started)
                if spare > 0:
                    clock.sleep(spare)
        except KeyboardInterrupt:
            print("Stopped.")
        finally:
            server.stop()


if __name__ == "__main__":
    main()
