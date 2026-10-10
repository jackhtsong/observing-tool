# observer-tool

A tool that watches the VALORANT minimap and predicts which players are about to take the next fight, so a match observer knows whose POV to switch to, especially in chaotic moments.

It works only from what is on screen, using computer vision on the minimap, kill feed and HUD. It never reads game memory or touches the game process.

Full design: [`docs/next-fight-predictor.md`](docs/next-fight-predictor.md)

## How it works

```
video / screen  →  perception  →  tracking  →  prediction  →  observer suggestion
 (frames)          (icons, HUD)   (GameState)   (P(fight))     ("switch to Jett")
```

1. **Capture** frames from a recorded VOD (phase 0) or the observer's screen (later).
2. **Perception** finds each player's icon on the minimap and reads kill-feed events.
3. **Tracking** turns per-frame detections into smooth positions and speeds for all 10 players.
4. **Prediction** scores every pair of enemies for how likely they are to fight in the next few seconds.
5. **Policy and UI** turn those scores into one clear suggestion for the observer.

## Current phase: Phase 1, live MVP

Goal: run the whole tool live, frame by frame, and show the observer one calm suggestion.

- ✓ One shared per-frame pipeline (`live/pipeline.py`): minimap check, kill feed, spike, dots, tracker, predictor. The offline tracking uses it too.
- ✓ Suggestion with hysteresis (`policy/suggestion.py`): a new pair only takes over when it scores clearly higher (+0.1) for 0.3 s, or straight away if the current pair is gone. This cuts changes from ~89 to ~24 per minute without losing hit rate.
- ✓ Observer page (`ui/`): a local web page with the pair to watch, an "also watch" pair, the minimap with the pair drawn on, and status (live, spike, alive counts, latency). Works as an OBS browser source. Standard library only.
- ✓ A "live" input made from a VOD played in real time (`capture/realtime_video.py`), so everything can be tested without a game.
- ✓ Latency: the pipeline takes ~22 ms per frame (target: under 100 ms).
- ○ Real screen capture (needs a capture library, e.g. `mss`; not installed yet).
- ○ Fixed observer settings documented; tested in a custom game end to end.

```bash
.venv/bin/python -m scripts.run_live                        # then open http://localhost:8765/
.venv/bin/python -m scripts.run_live --video data/vods/ascent_test.mp4 --start 120 --predictor "closest pair"
```

**Done when** it runs through a custom game end to end without crashing, with latency under 100 ms.

## Phase 0: offline prototype (done)

Goal: prove that minimap positions can predict fights, on recorded matches.

**Result**, on 314 kills from 3 maps of one match (Ascent, Lotus, Breeze): this is how often the player who died was in the top 2 suggested pairs.

| Before the kill | Held angles | Closest pair | Random |
|---|---|---|---|
| 2 s | 50% | 48% | 40% |
| 5 s | 42% | 38% | 36% |

The minimap clearly beats random. "Held angles" (`predict/angles.py`: pair whoever holds an angle with the enemy closest to walking into it) is the best or tied-best predictor at every lead time and the clearest gain at 5 s, but at 2 s it is not *clearly* better than closest pair (the original exit test). Also tried and not better: walking distance + closing speed (`predict/heuristic.py`), facing alone, and a pull towards the planted spike (`predict/context.py`). Phase 0 was closed with held angles as the predictor.

What was built:
- Player dots on the minimap with team and facing (`perception/minimap_dots.py`): 80% of hand-labelled players found, 83% precision.
- Kill feed reader (`perception/kill_feed.py`), spike state (`perception/round_state.py`).
- Tracker with smoothed position, speed and facing (`tracking/player_tracker.py`).
- Map model per map with walking distances and sightlines (`maps/`), editable by hand.
- Evaluation (`scripts/evaluate_predictor.py`), replay viewer (`scripts/replay_viewer.py`), and a click tool to label players (`scripts/label_dots.py`).

### Running the offline tools

```bash
.venv/bin/python -m scripts.process_vods                      # track every video in data/vods/ not tracked yet
.venv/bin/python -m scripts.build_map_model --map lotus       # map model pictures in maps/<map>/ + a preview
.venv/bin/python -m scripts.evaluate_predictor                # all predictors, pooled over all tracked videos
.venv/bin/python -m scripts.replay_viewer                     # watch predictions next to real kills
.venv/bin/python -m scripts.track_players --save-video        # one video's tracks, plus a minimap video to check them
.venv/bin/python -m scripts.detect_dots --times 125 175 250   # check the dot detector on a few moments
.venv/bin/python -m scripts.detect_kills                      # every kill in a clip
.venv/bin/python -m scripts.label_dots                        # click tool: mark every player on 50 frames
.venv/bin/python -m scripts.score_dots                        # grade the dot detector against those marks
.venv/bin/python -m pytest                                     # run the tests
```

Results land in `data/out/`:
- `tracks/<video name>/`: `tracks.csv` (each player's position, speed and facing, 10 times a second), `deaths.csv` (where each victim died), `round_state.csv` (spike planted, and where), `tracks.mp4` with `--save-video`.
- `eval/recall.png` and `eval/per_kill.csv`: how often each predictor had the victim in its top 2 pairs, 1 to 5 seconds before each kill.
- `replay/`: clips saved by `replay_viewer --save`.
- `dots/`, `kills/`: outputs of the single-step scripts.

**Map model:** `maps/<map>/walkable.png` (where players can walk) and `walls.png` (what blocks sight) are a first guess made from the minimap. Fix mistakes by painting them in black or white in any image editor; the distance and sight tables rebuild automatically. `data/out/map_model/preview.png` shows the result.

**More VODs:** drop more 1080p videos with the same broadcast overlay into `data/vods/`, named starting with the map (e.g. `breeze_prx_lev.mp4`; anything else counts as Ascent). Run `process_vods`, then once per new map `build_map_model --map <map>`, then `evaluate_predictor`.

## Tech stack

**Phase 0**

| Area | Technology |
|---|---|
| Language | Python 3.11+ |
| Video and image processing | OpenCV, NumPy |
| Data handling and analysis | pandas |
| Replay viewer and plots | OpenCV windows, Matplotlib |
| Testing | pytest |
| Environment | `venv` + `requirements.txt` |

**Phase 1 (so far)**

| Area | Technology |
|---|---|
| Observer page | Python standard-library web server (`http.server`) + a plain HTML page polling it; usable as an OBS browser source |
| Live input | a VOD played in real time for now; screen capture to come |

**Planned for later phases**

| Area | Technology |
|---|---|
| Minimap icon detector | PyTorch, Ultralytics YOLO, exported to ONNX Runtime |
| Prediction models | scikit-learn, LightGBM |
| Live screen capture | Windows Graphics Capture / DXGI, or a capture card |
| Observer overlay (if the simple server isn't enough) | FastAPI + WebSockets backend |
| Match recording | Parquet |

## Project structure

```
observer-tool/
  capture/      # frame sources: video file, VOD played in real time, live screen later
  live/         # the per-frame pipeline that ties everything together
  perception/   # minimap dot detection, kill-feed parsing, calibration
  tracking/     # per-player tracks → GameState
  maps/         # per-map data (walls, distances, sightlines)
  predict/      # heuristic now, ML models later
  policy/       # turns scores into a single suggestion
  ui/           # observer-facing overlay
  record/       # saving and replaying matches
  scripts/      # command-line tools (run with .venv/bin/python -m scripts.<name>)
  tests/        # pytest tests
  train/        # (later) data generation, training
```

## Rules

- Passive screen or video capture only. No memory reading, injection or automation of the game client.
- The observer stays in control. Any auto-switching will be opt-in.
