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

## Current phase: Phase 0, offline prototype

Goal: prove that minimap positions can predict fights before building anything live.

- Input is a recorded match video of one map (Ascent first).
- Minimap is cropped from a fixed screen region.
- Team-coloured player dots are found with colour thresholds (no ML yet).
- Kill-feed events are detected to get the time and players of each real fight.
- A hand-tuned heuristic scores enemy pairs using distance, closing speed and position.
- A replay viewer shows predictions next to real kills.

### Running what exists so far

Built so far: finding both teams' player dots on the minimap, and reading kills from the kill feed.

```bash
.venv/bin/python -m scripts.detect_dots --times 125 175 250   # check a few moments, saves pictures
.venv/bin/python -m scripts.detect_dots --step 1              # whole clip, one frame per second
.venv/bin/python -m scripts.detect_kills                      # every kill in the clip
.venv/bin/python -m pytest                                     # run the tests
```

Results land in `data/out/`:
- `dots/dots.csv` has one row per detected player icon, and `dots/dots_<seconds>.png` shows red circles for PRX and cyan circles for LEV.
- `kills/kills.csv` has one row per kill (time, which team killed which), and `kills/kill_<seconds>.png` is that kill-feed row, so you can read the names.

**Done when** the heuristic predicts the players in upcoming kills (top-2 suggestions, 2 seconds ahead) clearly better than a naive "closest pair" baseline.

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

**Planned for later phases**

| Area | Technology |
|---|---|
| Minimap icon detector | PyTorch, Ultralytics YOLO, exported to ONNX Runtime |
| Prediction models | scikit-learn, LightGBM |
| Live screen capture | Windows Graphics Capture / DXGI, or a capture card |
| Observer overlay | FastAPI + WebSockets backend, web page (also usable as an OBS browser source) |
| Match recording | Parquet |

## Project structure (planned)

```
observer-tool/
  capture/      # frame sources: video file now, live screen later
  perception/   # minimap dot detection, kill-feed parsing, calibration
  tracking/     # per-player tracks → GameState
  maps/         # per-map data (walls, distances, sightlines)
  predict/      # heuristic now, ML models later
  policy/       # turns scores into a single suggestion
  ui/           # observer-facing overlay
  record/       # saving and replaying matches
  train/        # data generation, training, evaluation
```

Phase 0 only needs `capture`, `perception`, `tracking`, `predict` and a simple replay viewer.

## Rules

- Passive screen or video capture only. No memory reading, injection or automation of the game client.
- The observer stays in control. Any auto-switching will be opt-in.
