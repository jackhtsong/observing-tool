# Next-Fight Predictor for VALORANT Observers — Design

**Goal:** watch the minimap in real time, estimate which players are about to fight in the next few seconds, and tell the observer whose POV to be on, especially when several things are happening at once.

---

## 1. Key constraints that shape the design

| Constraint | Consequence |
|---|---|
| There is no public live game-state API for VALORANT positions (inferred from Riot's public API offerings; worth confirming with Riot if this is for a sanctioned tournament, since esports partners may get a data feed). | Read the game the way a human does: computer vision on the observer's screen. |
| Vanguard and Riot's ToS forbid memory reading and injection. | Only passive screen/video capture. Never touch the game process. |
| An observer needs about 300 to 800 ms to react and press a key. | Predictions must lead events by **2 to 5 s**; pipeline latency under ~100 ms is plenty. |
| The observer client shows all 10 players on the minimap. | Full information, unlike a player's minimap, so position-based prediction is viable. |
| Each agent appears at most once per team. | Agent icon plus team colour gives a free, stable player identity. |

## 2. Architecture

```
 Observer PC screen / capture card / NDI
            │  (30-60 fps)
            ▼
 ┌─────────────────┐   ┌──────────────────┐   ┌────────────────────┐
 │ 1. Capture      │──▶│ 2. Perception    │──▶│ 3. State tracker   │
 │ crop minimap,   │   │ icon detector,   │   │ Kalman + identity  │
 │ HUD, killfeed   │   │ HUD/killfeed OCR │   │ → GameState @10 Hz │
 └─────────────────┘   └──────────────────┘   └─────────┬──────────┘
                                                        ▼
 ┌──────────────────┐   ┌──────────────────┐   ┌────────────────────┐
 │ 6. Observer UI   │◀──│ 5. Alert policy  │◀──│ 4. Predictor       │
 │ 2nd-monitor app/ │   │ hysteresis,      │   │ features → P(fight)│
 │ OBS source/deck  │   │ priority, chaos  │   │ per player & pair  │
 └──────────────────┘   └──────────────────┘   └────────────────────┘
            ▲                                            │
            └──────── 7. Recorder (frames, state, labels, observer actions) ──▶ training data
```

Everything is one process to start (Python, OpenCV, ONNX Runtime with GPU), with the UI served over a local WebSocket. Split into services only if needed.

### 2.1 Capture
- Windows Graphics Capture or DXGI Desktop Duplication on the observer PC, or a capture card / NDI feed on a second machine (preferred: zero load on the game PC).
- Fixed regions of interest per HUD layout: minimap, top bar (round timer, alive counts, spike), kill feed, player cards (HP bars).
- **Standardise observer settings** (minimap fixed orientation, fixed zoom, fixed size). This removes most perception complexity. Document them as a requirement.
- Per-map calibration: a homography from minimap pixels to world coordinates. Community map metadata (for example valorant-api.com's per-map `xMultiplier / yMultiplier / scalarToAdd` and minimap images) gives the transform; verify it once per map with a few clicks.

### 2.2 Perception
- **Agent icon detector:** small one-stage detector (YOLO-nano class) on the ~300×300 minimap crop, outputting box, agent class, team colour, and a "dead" flag. Run at 15 to 30 Hz; ~3 to 8 ms on a mid-range GPU.
- **Facing:** the view cone; a tiny regression head or classical angle estimate on the cone wedge.
- **Spike, smokes, walls, utility:** template matching first, then add classes to the detector.
- **HUD OCR:** round timer, spike-planted state, alive counts. **Kill feed:** killer agent, victim agent, weapon. These become ground-truth labels as much as live features.
- **Training data for the detector:** mostly synthetic. Composite official agent icons onto the map images with random rotation, overlap, cone, ping and ability clutter, then fine-tune on a few hundred hand-labelled real frames (CVAT). Retrain each patch that changes HUD art.

### 2.3 State tracker
- One track per player, keyed by (team, agent). Hungarian matching on detections, Kalman filter for position and velocity, so a player stays tracked through 1 to 2 s of icon occlusion.
- Output a `GameState` at 10 Hz:

```python
@dataclass
class PlayerState:
    id: str; team: Literal["atk","def"]; agent: str
    pos: Vec2; vel: Vec2; facing: float
    alive: bool; hp: int | None; zone: str  # callout, e.g. "A Main"

@dataclass
class GameState:
    t: float; round_time_left: float; phase: Literal["buy","live","post_plant"]
    spike: SpikeState; players: list[PlayerState]; utility: list[Utility]
```

### 2.4 Map knowledge (precomputed per map, offline)
- Walkable mask and wall mask traced from the minimap image.
- **Geodesic distance field** between all grid cells (players can't fight through walls; straight-line distance lies).
- **Line-of-sight table:** coarse visibility between cells via raycast on the wall mask; mark cells that are "one step from LOS" (peek positions).
- Callout polygons and choke points (from community callout maps), used for features and for UI wording.

## 3. Prediction

**Target:** for each opposing pair (i, j), the probability they are involved in an engagement within the next T seconds (T = 2, 3, 5). Per-player probability is derived from the pair probabilities. An engagement is a kill, or a damage event when HP is visible, between them.

This framing scales: 25 opposing pairs per tick, trivially cheap.

### Phase A: heuristic score (day one, no data)
`score(i,j) = w1·f(geodesic_dist) + w2·closing_speed + w3·LOS_now_or_next + w4·facing_alignment + w5·at_choke + context`

where context adds spike state (post-plant retakes force fights), time left, and man advantage. Hand-tune on a few VODs. This is surprisingly strong because most fights are "two people getting close along a path with sightlines".

### Phase B: gradient-boosted trees
Same pairwise features plus history (distance 0.5/1/2 s ago, time since last seen moving, teammates nearby, enemies nearby, utility in between). Train LightGBM per horizon. Calibrate with isotonic regression so "70%" means 70%.

### Phase C: sequence model
Treat each tick as a set of 10 player tokens plus spike/utility tokens, with a 2 to 3 s history window. A small transformer or graph network outputs per-pair hazards for multiple horizons (survival-style). Only worth it once you have thousands of rounds; keep the GBDT as the fallback and the baseline to beat.

## 4. Observer-facing behaviour (the part that decides if anyone uses it)

- **Primary output:** one recommendation, "Switch to Jett (key 3), 74% fight in ~3 s vs Sova at B Main", plus a ranked list of the next two.
- **Minimap overlay:** highlighted pair with an arc between them, intensity by probability.
- **Hysteresis:** only recommend a change when the top candidate beats the current POV's score by a margin for ≥300 ms. Flicker kills trust faster than misses do.
- **Chaos mode:** when two or more pairs exceed the threshold at once, rank by a priority policy (clutch situations, spike plant/defuse, likely multi-kill, star player, first blood), and show the runner-up as "also watch".
- **Delivery:** second-monitor window or OBS browser source, optional Stream Deck tile, optional soft audio tick. Auto-switching (sending the number key) is an opt-in setting, off by default, so the observer stays in control.
- **Feedback capture:** log whether the observer followed the suggestion. This is free training signal and the core product metric.

## 5. Latency budget

| Stage | Target |
|---|---|
| Capture + crop | 5 to 15 ms |
| Detection (GPU, small crop) | 3 to 8 ms |
| OCR (kill feed, timer; can run at 5 Hz) | 5 to 10 ms |
| Tracking + features + model | < 3 ms |
| UI push + render | ~16 ms |
| **End to end** | **< 60 ms typical, < 100 ms worst** |

Run inference at 10 to 15 Hz. Higher rates add nothing given a 2 to 5 s horizon.

## 6. Data and labelling

- **Source:** pro and amateur broadcast VODs (minimap and kill feed are on screen) and your own recorded custom games or scrims through the observer client. Check usage rights before using third-party VODs beyond internal research.
- **Auto-labelling:** run the perception pipeline on VODs; kill-feed events (with timestamps and killer/victim agents) give engagement labels with no human effort. Where HP bars are visible, HP drops label non-lethal fights too.
- **Human labelling** only for the detector bootstrap and an evaluation set.
- **Recorder** stores frames (sampled), `GameState` stream, predictions, and observer actions in Parquet, keyed by match/round, so any model can be replayed offline.

## 7. Evaluation

- **Perception:** detector mAP; identity switches per round (IDF1); position error in metres after homography.
- **Prediction (offline replay):** for every kill, was the killer or victim in the top-1 / top-2 recommendation 1, 2, 3 and 5 s before it? Report recall@k by lead time, plus Brier score and a calibration curve, plus false alerts per round.
- **Product metric:** *kill coverage*, the share of kills where the observed POV was on the killer or victim at the moment of the kill. Measure the human-only baseline from VODs, then with the tool in scrims. This is the number to put in front of observers.
- Split train/test by match and by map, never by frame, or the numbers will be fake.

## 8. Phased build plan

| Phase | Scope | Exit criterion |
|---|---|---|
| **0. Offline prototype** | Load a VOD, crop minimap, detect team-coloured dots with colour thresholds, OCR the kill feed, heuristic pair score, replay viewer showing predictions against real kills. One map (e.g. Ascent). | Heuristic shows meaningful recall@2 at 2 s over a naive "closest pair" baseline. |
| **1. Live MVP** | Live capture, fixed observer settings, tracker, heuristic predictor, second-monitor UI with hysteresis. All maps calibrated. | Used in a custom game end to end without crashes; latency < 100 ms. |
| **2. Learned perception + GBDT** | Synthetic-data detector with agent identity and facing, map LOS/geodesic precompute, LightGBM per horizon, recorder. | Beats heuristic on held-out matches; calibrated. |
| **3. Observer trial** | Chaos-mode priority policy, Stream Deck/OBS integration, follow/ignore logging. Trial with a real observer in scrims. | Kill coverage improves over that observer's baseline. |
| **4. Sequence model + utility** | Transformer/GNN over player sets, utility awareness, patch-robust retraining pipeline. | Beats GBDT on recall@2 at 3 s without hurting calibration. |

## 9. Suggested repo layout

```
observer-tool/
  capture/      # screen, NDI, video-file sources behind one interface
  perception/   # detector, OCR, HUD parsers, calibration
  tracking/     # Kalman + identity assignment → GameState
  maps/         # per-map masks, distance fields, LOS tables, callouts (generated)
  predict/      # heuristic, gbdt, seq model behind one Predictor interface
  policy/       # hysteresis, priority, chaos mode
  ui/           # WebSocket server + web overlay (works as OBS browser source)
  record/       # Parquet recorder, replay tool
  train/        # synthetic data gen, labelling scripts, training, eval harness
```

Keep `capture` pluggable with a video-file source from day one: every later phase is developed against recorded footage, not a live game.

## 10. Risks

- **HUD changes each patch** break perception. Mitigate with synthetic data regenerated from current icon assets and a regression test set of frames.
- **Minimap clutter** (pings, ability icons, stacked players) causes identity swaps. Mitigate with a Kalman tracker and agent-class constraints.
- **Fights the minimap can't foresee** (Operator picks across long sightlines, ultimates, lurkers) limit ceiling accuracy. That is fine: the tool's job is to beat a busy human, not to be perfect.
- **Rules:** confirm with the tournament organiser that a passive screen-reading aid for the observer is allowed on the observer PC; a capture-card setup on a separate machine avoids most concerns.
