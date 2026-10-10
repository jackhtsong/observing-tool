"""
train_fight_model.py: learn "chance of a kill in the next 3 s" for each pair, and check it honestly.

Steps:
  1. Build examples from every tracked video (data/out/tracks/<video>/):
     every `every` seconds, for every red/teal pair, the pair's features
     (predict/probability.py) and a label: 1 if either player died within
     the next `horizon` seconds (from deaths.csv), else 0.
  2. Check: leave one video out at a time, train on the rest, predict the
     one left out. This is how the model will be used (on matches it never
     saw), so these are the numbers to trust. Training and testing on the
     same video would look better than it really is.
  3. Train on all videos and save predict/models/fight_within_3s.json, used
     by the live tool.

What it prints:
  - calibration: for predictions in each band (0-10%, 10-20%, ...), how
    often a kill really followed. A good model's two columns match.
  - the "over 50%" rule: how often some pair is over 50%, how often a kill
    really follows when it is (precision), and how many kills had their
    victim's pair over 50% 2 s before (recall).

Run it from the project root, after scripts/process_vods.py:

    .venv/bin/python -m scripts.train_fight_model
    .venv/bin/python -m scripts.train_fight_model --horizon 3 --threshold 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from maps.map_model import load_map_for
from predict.probability import DEFAULT_MODEL, FEATURES, LogisticModel, pair_features
from record.track_log import load_deaths, load_round_state, load_tracks

TRACKS_DIR = Path("data/out/tracks")


def build_examples(folder: Path, horizon: float, every: float) -> dict:
    """Features and labels for one tracked video (step 1).

    Returns a dict of arrays: features (rows x FEATURES), labels (0/1),
    time and frame (which moment each row came from, so moments can be
    grouped), plus `kills`: for each death, whether its victim's pair
    was found 2 s before (used for recall).
    """
    log = load_tracks(str(folder / "tracks.csv"))
    deaths = load_deaths(str(folder / "deaths.csv"))
    rounds = load_round_state(str(folder / "round_state.csv"))
    model = load_map_for(folder.name)

    features, labels, frame_ids, times, pair_ids = [], [], [], [], []
    next_sample = -1e9
    for frame_number, t in enumerate(log.times):
        if t < next_sample:
            continue
        next_sample = t + every
        pairs, table = pair_features(log.frames[t], model, rounds.spike_at(t))
        dying = {d.track_id for d in deaths if t < d.time <= t + horizon}
        for (red, teal), row in zip(pairs, table):
            features.append(row)
            labels.append(1.0 if red.id in dying or teal.id in dying else 0.0)
            frame_ids.append(frame_number)
            times.append(t)
            pair_ids.append((red.id, teal.id))
    return {"name": folder.name, "features": np.array(features), "labels": np.array(labels),
            "frame": np.array(frame_ids), "time": np.array(times), "pairs": pair_ids,
            "deaths": deaths}


def report(name: str, probs: np.ndarray, data: dict, threshold: float) -> dict:
    """Calibration and threshold numbers for predictions on one or more videos."""
    labels = data["labels"]
    out = {"name": name, "rows": len(labels), "base_rate": labels.mean(),
           "brier": float(np.mean((probs - labels) ** 2)),
           "brier_base": float(np.mean((labels.mean() - labels) ** 2))}

    # Calibration bands.
    bands = []
    for lo in np.arange(0, 1, 0.1):
        inside = (probs >= lo) & (probs < lo + 0.1)
        if inside.any():
            bands.append((lo, inside.sum(), probs[inside].mean(), labels[inside].mean()))
    out["bands"] = bands

    # The threshold rule, per moment: is the best pair over the threshold?
    frames = data["frame"]
    best_rows = {}
    for i, f in enumerate(frames):
        if f not in best_rows or probs[i] > probs[best_rows[f]]:
            best_rows[f] = i
    over = [i for i in best_rows.values() if probs[i] > threshold]
    out["moments"] = len(best_rows)
    out["moments_over"] = len(over)
    out["precision"] = float(np.mean([labels[i] for i in over])) if over else float("nan")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Train and check the kill-likelihood model.")
    p.add_argument("--horizon", type=float, default=3.0, help="seconds ahead a kill counts")
    p.add_argument("--threshold", type=float, default=0.5, help="the 'suggest only above this' level to check")
    p.add_argument("--every", type=float, default=0.5, help="seconds between sampled moments")
    args = p.parse_args()

    folders = sorted(f for f in TRACKS_DIR.iterdir() if (f / "deaths.csv").exists())
    print(f"Building examples from {len(folders)} videos...")
    videos = [build_examples(f, args.horizon, args.every) for f in folders]
    for v in videos:
        print(f"  {v['name']}: {len(v['labels'])} pair-moments, {int(v['labels'].sum())} positive "
              f"({v['labels'].mean():.1%})")

    # Step 2: leave one video out.
    print(f"\nChecking on videos the model didn't train on (horizon {args.horizon:.0f} s):")
    all_probs, all_labels, held = [], [], []
    for i, test in enumerate(videos):
        train = [v for j, v in enumerate(videos) if j != i]
        model = LogisticModel.fit(np.vstack([v["features"] for v in train]),
                                  np.concatenate([v["labels"] for v in train]))
        probs = model.predict(test["features"])
        r = report(test["name"], probs, test, args.threshold)
        held.append(r)
        all_probs.append(probs)
        all_labels.append(test["labels"])
        print(f"  {r['name']:22s} Brier {r['brier']:.4f} (always-guess-average {r['brier_base']:.4f}); "
              f"best pair over {args.threshold:.0%} in {r['moments_over']}/{r['moments']} moments, "
              f"kill followed in {r['precision']:.0%} of those")

    pooled = {"labels": np.concatenate(all_labels),
              "frame": np.concatenate([v["frame"] + 10_000_000 * i for i, v in enumerate(videos)])}
    r = report("all held-out", np.concatenate(all_probs), pooled, args.threshold)
    print(f"\nCalibration on held-out videos (predicted vs what really happened):")
    print(f"  {'band':>9}  {'pair-moments':>12}  {'predicted':>9}  {'actual':>7}")
    for lo, n, predicted, actual in r["bands"]:
        print(f"  {lo:4.0%}-{lo + 0.1:3.0%}  {n:12d}  {predicted:9.1%}  {actual:7.1%}")
    print(f"\n'{args.threshold:.0%} rule' on held-out videos: some pair is over {args.threshold:.0%} in "
          f"{r['moments_over']} of {r['moments']} moments ({r['moments_over'] / r['moments']:.1%}); "
          f"a kill followed within {args.horizon:.0f} s in {r['precision']:.0%} of those.")

    # Step 3: final model on everything.
    model = LogisticModel.fit(np.vstack([v["features"] for v in videos]),
                              np.concatenate([v["labels"] for v in videos]))
    info = {"horizon_s": args.horizon, "videos": [v["name"] for v in videos],
            "held_out_brier": r["brier"], "held_out_brier_base": r["brier_base"]}
    model.save(DEFAULT_MODEL, info)
    print(f"\nSaved {DEFAULT_MODEL}. Weights (on standardised features):")
    for name, w in zip(FEATURES, model.weights):
        print(f"  {name:14s} {w:+.2f}")


if __name__ == "__main__":
    main()
