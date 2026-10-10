"""
evaluate_predictor.py: does the heuristic see fights coming better than "closest pair"?

This is the phase 0 exit test from the README. For every kill we go back in
time by a "lead" (1, 2, 3 and 5 seconds), ask each predictor for its top
pairs at that moment, and check whether it picked the players who then
fought. If an observer had switched to the top suggestion that early, would
they have been on the fight?

Two ways to count a hit:
  victim   one of the top-k pairs includes the player who died. This is
           certain, since the tracker tied every death to a track.
  pair     one of the top-k pairs is exactly the victim and the killer. The
           kill feed tells us the killer's team but not which track, so we
           take the enemy closest (walking) to the victim just before the
           death. This is an educated guess.

We also show what picking pairs at random would score, so the numbers have
something to compare against.

A kill is "not evaluable" at a lead if the victim's track didn't exist yet
at that moment (for example it started less than 2 s before the death, or
the moment falls in a replay). Those count as misses for every method, so
the comparison stays fair.

All tracked videos in data/out/tracks/ are pooled together, since more
kills make the comparison more trustworthy.

Run it from the project root, after scripts/process_vods.py (or
scripts/track_players.py for a single video):

    .venv/bin/python -m scripts.evaluate_predictor

Outputs go to data/out/eval/:
    per_kill.csv   one row per video, kill and lead: what each method picked and whether it hit
    recall.png     chart of hit rate (top 2) against lead time for each method
"""

from __future__ import annotations

import csv
from math import comb
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # draw charts to files, no window needed
import matplotlib.pyplot as plt

from maps.map_geometry import MapGeometry
from perception.map_background import floor_mask
from predict.heuristic import ClosestPairBaseline, HeuristicPredictor, PairScore
from record.track_log import TrackLog, load_deaths, load_tracks
from tracking.player_tracker import Death, Track

LEADS = [1.0, 2.0, 3.0, 5.0]     # seconds before the kill
TOP_K = [1, 2]
TRACKS_DIR = Path("data/out/tracks")   # one subfolder per tracked video
OUT_DIR = Path("data/out/eval")


def guess_killer(log: TrackLog, death: Death, geometry: MapGeometry) -> str | None:
    """Best guess at the killer's track id: the enemy walking-closest to the victim.

    We use the last frame (up to 2 s before the kill) where the victim is
    still tracked, because by the kill time itself the victim's dot is gone.
    """
    for back in [x / 10 for x in range(0, 21)]:
        tracks = log.tracks_at(death.time - back)
        if not tracks:
            continue
        victim = next((t for t in tracks if t.id == death.track_id), None)
        if victim is None:
            continue
        enemies = [t for t in tracks if t.team != victim.team]
        if not enemies:
            return None
        nearest = min(enemies, key=lambda e: geometry.walking_distance((victim.x, victim.y), (e.x, e.y)))
        return nearest.id
    return None


def random_hit_chance(tracks: list[Track], victim_id: str, k: int) -> float:
    """Chance that k randomly picked pairs include the victim.

    With N pairs in total, m of which include the victim, the chance that
    none of k random picks does is C(N-m, k) / C(N, k).
    """
    reds = sum(t.team == "red" for t in tracks)
    teals = len(tracks) - reds
    total = reds * teals
    victim_team = next(t.team for t in tracks if t.id == victim_id)
    with_victim = teals if victim_team == "red" else reds
    if total == 0:
        return 0.0
    k = min(k, total)
    return 1.0 - comb(total - with_victim, k) / comb(total, k)


def is_pair(p: PairScore, a: str, b: str) -> bool:
    """True if pair p is exactly the two tracks a and b."""
    return {p.red.id, p.teal.id} == {a, b}


def evaluate_video(folder: Path, hits: dict, evaluable: dict, rows: list) -> int:
    """Score both predictors on every kill of one tracked video.

    Adds to the running totals in `hits`, `evaluable` and `rows`, so several
    videos can be pooled. Returns the number of kills in this video.
    """
    log = load_tracks(str(folder / "tracks.csv"))
    deaths = load_deaths(str(folder / "deaths.csv"))
    geometry = MapGeometry(floor_mask(cv2.imread(str(folder / "background.png"))))
    methods = {"heuristic": HeuristicPredictor(geometry), "closest pair": ClosestPairBaseline()}

    killers = {d.time: guess_killer(log, d, geometry) for d in deaths}

    for death in deaths:
        for lead in LEADS:
            tracks = log.tracks_at(death.time - lead)
            if not tracks or not any(t.id == death.track_id for t in tracks):
                continue  # victim not tracked yet: a miss for everyone
            evaluable[lead] += 1

            for k in TOP_K:
                key = ("random", "victim", lead, k)
                hits[key] = hits.get(key, 0) + random_hit_chance(tracks, death.track_id, k)

            for name, method in methods.items():
                ranked = method.rank_pairs(tracks)
                for k in TOP_K:
                    top = ranked[:k]
                    victim_hit = any(p.involves(death.track_id) for p in top)
                    killer = killers[death.time]
                    pair_hit = killer is not None and any(is_pair(p, death.track_id, killer) for p in top)
                    hits[(name, "victim", lead, k)] = hits.get((name, "victim", lead, k), 0) + victim_hit
                    hits[(name, "pair", lead, k)] = hits.get((name, "pair", lead, k), 0) + pair_hit
                best = ranked[0] if ranked else None
                rows.append([folder.name, f"{death.time:.1f}", death.track_id, killers[death.time] or "",
                             lead, name, f"{best.red.id}+{best.teal.id}" if best else "",
                             int(any(p.involves(death.track_id) for p in ranked[:2]))])
    return len(deaths)


def main() -> None:
    """Score both predictors on every tracked video, print a table, save a chart."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Every folder written by track_players.py / process_vods.py.
    folders = sorted(f for f in TRACKS_DIR.iterdir() if (f / "deaths.csv").exists())
    if not folders:
        print(f"No tracked videos in {TRACKS_DIR}/. Run scripts.process_vods first.")
        return

    # hits[(method, measure, lead, k)] = number of kills hit, pooled over all videos
    hits: dict[tuple, float] = {}
    evaluable = {lead: 0 for lead in LEADS}
    rows: list = []
    n = 0
    for folder in folders:
        kills = evaluate_video(folder, hits, evaluable, rows)
        print(f"{folder.name}: {kills} kills")
        n += kills
    print()

    with open(OUT_DIR / "per_kill.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "kill_time", "victim", "killer_guess", "lead", "method", "top_pair", "victim_in_top2"])
        writer.writerows(rows)

    # ----- print the results table -----
    print(f"{n} kills. Hit rate = kills hit / all {n} kills.\n")
    print(f"{'lead':>5}  {'evaluable':>9}  {'measure':>7}  {'k':>2}  {'heuristic':>9}  {'closest':>8}  {'random':>7}")
    for lead in LEADS:
        for measure in ("victim", "pair"):
            for k in TOP_K:
                h = hits.get(("heuristic", measure, lead, k), 0) / n
                c = hits.get(("closest pair", measure, lead, k), 0) / n
                r = hits.get(("random", "victim", lead, k), 0) / n if measure == "victim" else float("nan")
                r_text = f"{r:7.0%}" if measure == "victim" else "      -"
                print(f"{lead:4.0f}s  {evaluable[lead]:>9}  {measure:>7}  {k:>2}  {h:9.0%}  {c:8.0%}  {r_text}")

    # ----- chart: victim in top 2, by lead time -----
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for name, label in (("heuristic", "Heuristic"), ("closest pair", "Closest pair"), ("random", "Random")):
        ys = [hits.get((name, "victim", lead, 2), 0) / n for lead in LEADS]
        ax.plot(LEADS, ys, marker="o", label=label)
    ax.set_xlabel("Seconds before the kill")
    ax.set_ylabel("Victim in top 2 pairs")
    ax.set_ylim(0, 1)
    ax.set_xticks(LEADS)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title(f"Who fights next? ({n} kills, {len(folders)} video{'s' if len(folders) > 1 else ''})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "recall.png", dpi=150)
    print(f"\nWrote {OUT_DIR}/per_kill.csv and {OUT_DIR}/recall.png")


if __name__ == "__main__":
    main()
