"""
label_dots.py: a click tool for marking where every player is on the minimap.

It shows one minimap frame at a time, enlarged. You click on every living
player's icon; the clicks are saved to data/labels/dots.json and used by
scripts/score_dots.py to grade the dot detector.

Controls
--------
    left click             mark a red (PRX) player
    right click or         mark a teal (LEV) player
      shift + left click
    click on a mark        remove it
    space / enter / n      this frame is done, go to the next one
    b                      go back one frame
    u                      undo the last mark on this frame
    c                      clear all marks on this frame
    d                      show / hide the detector's guesses (off by default,
                           so they don't sway you)
    q / esc                save and quit (you can carry on later)

Tips: click the centre of the round portrait, not the pointer. Mark players
even when their icon is half hidden under another. Don't mark death X's,
smokes or ability icons. If a frame is a replay or makes no sense, just
press space without marking anything.

Run it from the project root:

    .venv/bin/python -m scripts.label_dots              # 50 frames from all videos in data/vods/
    .venv/bin/python -m scripts.label_dots --count 80   # add more frames to label
    .venv/bin/python -m scripts.label_dots --scale 1.5  # smaller window, for small screens

The first run picks the frames (random live-round moments, the same every
time thanks to a fixed random seed). Later runs carry on where you stopped.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from capture.video_source import VideoSource
from perception.dot_labels import LabelledFrame, LabelledPlayer, load_labels, save_labels
from perception.map_background import floor_mask, load_or_build_background
from perception.minimap_dots import DotDetector
from perception.regions import MINIMAP
from perception.teams import RED, TEAL

WINDOW = "Label players"
SCALE = 2.0          # how much to enlarge the minimap on screen (--scale changes it)
BAR_HEIGHT = 46      # height of the instructions bar above the minimap
REMOVE_RADIUS = 6    # clicking within this many minimap px of a mark removes it
COLOURS = {RED: (40, 40, 255), TEAL: (255, 230, 0)}   # BGR
VOD_DIR = Path("data/vods")


# ----- picking which frames to label -----

def live_times(video: VideoSource, detector: DotDetector, every: float = 2.0) -> list[float]:
    """Times (every `every` seconds) where the minimap shows a live round."""
    return [t for t, frame in video.frames(0, None, every)
            if detector.is_standard_layout(MINIMAP.crop(frame))]


def pick_frames(count: int, already: list[LabelledFrame], seed: int = 7) -> list[LabelledFrame]:
    """Choose `count` frames to label, spread across all videos in data/vods/.

    Each video gets a share proportional to how much live-round time it
    has. Frames already in `already` are kept, and only the missing number
    of new ones is added.
    """
    videos = sorted(p for p in VOD_DIR.iterdir() if p.suffix.lower() in {".mp4", ".mkv", ".mov"})
    candidates = []   # (video path, time)
    for path in videos:
        print(f"Finding live-round moments in {path.name}...")
        with VideoSource(str(path)) as video:
            background = load_or_build_background(video, MINIMAP)
            detector = DotDetector(background, floor_mask(background))
            candidates += [(str(path), t) for t in live_times(video, detector)]

    taken = {(f.video, round(f.time, 2)) for f in already}
    fresh = [c for c in candidates if (c[0], round(c[1], 2)) not in taken]
    random.Random(seed).shuffle(fresh)   # fixed seed: the same picks every run
    needed = max(count - len(already), 0)
    new = [LabelledFrame(video=v, time=t) for v, t in fresh[:needed]]
    return already + new


# ----- the labelling session (no window code here, so it can be tested) -----

class LabelSession:
    """Holds the frames being labelled and reacts to clicks and keys."""

    def __init__(self, frames: list[LabelledFrame]):
        self.frames = frames
        # Start at the first frame not marked done yet.
        self.index = next((i for i, f in enumerate(frames) if not f.done), len(frames) - 1)
        self.show_detector = False
        self._videos: dict[str, VideoSource] = {}
        self._detectors: dict[str, DotDetector] = {}
        self._images: dict[int, np.ndarray] = {}

    @property
    def frame(self) -> LabelledFrame:
        """The frame currently on screen."""
        return self.frames[self.index]

    def click(self, x: float, y: float, team: str) -> None:
        """A click at minimap pixel (x, y): remove a nearby mark, or add a new one."""
        for player in self.frame.players:
            if np.hypot(player.x - x, player.y - y) <= REMOVE_RADIUS:
                self.frame.players.remove(player)
                return
        self.frame.players.append(LabelledPlayer(x=round(x, 1), y=round(y, 1), team=team))

    def undo(self) -> None:
        """Remove the most recent mark on this frame."""
        if self.frame.players:
            self.frame.players.pop()

    def clear(self) -> None:
        """Remove every mark on this frame."""
        self.frame.players.clear()

    def next(self) -> bool:
        """Mark this frame done and move on. Returns False after the last frame."""
        self.frame.done = True
        if self.index + 1 >= len(self.frames):
            return False
        self.index += 1
        return True

    def back(self) -> None:
        """Go back one frame (its marks stay editable)."""
        self.index = max(self.index - 1, 0)

    def done_count(self) -> int:
        return sum(f.done for f in self.frames)

    # ----- images -----

    def minimap(self) -> np.ndarray:
        """The current frame's minimap crop (videos stay open, crops are cached)."""
        if self.index not in self._images:
            video = self._video(self.frame.video)
            self._images[self.index] = MINIMAP.crop(video.frame_at(self.frame.time)).copy()
        return self._images[self.index]

    def detector_guesses(self):
        """What the dot detector finds on the current frame (for the 'd' overlay)."""
        if self.frame.video not in self._detectors:
            video = self._video(self.frame.video)
            background = load_or_build_background(video, MINIMAP)
            self._detectors[self.frame.video] = DotDetector(background, floor_mask(background))
        return self._detectors[self.frame.video].detect(self.minimap())

    def _video(self, path: str) -> VideoSource:
        if path not in self._videos:
            self._videos[path] = VideoSource(path)
        return self._videos[path]

    def close(self) -> None:
        for video in self._videos.values():
            video.close()

    def render(self) -> np.ndarray:
        """Draw the screen: instructions bar on top, enlarged minimap with marks below."""
        h, w = self.minimap().shape[:2]
        picture = cv2.resize(self.minimap(), (int(w * SCALE), int(h * SCALE)), interpolation=cv2.INTER_CUBIC)

        if self.show_detector:
            for d in self.detector_guesses():
                if d.team in COLOURS:
                    centre = (int(d.x * SCALE), int(d.y * SCALE))
                    cv2.circle(picture, centre, int((d.radius + 3) * SCALE), COLOURS[d.team], 1, cv2.LINE_AA)

        for p in self.frame.players:
            centre = (int(p.x * SCALE), int(p.y * SCALE))
            cv2.circle(picture, centre, 4, COLOURS[p.team], -1, cv2.LINE_AA)       # filled dot
            cv2.circle(picture, centre, 5, (0, 0, 0), 1, cv2.LINE_AA)               # black edge

        bar = np.full((BAR_HEIGHT, picture.shape[1], 3), 30, np.uint8)
        name = Path(self.frame.video).stem
        reds = sum(p.team == RED for p in self.frame.players)
        teals = len(self.frame.players) - reds
        status = (f"Frame {self.index + 1}/{len(self.frames)}  ({self.done_count()} done)   "
                  f"{name} {self.frame.time:.1f}s   red {reds}  teal {teals}")
        help_text = "L-click red | R-click/shift teal | click mark: remove | space next  b back  u undo  d detector  q quit"
        cv2.putText(bar, status, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(bar, help_text, (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (190, 190, 190), 1, cv2.LINE_AA)
        return np.vstack([bar, picture])


# ----- the window -----

def main() -> None:
    """Pick (or reload) the frames, then run the click-and-key loop until quit."""
    p = argparse.ArgumentParser(description="Click on every player to make detector test labels.")
    p.add_argument("--count", type=int, default=50, help="how many frames to label in total")
    p.add_argument("--scale", type=float, default=2.0,
                   help="how much to enlarge the minimap; use 1.5 if the window is taller than your screen")
    args = p.parse_args()
    global SCALE
    SCALE = args.scale

    frames = load_labels()
    if len(frames) < args.count:
        frames = pick_frames(args.count, frames)
        save_labels(frames)
    session = LabelSession(frames)

    def on_mouse(event, x, y, flags, _param):
        """OpenCV calls this for every mouse event in the window."""
        mx, my = x / SCALE, (y - BAR_HEIGHT) / SCALE   # window pixels -> minimap pixels
        if my < 0:
            return  # clicked on the instructions bar
        if event == cv2.EVENT_LBUTTONDOWN:
            session.click(mx, my, TEAL if flags & cv2.EVENT_FLAG_SHIFTKEY else RED)
        elif event == cv2.EVENT_RBUTTONDOWN:
            session.click(mx, my, TEAL)

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)
    print(f"Labelling {len(frames)} frames; {session.done_count()} already done. Press q to save and quit.")

    try:
        while True:
            cv2.imshow(WINDOW, session.render())
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):                       # q or esc
                break
            elif key in (ord(" "), 13, ord("n")):           # space, enter, n
                more = session.next()
                save_labels(frames)                         # save after every frame
                if not more:
                    print("All frames done.")
                    break
            elif key == ord("b"):
                session.back()
            elif key == ord("u"):
                session.undo()
            elif key == ord("c"):
                session.clear()
            elif key == ord("d"):
                session.show_detector = not session.show_detector
            # Closing the window with its red button also quits.
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        save_labels(frames)
        session.close()
        cv2.destroyAllWindows()
        print(f"Saved. {session.done_count()}/{len(frames)} frames done.")


if __name__ == "__main__":
    main()
