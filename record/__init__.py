"""
record: saving and loading what the pipeline produced, so later steps can
replay it without decoding the video again.

Phase 0 contents:
  track_log.py  reads the tracks.csv and deaths.csv written by
                scripts/track_players.py back into Python objects
"""
