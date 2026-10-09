"""
perception: turn raw video frames into facts about the game.

Phase 0 contents:
  regions.py         where the minimap and kill feed sit on the 1080p screen
  map_background.py  an "empty" picture of the minimap, built from the VOD
  minimap_dots.py    finds each team's player icons on the minimap
  kill_feed.py       reads the kill feed and reports each new kill
  teams.py           team labels (red / teal) and their names in the VOD
"""
