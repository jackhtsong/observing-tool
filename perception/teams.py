"""
teams.py: the team labels shared by every part of the project.

The broadcast shows each team in one colour everywhere: minimap rings, kill
feed rows and player cards. So the code calls teams by their colour, and
this file maps colours to the team names for the current VOD.
"""

RED = "red"
TEAL = "teal"
UNKNOWN = "unknown"

# Which team wears which colour in data/vods/ascent_test.mp4.
TEAM_NAMES = {RED: "PRX", TEAL: "LEV"}
