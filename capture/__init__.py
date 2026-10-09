"""
capture: where frames come from.

Phase 0 reads frames from a recorded VOD file. Later phases will add a live
screen-capture source with the same interface, so nothing downstream needs to
change when we switch from recordings to a live game.
"""
