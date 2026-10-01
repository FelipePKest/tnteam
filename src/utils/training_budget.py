"""Monotonic training deadline; callers finish an in-flight update before saving."""
import time


class TrainingBudget:
    def __init__(self, seconds=0, clock=time.monotonic):
        if seconds < 0:
            raise ValueError('training_wall_time_seconds must be nonnegative')
        self.clock = clock
        self.deadline = clock() + seconds if seconds else None

    def expired(self):
        return self.deadline is not None and self.clock() >= self.deadline
