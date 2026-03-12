"""
Schedulers for learning rate, epsilon-greedy exploration, entropy coefficient, etc.

All schedulers expose a single .value(step) method that returns the
current scalar, so they can be dropped anywhere without coupling to
an optimiser or algorithm.
"""

import math


class LinearSchedule:
    """
    Linear annealing from `start` to `end` over `duration` steps,
    then holds `end` forever.

    Typical use: epsilon decay in DQN.
    """

    def __init__(self, start: float, end: float, duration: int):
        self.start = start
        self.end = end
        self.duration = duration

    def value(self, step: int) -> float:
        fraction = min(step / self.duration, 1.0)
        return self.start + fraction * (self.end - self.start)


class ExponentialSchedule:
    """
    Exponential decay: v(t) = max(end, start * decay^t)

    Typical use: entropy temperature annealing in SAC.
    """

    def __init__(self, start: float, end: float, decay: float):
        self.start = start
        self.end = end
        self.decay = decay

    def value(self, step: int) -> float:
        return max(self.end, self.start * (self.decay**step))


class CosineSchedule:
    """
    Cosine annealing between `start` and `end` over `duration` steps.

    Typical use: learning rate warm restarts.
    """

    def __init__(self, start: float, end: float, duration: int):
        self.start = start
        self.end = end
        self.duration = duration

    def value(self, step: int) -> float:
        t = min(step, self.duration)
        cos_val = 0.5 * (1 + math.cos(math.pi * t / self.duration))
        return self.end + (self.start - self.end) * cos_val


class ConstantSchedule:
    """No-op schedule – always returns the same value."""

    def __init__(self, value: float):
        self._value = value

    def value(self, step: int = 0) -> float:
        return self._value
