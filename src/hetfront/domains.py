"""Compact geometry independent of the numerical scheme (mirrors fronttrack.periodic.Circle)."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Circle:
    """Flat circle R/(length Z), represented on the chart [origin, origin + length)."""

    length: float = 1.0
    origin: float = 0.0

    def __post_init__(self):
        if not math.isfinite(self.length) or self.length <= 0:
            raise ValueError("length must be positive and finite")
        if (not math.isfinite(self.origin) or not math.isfinite(self.origin + self.length)
                or self.origin + self.length <= self.origin):
            raise ValueError("origin and endpoint must be finite and distinguishable")

    @property
    def dimension(self):
        return 1

    @property
    def volume(self):
        return self.length

    def wrap(self, points):
        """Map finite coordinates into [origin, origin + length), same shape."""
        points = np.asarray(points, dtype=float)
        if not np.all(np.isfinite(points)):
            raise ValueError("coordinates must be finite")
        result = self.origin + np.remainder(points - self.origin, self.length)
        # remainder can return exactly `length` for tiny negative inputs.
        result = np.where(result >= self.origin + self.length, self.origin, result)
        return float(result) if points.ndim == 0 else result
