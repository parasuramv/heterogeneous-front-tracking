"""Ready-made experiments, shared by the CLI, the tests and the notebooks.

Each ``Example`` builds a fresh tracker from three resolution parameters:
δ (flux-level spacing), the number of g-mesh nodes, and the number of data
cells. It also records an observation window, a final time and the power p of
h(u) = |u|^p. The xu² examples are exact solutions only for p = 2.
"""

from dataclasses import dataclass, field
import math

import numpy as np

from .flux import MultiplicativeFlux
from .heterogeneity import PiecewiseAffine
from .periodic import PeriodicHeteroFrontTracker
from .solver import HeteroFrontTracker


@dataclass
class Example:
    name: str
    description: str
    g: object                   # exact heterogeneity g(x) (vectorised)
    mesh: tuple                 # (a, b) on the line, or (origin, period) on a circle
    window: tuple
    final_time: float
    data: object                # u0(x), or ("steps", positions, theta_values)
    periodic: bool = False
    extend: str = "constant"
    defaults: dict = field(default_factory=lambda: {"delta": 0.01, "nodes": 81, "cells": 120})
    characteristic_starts: tuple = ()
    power: float = 2.0

    def heterogeneity(self, nodes=None):
        nodes = self.defaults["nodes"] if nodes is None else int(nodes)
        if self.periodic:
            origin, period = self.mesh
            grid = origin + np.arange(nodes) * period / nodes
            return PiecewiseAffine.from_function(self.g, grid, period=period)
        return PiecewiseAffine.from_function(self.g, np.linspace(*self.mesh, nodes), extend=self.extend)

    def build(self, delta=None, nodes=None, cells=None, power=None, **kwargs):
        delta = self.defaults["delta"] if delta is None else float(delta)
        cells = self.defaults["cells"] if cells is None else int(cells)
        power = self.power if power is None else power
        flux = MultiplicativeFlux(self.heterogeneity(nodes), delta, power)
        if isinstance(self.data, tuple) and self.data[0] == "steps":
            _, positions, theta = self.data
            cls = PeriodicHeteroFrontTracker if self.periodic else HeteroFrontTracker
            return cls.from_theta(flux, positions, theta, **kwargs)
        if self.periodic:
            return PeriodicHeteroFrontTracker.from_function(flux, self.data, cells=cells, **kwargs)
        return HeteroFrontTracker.from_function(flux, self.data, self.window, cells=cells, **kwargs)

    def characteristics(self, flux, t_final=None):
        """Exact characteristics of the *initial data* from the listed starting points."""
        t_final = self.final_time if t_final is None else t_final
        out = []
        for x0 in self.characteristic_starts:
            u = float(np.asarray(self.data(np.array([x0]))).ravel()[0]) if callable(self.data) else 0.0
            out.append(flux.characteristic(x0, float(flux.theta_of(np.array([x0]), np.array([u]))[0]), t_final))
        return out


def _pulse(x):
    return np.where(np.abs(x) < 1, 1.5 * np.cos(np.pi * x / 2) - 0.5, 0.0)


EXAMPLES = {
    "sine": Example(
        "sine", "Well-signed g = 2 + sin 3x on the line; a signed pulse spreads and steepens.",
        lambda x: 2 + np.sin(3 * x), (-6, 6), (-3, 4), 3.0, _pulse),
    "burgers": Example(
        "burgers", "Constant g = 1/2: Burgers' equation (the homogeneous special case).",
        lambda x: 0.5 + 0 * x, (-1, 1), (-1, 4), 3.0,
        lambda x: np.where((x >= 0) & (x < 1), 1.0, 0.0), defaults={"delta": 0.005, "nodes": 2, "cells": 40}),
    "xu2-shock": Example(
        "xu2-shock", "g = x, u0 = -1/sqrt(-x) for x < -1, 0 otherwise. The shock follows "
        "y(t) = -(t-2)²/4 into the zero; the selected continuation after t = 2 is stationary.",
        lambda x: x, (-1, 1), (-2, 1), 3.0, ("steps", [-1.0], [1.0, 0.0]),
        extend="affine", defaults={"delta": 0.01, "nodes": 3, "cells": 1}),
    "xu2-blowup": Example(
        "xu2-blowup", "g = x, u0 = -1 on |x| <= 1: characteristics q0(1-t)² reach the zero at "
        "t = 1, where the L∞ norm blows up while θ stays bounded.",
        lambda x: x, (-1, 1), (-1.6, 1.6), 1.5,
        lambda x: np.where(np.abs(x) <= 1, -1.0, 0.0), extend="affine",
        defaults={"delta": 0.01, "nodes": 3, "cells": 120},
        characteristic_starts=tuple(np.linspace(-0.9, 0.9, 7))),
    "two-zeros": Example(
        "two-zeros", "g = x(1 - x²): three transverse zeros split the line into four "
        "independent sign intervals.",
        lambda x: x * (1 - x * x), (-2, 2), (-2.5, 2.5), 2.0,
        lambda x: np.where(np.abs(x) < 1.8, 0.6 * np.sin(2 * x), 0.0)),
    "power-sine": Example(
        "power-sine", "Well-signed g = 2 + sin 3x with h(u) = |u|³: the sine example with a "
        "steeper flux; fronts are straight in σ = |g|^(2/3).",
        lambda x: 2 + np.sin(3 * x), (-6, 6), (-3, 4), 3.0, _pulse, power=3.0),
    "power-zeros": Example(
        "power-zeros", "g = x(1 - x²) with h(u) = |u|^(3/2): absorbing zeros where u blows up "
        "like |x - z|^(-2/3).",
        lambda x: x * (1 - x * x), (-2, 2), (-2.5, 2.5), 2.0,
        lambda x: np.where(np.abs(x) < 1.8, 0.6 * np.sin(2 * x), 0.0), power=1.5),
    "circle-sine": Example(
        "circle-sine", "Well-signed g = 2 + sin x on the circle of length 2π; fronts circulate "
        "and total mass is conserved exactly.",
        lambda x: 2 + np.sin(x), (0.0, 2 * math.pi), (0.0, 2 * math.pi), 8.0,
        lambda x: 1 + 0.8 * np.sin(x) + 0.3 * np.cos(3 * x), periodic=True,
        defaults={"delta": 0.01, "nodes": 48, "cells": 120}),
    "circle-zeros": Example(
        "circle-zeros", "g = sin 2πx on the unit circle: two absorbing interfaces, two sign arcs.",
        lambda x: np.sin(2 * np.pi * x), (0.0, 1.0), (0.0, 1.0), 3.0,
        lambda x: 0.5 * np.cos(2 * np.pi * x) + 0.2, periodic=True,
        defaults={"delta": 0.005, "nodes": 40, "cells": 100}),
    "circle-homogenisation": Example(
        "circle-homogenisation", "Rapidly oscillating g = 2 + sin(16πx) on the unit circle "
        "(period ε = 1/8): the coarse dynamics follow an effective homogeneous flux.",
        lambda x: 2 + np.sin(16 * np.pi * x), (0.0, 1.0), (0.0, 1.0), 2.0,
        lambda x: 0.6 + 0.5 * np.sin(2 * np.pi * x), periodic=True,
        defaults={"delta": 0.005, "nodes": 256, "cells": 100}),
}
