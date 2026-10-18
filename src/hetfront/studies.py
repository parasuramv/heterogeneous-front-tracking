"""Empirical convergence in δ (flux levels) and Δ (mesh of g), measured exactly.

Errors are computed without quadrature against a fine reference run:

* ``theta_l1``  ∫|θ - θ_ref| over the window. Both are step functions, so this is exact.
* ``primitive`` max over a grid of |∫_a^x (u - u_ref)|, from exact cell
  integrals of the stationary profiles. It is robust where u blows up at zeros.

The test problems use initial flux levels that are multiples of 0.1, so the
data are represented exactly for every δ = 0.1·2^{-k}. δ then enters only
through δ-fans, and Δ only through g_Δ. Rates are fitted on a log-log scale;
they are observations, not theorems.
"""

from dataclasses import dataclass
import math
import operator
import time

import numpy as np

from .flux import MultiplicativeFlux
from .heterogeneity import PiecewiseAffine
from .periodic import PeriodicHeteroFrontTracker
from .solver import HeteroFrontTracker


@dataclass
class Problem:
    name: str
    g: object
    mesh: tuple            # (a, b) on the line, (origin, period) on a circle
    window: tuple
    positions: tuple
    theta: tuple           # initial flux levels, multiples of 0.1
    final_time: float
    periodic: bool = False
    power: float = 2.0     # theta data stay fixed; physical initial profiles vary with p
    max_events: int = 1_000_000

    def heterogeneity(self, nodes):
        if self.periodic:
            o, L = self.mesh
            return PiecewiseAffine.from_function(self.g, o + np.arange(nodes) * L / nodes, period=L)
        return PiecewiseAffine.from_function(self.g, np.linspace(*self.mesh, nodes))

    def tracker(self, delta, nodes, **kwargs):
        kwargs.setdefault("max_events", self.max_events)
        flux = MultiplicativeFlux(self.heterogeneity(nodes), delta, power=self.power)
        cls = PeriodicHeteroFrontTracker if self.periodic else HeteroFrontTracker
        return cls.from_theta(flux, self.positions, self.theta, record_history=False, **kwargs)


PROBLEMS = {
    "line-well-signed": Problem(
        "line-well-signed", lambda x: 2 + np.sin(3 * x), (-4, 4), (-3, 3),
        (-1.5, -0.5, 0.5, 1.5), (0.0, 1.2, 0.2, -0.6, 0.0), 1.0),
    "line-sign-changing": Problem(
        "line-sign-changing", lambda x: x * (1 - x * x), (-2, 2), (-2, 2),
        (-1.5, -0.6, 0.4, 1.3), (0.0, 0.4, -0.3, 0.5, 0.0), 1.0),
    "circle-well-signed": Problem(
        "circle-well-signed", lambda x: 2 + np.sin(2 * np.pi * x), (0.0, 1.0), (0.0, 1.0),
        (0.1, 0.4, 0.7), (1.0, 0.3, 0.6), 1.0, periodic=True),
    "circle-sign-changing": Problem(
        "circle-sign-changing", lambda x: np.sin(2 * np.pi * x) + 0.3, (0.0, 1.0), (0.0, 1.0),
        (0.1, 0.4, 0.7), (0.5, -0.2, 0.3), 1.0, periodic=True),
}


def _count(value, name, minimum):
    try:
        count = operator.index(value)
    except TypeError:
        raise ValueError(f"{name} must be an integer >= {minimum}") from None
    if isinstance(value, (bool, np.bool_)) or count < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return count


def errors(tracker, reference, window, samples=801):
    """Exact θ-L¹ and grid-sampled primitive maximum at the current time."""
    samples = _count(samples, "samples", 2)
    a, b = window
    xs = np.linspace(a, b, samples)
    prim = np.max(np.abs(tracker.primitive(xs) - reference.primitive(xs)))
    return tracker.theta_l1_distance(reference, a, b), float(prim)


def reference_run(problem, delta, nodes, final_time=None):
    """The fine run that other runs are compared with, plus its cost."""
    T = problem.final_time if final_time is None else final_time
    start = time.perf_counter()
    ref = problem.tracker(delta, nodes).run(T)
    return ref, {"power": problem.power, "final_time": T, "delta": delta,
                 "nodes": nodes, "max_events": problem.max_events, "events": ref.event_count,
                 "seconds": time.perf_counter() - start}


def refinement(problem, pairs, reference, *, samples=801):
    """Errors of runs with (δ, nodes) ``pairs`` against a finished ``reference`` tracker."""
    samples = _count(samples, "samples", 2)
    rows = []
    span = problem.mesh[1] if problem.periodic else problem.mesh[1] - problem.mesh[0]
    for delta, nodes in pairs:
        start = time.perf_counter()
        tr = problem.tracker(delta, nodes).run(reference.time)
        seconds = time.perf_counter() - start
        e_theta, e_prim = errors(tr, reference, problem.window, samples=samples)
        rows.append({"power": problem.power, "final_time": reference.time, "samples": samples,
                     "delta": delta, "nodes": nodes,
                     "h": span / (nodes if problem.periodic else nodes - 1), "theta_l1": e_theta,
                     "primitive": e_prim, "fronts": tr.front_count(), "events": tr.event_count,
                     "seconds": seconds})
    return rows


def fitted_order(x, y):
    """Least-squares slope of log y against log x (positive entries only)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = (x > 0) & (y > 0)
    if keep.sum() < 2:
        return math.nan
    return float(np.polyfit(np.log(x[keep]), np.log(y[keep]), 1)[0])


def standard_study(problem, *, levels=5, coarse_delta=0.1, coarse_nodes=16, depth=2, samples=801):
    """Three refinements: δ alone (fine mesh), Δ alone (fine δ), and both together.

    The reference is ``depth`` halvings beyond the finest run in each parameter.
    """
    levels = _count(levels, "levels", 2)
    depth = _count(depth, "depth", 1)
    coarse_nodes = _count(coarse_nodes, "coarse_nodes", 2)
    samples = _count(samples, "samples", 2)
    if not math.isfinite(coarse_delta) or coarse_delta <= 0:
        raise ValueError("coarse_delta must be finite and positive")
    deltas = [coarse_delta / 2 ** k for k in range(levels)]
    nodes = [coarse_nodes * 2 ** k for k in range(levels)]
    fine_delta, fine_nodes = deltas[-1] / 2 ** depth, nodes[-1] * 2 ** depth
    ref, info = reference_run(problem, fine_delta, fine_nodes)
    return {"reference": info,
            "delta": refinement(problem, [(d, fine_nodes) for d in deltas], ref, samples=samples),
            "mesh": refinement(problem, [(fine_delta, m) for m in nodes], ref, samples=samples),
            "joint": refinement(problem, list(zip(deltas, nodes)), ref, samples=samples)}
