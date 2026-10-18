"""Reproducible extended audit: 420 random cases on the line and circle.

240 cases use h(u) = u²; 180 more use h(u) = |u|^p with p = 3/2 or 3.

Run from any directory with this checkout's Python environment. This is slower
than a focused regression: it checks many coefficients and signed initial data
against exact conjugacy, mass budgets, ordering, and interface invariants.
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hetfront import (HeteroFrontTracker, MultiplicativeFlux,  # noqa: E402
                      PeriodicHeteroFrontTracker, PiecewiseAffine)
from hetfront.homogenisation import PeriodicStraightening  # noqa: E402

TIMES = (.001, .03, .1, .3, 1., 3.)


def invariants(tracker, mass, tv, count, tol=1e-10):
    """Check the current run against its initial budget and variation bounds."""
    tracker.check_levels()
    assert np.all(np.diff(tracker.positions()) >= -1e-11)
    assert tracker.total_variation() <= tv + 1e-12
    assert tracker.front_count() <= count
    assert tracker.interface_admissible()
    assert tracker.interface_entropy_admissible()
    assert tracker.counts["emission"] == 0
    # Line cases have zero exterior levels and a wide observation window.
    current = tracker.integral() if tracker.cyclic else tracker.integral(-20, 20)
    residual = abs(current + tracker.mass_absorbed() - mass)
    assert residual < tol, residual
    return residual


def main():
    worst, cases = 0., 0
    for seed in range(60):
        rng = np.random.default_rng(seed)
        for sign in (-1, 1):
            # Exact change of variables: the reference has no coefficient
            # corners, while the physical tracker crosses many of them.
            g = PiecewiseAffine(np.arange(8) / 8, sign * rng.uniform(.3, 3, 8), period=1)
            mapping = PeriodicStraightening(g)
            positions = np.sort(rng.uniform(.001, .999, 10))
            levels = rng.integers(-10, 11, 10)
            tracker = PeriodicHeteroFrontTracker(MultiplicativeFlux(g, .1), positions, levels,
                                                  record_history=False)
            reference_g = PiecewiseAffine.constant(mapping.coefficient, period=1)
            reference = PeriodicHeteroFrontTracker(
                MultiplicativeFlux(reference_g, .1), mapping.coordinate(positions), levels,
                record_history=False)
            mass, tv, count = tracker.integral(), tracker.total_variation(), tracker.front_count()
            for t in TIMES:
                tracker.run(t)
                reference.run(t)
                reference.check_levels()
                assert [(f.kL, f.kR) for f in tracker.fronts] == [(f.kL, f.kR) for f in reference.fronts], (seed, sign, t)
                np.testing.assert_allclose(mapping.coordinate(tracker.positions()), reference.positions(),
                                           atol=1e-10, rtol=0)
                worst = max(worst, invariants(tracker, mass, tv, count))
            cases += 1

        for periodic in (False, True):
            # Direct nodal input generates genuine piecewise-affine sign
            # changes without depending on the callable root-finding heuristic.
            nodes = np.arange(8) / 8 if periodic else np.linspace(-1, 1, 9)
            g = PiecewiseAffine(nodes, rng.uniform(-2, 2, len(nodes)), period=1 if periodic else None)
            cls = PeriodicHeteroFrontTracker if periodic else HeteroFrontTracker
            positions = np.sort(rng.uniform(.01, .99, 8) if periodic else rng.uniform(-.9, .9, 8))
            levels = rng.integers(-8, 9, 8 if periodic else 9)
            if not periodic:
                levels[0] = levels[-1] = 0  # no boundary inflow on the line
            tracker = cls(MultiplicativeFlux(g, .1), positions, levels, record_history=False)
            mass = tracker.integral() if periodic else tracker.integral(-20, 20)
            tv, count = tracker.total_variation(), tracker.front_count()
            for t in TIMES:
                tracker.run(t)
                worst = max(worst, invariants(tracker, mass, tv, count))
            cases += 1

        # h(u) = |u|^p with p ≠ 2, from a separate stream so the cases above
        # stay reproducible: sign-changing nodal g on both domains, and exact
        # conjugacy with the effective flux η<|g|^{-1/p}>^{-p}|v|^p.
        prng = np.random.default_rng(10_000 + seed)
        power = (1.5, 3.0)[seed % 2]
        # Fronts emitted from a zero leave it like τ^m (m = p/(p-1) = 3 for
        # p = 3/2), so at t = 0.001 they can sit ~1e-12 from the zero, where
        # |u| ~ |x-z|^{-1/p} ~ 1e8. One ulp of position then moves the mass by
        # up to ~1e-8: the budget is roundoff-limited there, hence tol = 1e-9.
        tol = 1e-9
        for periodic in (False, True):
            nodes = np.arange(8) / 8 if periodic else np.linspace(-1, 1, 9)
            g = PiecewiseAffine(nodes, prng.uniform(-2, 2, len(nodes)), period=1 if periodic else None)
            cls = PeriodicHeteroFrontTracker if periodic else HeteroFrontTracker
            positions = np.sort(prng.uniform(.01, .99, 8) if periodic else prng.uniform(-.9, .9, 8))
            levels = prng.integers(-8, 9, 8 if periodic else 9)
            if not periodic:
                levels[0] = levels[-1] = 0
            tracker = cls(MultiplicativeFlux(g, .1, power), positions, levels, record_history=False)
            mass = tracker.integral() if periodic else tracker.integral(-20, 20)
            tv, count = tracker.total_variation(), tracker.front_count()
            for t in TIMES:
                tracker.run(t)
                worst = max(worst, invariants(tracker, mass, tv, count, tol))
            assert tracker.counts["multi_front_collision"] == 0
            cases += 1
        sign = (-1, 1)[(seed // 2) % 2]
        g = PiecewiseAffine(np.arange(8) / 8, sign * prng.uniform(.3, 3, 8), period=1)
        mapping = PeriodicStraightening(g, power=power)
        positions = np.sort(prng.uniform(.001, .999, 10))
        levels = prng.integers(-10, 11, 10)
        tracker = PeriodicHeteroFrontTracker(MultiplicativeFlux(g, .1, power), positions, levels,
                                              record_history=False)
        reference = PeriodicHeteroFrontTracker(
            MultiplicativeFlux(PiecewiseAffine.constant(mapping.coefficient, period=1), .1, power),
            mapping.coordinate(positions), levels, record_history=False)
        mass, tv, count = tracker.integral(), tracker.total_variation(), tracker.front_count()
        for t in TIMES:
            tracker.run(t)
            reference.run(t)
            assert [(f.kL, f.kR) for f in tracker.fronts] == [(f.kL, f.kR) for f in reference.fronts], (seed, power, t)
            np.testing.assert_allclose(mapping.coordinate(tracker.positions()), reference.positions(),
                                       atol=1e-10, rtol=0)
            worst = max(worst, invariants(tracker, mass, tv, count, tol))
        cases += 1
    print(f"PASS: {cases} randomized cases; largest mass residual {worst:.3e}")


if __name__ == "__main__":
    main()
