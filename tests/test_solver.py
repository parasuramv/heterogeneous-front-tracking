"""Whole-line driver: exact xu² solutions, invariants, interfaces, oracle agreement."""
import unittest

import numpy as np

from hetfront import HeteroFrontTracker, MultiplicativeFlux, PiecewiseAffine
from hetfront.fronts import Front, Interface

from . import _oracles


def xu2(delta=1.0):
    return MultiplicativeFlux(PiecewiseAffine([-1, 1], [-1, 1], extend="affine"), delta)


def assert_invariants(test, tracker, times, *, window=None, mass_tol=1e-12):
    """Monotone TV and front count, ordered positions, intact level chain, mass budget.

    The budget assumes no front crosses the window edges, so their boundary
    fluxes f(A), f(B) stay constant.
    """
    tv, count = tracker.total_variation(), tracker.front_count()
    if window:
        m0 = tracker.integral(*window)
        edges = np.array(window, dtype=float)
        f_a, f_b = tracker.flux.flux_value(tracker.sample_levels(edges), edges)
        t0 = tracker.time
    for t in times:
        tracker.run(t)
        tracker.check_levels()
        test.assertTrue(np.all(np.diff(tracker.positions()) >= -1e-12))
        test.assertLessEqual(tracker.total_variation(), tv + 1e-12)
        test.assertLessEqual(tracker.front_count(), count)
        test.assertTrue(tracker.interface_admissible())
        tv, count = tracker.total_variation(), tracker.front_count()
        if window:
            inflow = (f_a - f_b) * (tracker.time - t0)
            budget = tracker.integral(*window) + tracker.mass_absorbed() - m0 - inflow
            test.assertLess(abs(budget), mass_tol)


class ExactSolutionTests(unittest.TestCase):
    def test_shock_reaches_zero_at_t2_and_is_absorbed(self):
        tracker = HeteroFrontTracker(xu2(), [1.0], [0, -1])
        for t in (0.5, 1.0, 1.9):
            tracker.run(t)
            self.assertAlmostEqual(tracker.fronts[0].x(t), (t - 2) ** 2 / 4, places=14)
        tracker.run(2.5)
        self.assertEqual(tracker.front_count(), 0)
        self.assertEqual(tracker.counts["absorption"], 1)
        (iface,) = tracker.interfaces
        self.assertEqual((iface.kL, iface.kR), (0, -1))
        # After t = 2 the zero releases flux f(0+) = 1, matching the stationary
        # profile -1/sqrt(x) (an entropy solution that is not a weak solution).
        self.assertAlmostEqual(tracker.mass_absorbed(), -0.5)

    def test_blowup_matches_characteristics(self):
        # u0 = -1 on |x| <= 1: exactly u = -1/(1-t) for |x| < (1-t)², t < 1.
        flux = xu2(delta=2e-4)
        tracker = HeteroFrontTracker.from_function(flux, lambda x: np.where(np.abs(x) <= 1, -1.0, 0.0),
                                                   (-1.5, 1.5), cells=600)
        for t in (0.2, 0.5):
            tracker.run(t)
            for x in (-0.3 * (1 - t) ** 2, 0.5 * (1 - t) ** 2):
                self.assertAlmostEqual(tracker.sample_u(x) * (1 - t), -1.0, delta=0.03)
        self.assertTrue(tracker.interface_admissible())

    def test_blowup_mass_budget(self):
        flux = xu2(delta=0.01)
        tracker = HeteroFrontTracker.from_function(flux, lambda x: np.where(np.abs(x) <= 1, -1.0, 0.0),
                                                   (-1.5, 1.5), cells=120)
        assert_invariants(self, tracker, np.linspace(0.1, 2.0, 20), window=(-3, 3), mass_tol=1e-11)
        self.assertGreater(tracker.counts["absorption"], 0)
        self.assertEqual(tracker.counts["emission"], 0)


class WellSignedTests(unittest.TestCase):
    def test_invariants_and_exact_mass(self):
        g = PiecewiseAffine.from_function(lambda x: 2 + np.sin(3 * x), np.linspace(-6, 6, 61))
        flux = MultiplicativeFlux(g, 0.02)
        u0 = lambda x: np.where(np.abs(x) < 1, 1.5 * np.cos(np.pi * x / 2) - 0.5, 0.0)
        tracker = HeteroFrontTracker.from_function(flux, u0, (-2, 2), cells=80)
        assert_invariants(self, tracker, np.linspace(0.1, 4, 40), window=(-10, 10))
        self.assertEqual(tracker.counts["multi_front_collision"], 0)
        self.assertGreater(tracker.counts["corner"], 100)

    def test_l1_contraction(self):
        g = PiecewiseAffine.from_function(lambda x: 1.5 + np.cos(2 * x), np.linspace(-5, 5, 41))
        flux = MultiplicativeFlux(g, 0.01)
        a = HeteroFrontTracker.from_function(flux, lambda x: np.exp(-x * x), (-3, 3), cells=60)
        b = HeteroFrontTracker.from_function(flux, lambda x: 0.8 * np.exp(-(x - 0.3) ** 2) - 0.1,
                                             (-3, 3), cells=60)
        dist = lambda: a.u_l1_distance(b, -8, 8)
        d = dist()
        for t in (0.5, 1, 2, 3):
            a.run(t), b.run(t)
            new = dist()
            self.assertLessEqual(new, d + 1e-11)
            d = new


class InterfaceTests(unittest.TestCase):
    def test_inadmissible_data_emits_fan_from_zero(self):
        tracker = HeteroFrontTracker(xu2(delta=0.25), [0.0, 1.0], [0, 4, 0])  # θ = +1 on (0, 1)
        self.assertEqual(tracker.counts["initial_emission"], 1)
        self.assertTrue(tracker.interface_admissible())
        emitted = [f for f in tracker.fronts if f.x0 == 0.0]
        self.assertEqual(len(emitted), 4)
        self.assertTrue(all(f.kappa > 0 for f in emitted))
        tracker.run(3.0)
        tracker.check_levels()
        self.assertEqual(tracker.counts["emission"], 0)

    def test_passive_interface_keeps_nonselected_solution(self):
        # Balanced entropy fluxes theta_left = theta_right = 1 permit the
        # outgoing right profile. A zero left trace would violate entropy.
        tracker = HeteroFrontTracker(xu2(), [0.0, 1.0], [1, 1, 0], enforce_interface=False)
        self.assertTrue(tracker.interface_entropy_admissible())
        self.assertFalse(tracker.interface_admissible())
        tracker.run(1.0)
        self.assertAlmostEqual(tracker.sample_u(0.25), 2.0)
        self.assertEqual(tracker.front_count(), 1)

    def test_interface_layer_zeroes_data_near_zero(self):
        tracker = HeteroFrontTracker(xu2(delta=0.5), [], [-2], interface_layer=0.1)
        levels = tracker.sample_levels(np.array([-0.5, -0.05, 0.05, 0.5]))
        self.assertEqual(levels.tolist(), [-2, 0, 0, -2])
        with self.assertRaises(ValueError):
            HeteroFrontTracker(MultiplicativeFlux(PiecewiseAffine([-1, 0.1, 1], [-1, 1, -1]), 1.0),
                               [], [0], interface_layer=0.6)

    def test_two_zeros_independent_sign_intervals(self):
        g = PiecewiseAffine.from_function(lambda x: x * (1 - x * x), np.linspace(-2, 2, 21))
        flux = MultiplicativeFlux(g, 0.01)
        tracker = HeteroFrontTracker.from_function(flux, lambda x: 0.6 * np.sin(2 * x), (-1.8, 1.8),
                                                   cells=90)
        self.assertEqual(len(tracker.interfaces), 3)
        assert_invariants(self, tracker, np.linspace(0.1, 3, 30), window=(-12, 12), mass_tol=1e-11)


class EngineTests(unittest.TestCase):
    def test_event_limit_preserves_state(self):
        g = PiecewiseAffine.from_function(lambda x: 2 + np.sin(x), np.linspace(-3, 3, 31))
        tracker = HeteroFrontTracker(MultiplicativeFlux(g, 0.1), [0.0], [10, 0], max_events=3)
        with self.assertRaises(RuntimeError):
            tracker.run(10.0)
        tracker.check_levels()

    def test_backward_time_rejected(self):
        tracker = HeteroFrontTracker(xu2(), [1.0], [0, -1]).run(1.0)
        with self.assertRaises(ValueError):
            tracker.run(0.5)

    def test_exact_integral_of_stationary_profile(self):
        tracker = HeteroFrontTracker(xu2(), [], [-1])
        # u = -1/sqrt(x) on x > 0 and +1/sqrt(-x) on x < 0 (θ = -1 both sides, g = x).
        self.assertAlmostEqual(tracker.integral(0.0, 4.0), -4.0)
        self.assertAlmostEqual(tracker.integral(-1.0, 0.0), 2.0)

    def test_history_pieces_cover_paths(self):
        g = PiecewiseAffine.from_function(lambda x: 2 + np.sin(x), np.linspace(-3, 3, 13))
        tracker = HeteroFrontTracker(MultiplicativeFlux(g, 0.1), [0.0], [10, 0]).run(2.0)
        pieces = tracker.pieces_until()
        self.assertEqual(len(pieces), tracker.counts["corner"] + 1)
        last = pieces[-1]
        self.assertAlmostEqual(last.t1, 2.0)


class BoundaryFluxTests(unittest.TestCase):
    def test_flux_through_matches_exact_shock(self):
        # The shock x = (t-2)²/4 crosses x = 1/4 at t = 1; behind it θ = -1 and
        # g = x > 0, so f = 1. At the node x = 1 the front leaves at t = 0.
        tracker = HeteroFrontTracker(xu2(), [1.0], [0, -1]).run(1.9)
        times = np.linspace(0, 1.9, 39)
        np.testing.assert_allclose(tracker.flux_through(0.25, times), np.maximum(times - 1, 0),
                                   atol=1e-14, rtol=0)
        np.testing.assert_allclose(tracker.flux_through(1.0, times), times, atol=1e-14, rtol=0)
        np.testing.assert_array_equal(tracker.flux_through(-0.5, times), 0.0)

    def test_observed_budget_closes_when_fronts_cross_edges(self):
        from hetfront.visualization import observe_tracker
        rng = np.random.default_rng(7)
        crossings = 0
        for seed in range(6):
            nodes = np.linspace(-1, 1, 9)
            g = PiecewiseAffine(nodes, rng.uniform(-2, 2, len(nodes)), extend="affine")
            positions = np.sort(rng.uniform(-.9, .9, 10))
            levels = rng.integers(-8, 9, 11)
            # One edge at a coefficient node, one inside a cell; neither at a zero.
            window = (nodes[2], .37)
            if np.any(np.isin(window, g.zeros)):
                continue
            for frames in (3, 41):
                tracker = HeteroFrontTracker(MultiplicativeFlux(g, .1), positions, levels)
                with self.subTest(seed=seed, frames=frames):
                    d = observe_tracker(tracker, np.linspace(*window, 50), np.linspace(0, 1.5, frames),
                                        mass_window=window).diagnostics
                    budget = d["mass"] + d["absorbed"] - d["inflow"]
                    self.assertLess(np.max(np.abs(budget - budget[0])), 1e-12)
                    crossings += np.ptp(d["edge_flux"]) > 0
        self.assertGreater(crossings, 0)

    def test_flux_through_needs_history_and_rejects_zeros(self):
        with self.assertRaises(ValueError):
            HeteroFrontTracker(xu2(), [1.0], [0, -1], record_history=False).run(1).flux_through(.5, [0, 1])
        with self.assertRaises(ValueError):
            HeteroFrontTracker(xu2(), [1.0], [0, -1]).run(1).flux_through(0.0, [0, 1])
        with self.assertRaises(ValueError):
            HeteroFrontTracker(xu2(), [1.0], [0, -1]).run(1).flux_through(.5, [0, 2])


@unittest.skipIf(_oracles.fronttrack() is None, "fronttrack (scalar-front-tracking) not available")
class HomogeneousOracleTests(unittest.TestCase):
    """For constant g the scheme is Dafermos front tracking on the grid U[δZ]."""

    def compare(self, b, seed):
        ft = _oracles.fronttrack()
        rng = np.random.default_rng(seed)
        delta = 0.05
        flux = MultiplicativeFlux(PiecewiseAffine.constant(b), delta)
        ks = np.arange(-40, 41)
        grid = np.sign(b) * flux.s(ks) / np.sqrt(abs(b))
        order = np.argsort(grid)
        reference_flux = ft.DiscreteFlux(lambda u: b * u ** 2, grid[order])
        positions = np.sort(rng.uniform(-3, 3, 25))
        levels = rng.integers(-30, 31, 26)
        indices = np.searchsorted(grid[order], np.sign(b) * flux.s(levels) / np.sqrt(abs(b)))
        ours = HeteroFrontTracker(flux, positions, levels)
        reference = ft.FrontTracker(reference_flux, positions, indices)
        for t in (0.5, 1.0, 2.0, 4.0):
            ours.run(t), reference.run(t)
            self.assertEqual([(f.kL, f.kR) for f in ours.fronts],
                             [(ks[order][f.iL], ks[order][f.iR]) for f in reference.fronts])
            np.testing.assert_allclose(ours.positions(), [f.x for f in reference.fronts], atol=1e-12)

    def test_convex(self):
        self.compare(1.5, 3)

    def test_concave(self):
        self.compare(-0.7, 4)


if __name__ == "__main__":
    unittest.main()
