"""Circle driver: wraparound, exact mass, sign arcs through the seam, oracle agreement."""
import math
import unittest

import numpy as np

from hetfront import Circle, MultiplicativeFlux, PeriodicHeteroFrontTracker, PiecewiseAffine

from . import _oracles
from .test_solver import assert_links


def run_checks(test, tracker, times, *, budget=True, tol=1e-12):
    m0 = tracker.integral()
    tv, count = tracker.total_variation(), tracker.front_count()
    for t in times:
        tracker.run(t)
        tracker.check_levels()
        test.assertTrue(np.all(np.diff(tracker.positions()) >= -1e-12))
        test.assertLessEqual(tracker.total_variation(), tv + 1e-12)
        test.assertLessEqual(tracker.front_count(), count)
        test.assertTrue(tracker.interface_admissible())
        tv, count = tracker.total_variation(), tracker.front_count()
        if budget:
            test.assertLess(abs(tracker.integral() + tracker.mass_absorbed() - m0), tol)


class WellSignedCircleTests(unittest.TestCase):
    def setUp(self):
        L = 2 * np.pi
        self.g = PiecewiseAffine.from_function(lambda x: 2 + np.sin(x),
                                               np.linspace(0, L, 24, endpoint=False), period=L)

    def test_mass_conserved_over_many_circuits(self):
        flux = MultiplicativeFlux(self.g, 0.02)
        tracker = PeriodicHeteroFrontTracker.from_function(
            flux, lambda x: 1 + 0.8 * np.sin(x) + 0.3 * np.cos(3 * x), cells=60)
        run_checks(self, tracker, np.linspace(0.2, 20, 60))
        self.assertGreater(tracker.counts["corner"], 5000)
        self.assertEqual(tracker.counts["absorption"], 0)

    def test_single_front_circulates_with_period_of_travel(self):
        # A small positive jump moves right forever; its circuit time is
        # Σ over cells of the closed-form travel times.
        flux = MultiplicativeFlux(self.g, 0.01)
        tracker = PeriodicHeteroFrontTracker(flux, [1.0, 2.0], [101, 100])
        circuit = 0.0
        kappa = tracker.fronts[0].kappa
        for c in range(self.g.n_cells):
            lo, hi = self.g.left[c], self.g.right[c]
            circuit += 2 * (hi - lo) / (kappa * (self.g.sigma_left[c] + self.g.sigma_right[c]))
        # Both fronts have κ = s(100) + s(101), so they circulate in lockstep.
        self.assertEqual(tracker.fronts[0].kappa, tracker.fronts[1].kappa)
        tracker.run(circuit)
        self.assertEqual(tracker.counts["collision"], 0)
        np.testing.assert_allclose(sorted(tracker.domain.wrap(tracker.positions())), [1.0, 2.0], atol=1e-9)

    def test_constant_state_and_sampling(self):
        flux = MultiplicativeFlux(self.g, 0.1)
        tracker = PeriodicHeteroFrontTracker(flux, [], [3])
        self.assertEqual(tracker.front_count(), 0)
        self.assertEqual(tracker.sample_levels(17.3), 3)
        expected = math.fsum(
            2 * (self.g.right[c] - self.g.left[c]) * flux.s(3)
            / (self.g.sigma_left[c] + self.g.sigma_right[c]) for c in range(self.g.n_cells))
        self.assertAlmostEqual(tracker.integral(), expected, places=12)

    def test_complete_annihilation_leaves_constant(self):
        flux = MultiplicativeFlux(PiecewiseAffine.constant(1.0, period=1.0), 0.25)
        tracker = PeriodicHeteroFrontTracker(flux, [0.25, 0.75], [4, 0])
        tracker.run(50.0)
        self.assertEqual(tracker.front_count(), 0)
        self.assertEqual(tracker.sample_levels(0.5), tracker.constant_level)
        self.assertAlmostEqual(tracker.integral(), 0.5 * flux.s(4))


class SignChangingCircleTests(unittest.TestCase):
    def test_arc_through_seam(self):
        # Zeros at 0.3 and 0.8: the arc (0.8, 1.3) crosses the chart seam.
        g = PiecewiseAffine.from_function(lambda x: np.sin(2 * np.pi * (x - 0.8)),
                                          np.linspace(0, 1, 20, endpoint=False), period=1.0)
        flux = MultiplicativeFlux(g, 0.01)
        tracker = PeriodicHeteroFrontTracker.from_function(
            flux, lambda x: 0.5 * np.cos(2 * np.pi * x) + 0.2, cells=50)
        self.assertEqual(len(tracker.interfaces), 2)
        run_checks(self, tracker, np.linspace(0.1, 5, 50), tol=1e-11)
        self.assertGreater(tracker.counts["absorption"], 0)
        self.assertEqual(tracker.counts["emission"], 0)

    def test_links_stay_cyclic_through_seams_and_absorptions(self):
        g = PiecewiseAffine.from_function(lambda x: np.sin(2 * np.pi * x) + 0.3 * np.sin(6 * np.pi * x),
                                          np.linspace(0, 1, 25)[:-1], period=1.0)
        tracker = PeriodicHeteroFrontTracker.from_function(MultiplicativeFlux(g, 0.02),
                                                           lambda x: np.sin(2 * np.pi * x), cells=80)
        for t in (0.05, 0.3, 1.0, 3.0):
            tracker.run(t)
            assert_links(self, tracker)
        self.assertGreater(tracker.counts["absorption"], 10)
        self.assertGreater(tracker.counts["collision"], 10)

    def test_local_reordering_matches_global_argmax(self):
        """_normalize compares only touched and remembered pairs; check it
        against the full descent argmax at every call, including coincident
        fronts that cross the seam one corner event at a time."""
        stats = {"calls": 0, "stale": 0}
        test = self

        class Checked(PeriodicHeteroFrontTracker):
            def _normalize(self, touched=None):
                expect = None
                if touched is not None and self._count >= 2:
                    pos = self.positions()
                    expect = self.items[int(np.argmax(np.roll(pos, 1) - pos))]
                rotated = super()._normalize(touched)
                if expect is not None:
                    test.assertIs(self._head, expect)
                    stats["calls"] += 1
                    stats["stale"] = max(stats["stale"], len(self._unsorted))
                return rotated

        g = PiecewiseAffine.from_function(lambda x: 2 + 0.9 * np.cos(4 * np.pi * x),
                                          np.linspace(0, 1, 17)[:-1], period=1.0)
        flux = MultiplicativeFlux(g, 0.05)
        Checked(flux, np.arange(8) / 8, np.tile([10, -10], 4), enforce_interface=False).run(5.0)
        self.assertGreater(stats["stale"], 0)  # the simultaneous-crossing case occurred
        rng = np.random.default_rng(7)
        for _ in range(6):
            pos = np.sort(rng.choice(np.arange(400) / 400, 50, replace=False))
            Checked(flux, pos, rng.integers(-15, 16, 50)).run(3.0)
        self.assertGreater(stats["calls"], 1000)

    def test_fronts_reaching_the_seam_together_collide_first(self):
        """A front reaching the seam where a front it closes on already sits
        must collide, not cross past it: the chart coordinate jumps by L, and
        the order broke, losing mass (16% here) or stopping the run."""
        g = PiecewiseAffine.from_function(lambda x: 2 + 0.9 * np.cos(4 * np.pi * x),
                                          np.linspace(0, 1, 17)[:-1], period=1.0)
        tracker = PeriodicHeteroFrontTracker(MultiplicativeFlux(g, 0.05), np.arange(36) / 36,
                                             np.tile([10, -10], 18), enforce_interface=False)
        mass = tracker.integral()
        for t in (0.7, 2.0, 9.0):
            tracker.run(t)
            self.assertAlmostEqual(tracker.integral(), mass, delta=1e-12)
            tracker.check_levels()

    def test_interface_layer_on_circle(self):
        g = PiecewiseAffine.from_function(lambda x: np.sin(2 * np.pi * x),
                                          np.linspace(0, 1, 16, endpoint=False), period=1.0)
        flux = MultiplicativeFlux(g, 0.1)
        tracker = PeriodicHeteroFrontTracker(flux, [], [2], interface_layer=0.05)
        self.assertEqual(tracker.sample_levels(np.array([0.98, 0.02, 0.48, 0.25])).tolist(), [0, 0, 0, 2])


@unittest.skipIf(_oracles.circle_fronttrack() is None, "fronttrack.periodic (scalar-front-tracking) not available")
class CircleOracleTests(unittest.TestCase):
    """Constant g on S¹: identical to fronttrack.PeriodicFrontTracker on U[δZ]."""

    def test_constant_g_matches_fronttrack_periodic(self):
        ft = _oracles.circle_fronttrack()
        rng = np.random.default_rng(7)
        b, delta, L = 1.3, 0.05, 1.0
        flux = MultiplicativeFlux(PiecewiseAffine.constant(b, period=L), delta)
        ks = np.arange(-40, 41)
        grid = flux.s(ks) / np.sqrt(b)
        reference_flux = ft.DiscreteFlux(lambda u: b * u ** 2, grid)
        positions = np.sort(rng.uniform(0, L, 12))
        levels = rng.integers(-30, 31, 12)
        ours = PeriodicHeteroFrontTracker(flux, positions, levels)
        reference = ft.PeriodicFrontTracker(reference_flux, positions, levels + 40,
                                            domain=ft.Circle(length=L))
        x = np.linspace(0, L, 2001, endpoint=False) + 1e-7
        for t in (0.3, 1.0, 3.0):
            ours.run(t), reference.run(t)
            self.assertEqual(ours.front_count(), len(reference.fronts))
            np.testing.assert_array_equal(ours.sample_levels(x), np.searchsorted(grid, reference.sample(x)) - 40)
            self.assertAlmostEqual(ours.integral(), reference.integral() , places=10)


if __name__ == "__main__":
    unittest.main()
