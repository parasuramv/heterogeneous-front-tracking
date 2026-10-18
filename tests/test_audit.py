"""Regression cases discovered by the October 2026 code/theory audit."""
import unittest

import numpy as np

from hetfront import HeteroFrontTracker, MultiplicativeFlux, PeriodicHeteroFrontTracker, PiecewiseAffine
from hetfront.homogenisation import PeriodicStraightening
from hetfront.studies import PROBLEMS, refinement


class InputTests(unittest.TestCase):
    def test_fractional_and_nonfinite_levels_rejected(self):
        for periodic in (False, True):
            cls = PeriodicHeteroFrontTracker if periodic else HeteroFrontTracker
            flux = MultiplicativeFlux(PiecewiseAffine.constant(1, period=1 if periodic else None), .1)
            for values in ([.9], [np.nan], [np.inf], [2.**63], [[1]], [True]):
                with self.subTest(periodic=periodic, values=values), self.assertRaises(ValueError):
                    cls(flux, [], values)
            for value in (np.nan, np.inf, 1e300):
                with self.assertRaises(ValueError):
                    cls.from_theta(flux, [], [value])
            for eps in (np.nan, np.inf, -1):
                with self.assertRaises(ValueError):
                    cls(flux, [], [0], interface_layer=eps)

    def test_disabling_selection_does_not_imply_entropy(self):
        g = PiecewiseAffine([-1, 1], [-1, 1], extend="affine")
        flux = MultiplicativeFlux(g, 1)
        invalid = HeteroFrontTracker(flux, [0, 1], [0, 1, 0], enforce_interface=False)
        valid = HeteroFrontTracker(flux, [0, 1], [1, 1, 0], enforce_interface=False)
        self.assertFalse(invalid.interface_entropy_admissible())
        self.assertTrue(valid.interface_entropy_admissible())
        self.assertFalse(valid.interface_admissible())

    def test_invalid_meshes_rejected_before_sampling(self):
        flux = MultiplicativeFlux(PiecewiseAffine.constant(1), .1)
        for cells in (0, -1, 1.2, True):
            with self.assertRaises(ValueError):
                HeteroFrontTracker.from_function(flux, lambda x: x, (0, 1), cells=cells)
        for window in ((1, 0), (0, 0), (0, np.inf)):
            with self.assertRaises(ValueError):
                HeteroFrontTracker.from_function(flux, lambda x: x, window)
        for option in ({'oversample': 0}, {'oversample': 1.5}, {'beta': np.nan}, {'period': np.inf}):
            with self.assertRaises(ValueError):
                PiecewiseAffine.from_function(lambda x: 1 + x * x, [0, .5], **option)

    def test_affine_extension_from_an_endpoint_zero(self):
        for nodes in ([-1., 0.], [0., 1.]):
            g = PiecewiseAffine(nodes, nodes, extend="affine")
            self.assertTrue(np.all(np.isfinite(g.sign)))
            flux = MultiplicativeFlux(g, 1)
            tr = HeteroFrontTracker(flux, [], [-1])
            self.assertAlmostEqual(tr.integral(0, 1), -2.)
            self.assertAlmostEqual(g.sigma(g.cell_of(1), 1), 1.)
            tr.run(.2)
            tr.check_levels()

    def test_zero_value_uses_right_trace_sign(self):
        g = PiecewiseAffine([-1, 1], [1, -1], extend="affine")
        self.assertEqual(MultiplicativeFlux(g, 1).u(1, 0), -np.inf)

    def test_positive_quartic_is_not_a_tangential_zero(self):
        # The old parabola through three samples crossed zero although g did not.
        g = PiecewiseAffine.from_function(lambda x: (x - .011)**4 + 1e-10, [-1, 0, .13, 1])
        self.assertTrue(g.is_well_signed)


class LiftedIntegralTests(unittest.TestCase):
    def test_integrals_translate_and_add_periods(self):
        for sign in (-1, 1):
            g = PiecewiseAffine([.2, .7], sign * np.array([1., 4.]), period=1.)
            flux = MultiplicativeFlux(g, .1)
            tr = PeriodicHeteroFrontTracker(flux, [.3, .8], [4, -2]).run(.4)
            mass = tr.integral()
            for start in (-2.35, .2, .63, 3.7):
                values = tr.primitive([start, start + 1, start + 2])
                np.testing.assert_allclose(values, [0, mass, 2 * mass], atol=2e-13)
            other = PeriodicHeteroFrontTracker(flux, [], [0])
            base = tr.theta_l1_distance(other, .2, 1.2)
            self.assertAlmostEqual(tr.theta_l1_distance(other, -1.8, 2.2), 4 * base, places=12)

    def test_primitive_rejects_nonfinite_grid(self):
        tr = HeteroFrontTracker(MultiplicativeFlux(PiecewiseAffine.constant(1), 1), [], [1])
        with self.assertRaises(ValueError):
            tr.primitive([0, np.nan])

    def test_leftward_characteristic_from_periodic_seam(self):
        g = PiecewiseAffine([0, .5], [1., 4.], period=1)
        flux = MultiplicativeFlux(g, 1)
        x, t = flux.characteristic(0, -1, .01)
        # On the last cell alpha=-6, x0=1, sigma0=1, kappa=-2.
        self.assertAlmostEqual(x[-1], 1 - 2 * .01 * (1 + .03), places=13)
        self.assertEqual(t[-1], .01)


class StraighteningTests(unittest.TestCase):
    def test_exact_conjugacy_for_signed_data_and_both_flux_signs(self):
        # A homogeneous tracker sees no coefficient corners. Agreement across
        # many collisions/seams tests the heterogeneous scheduler and its laws.
        rng = np.random.default_rng(823)
        for sign in (-1, 1):
            g = PiecewiseAffine(np.arange(12) / 12, sign * (2 + rng.uniform(-.8, .8, 12)), period=1)
            mapping = PeriodicStraightening(g)
            hg = PiecewiseAffine.constant(mapping.coefficient, period=1)
            positions = np.sort(rng.uniform(.02, .98, 8))
            levels = rng.integers(-8, 9, 8)
            het = PeriodicHeteroFrontTracker(MultiplicativeFlux(g, .1), positions, levels)
            hom = PeriodicHeteroFrontTracker(MultiplicativeFlux(hg, .1), mapping.coordinate(positions), levels)
            for t in (.07, .2, .6, 2.):
                het.run(t)
                hom.run(t)
                het.check_levels()
                hom.check_levels()
                self.assertEqual([(f.kL, f.kR) for f in het.fronts], [(f.kL, f.kR) for f in hom.fronts])
                np.testing.assert_allclose(mapping.coordinate(het.positions()), hom.positions(), atol=3e-12)
                self.assertAlmostEqual(het.integral(), hom.integral(), places=11)

    def test_stationary_corrector_and_effective_flux(self):
        for sign in (-1, 1):
            g = PiecewiseAffine([0, .5], sign * np.array([1., 4.]), period=1)
            mapping = PeriodicStraightening(g)
            self.assertAlmostEqual(mapping.mean_inverse_root, 2 / 3)
            self.assertAlmostEqual(mapping.coefficient, sign * 2.25)
            tr = PeriodicHeteroFrontTracker(MultiplicativeFlux(g, .1), [], [-4])
            hom = PeriodicHeteroFrontTracker(MultiplicativeFlux(PiecewiseAffine.constant(mapping.coefficient, period=1), .1), [], [-4])
            self.assertAlmostEqual(mapping.corrector_l1(tr, hom), 0, places=14)
            np.testing.assert_allclose(mapping.coordinate(np.array([-.7, .3, 1.3])), mapping.coordinate(.3) + np.array([-1, 0, 1]), atol=1e-14)

    def test_corrector_integral_and_displacement_extrema(self):
        g = PiecewiseAffine([0, .2, .6], [1., 3., 2.], period=1)
        mapping = PeriodicStraightening(g)
        het = PeriodicHeteroFrontTracker(MultiplicativeFlux(g, .1), [.15, .7], [3, -2]).run(.1)
        hg = PiecewiseAffine.constant(mapping.coefficient, period=1)
        hom = PeriodicHeteroFrontTracker(MultiplicativeFlux(hg, .1), [.15, .7], [3, -2]).run(.1)
        # Independent dense midpoint quadrature, away from singularities.
        x = (np.arange(200000) + .5) / 200000
        sampled = np.mean(np.abs(het.sample_u(x) - mapping.density(x) * hom.sample_u(x)))
        self.assertAlmostEqual(mapping.corrector_l1(het, hom), sampled, delta=2e-5)
        sampled_displacement = np.max(np.abs(mapping.coordinate(x) - x))
        self.assertGreaterEqual(mapping.displacement_bound() + 1e-14, sampled_displacement)
        self.assertAlmostEqual(mapping.displacement_bound(), sampled_displacement, places=9)

    def test_zeros_excluded(self):
        with self.assertRaises(ValueError):
            PeriodicStraightening(PiecewiseAffine([0, .5], [1, -1], period=1))

    def test_mesh_width_uses_intervals_on_line(self):
        problem = PROBLEMS['line-well-signed']
        ref = problem.tracker(.1, 9)
        row = refinement(problem, [(.1, 9)], ref)[0]
        self.assertEqual(row['h'], 1.)


if __name__ == '__main__':
    unittest.main()
