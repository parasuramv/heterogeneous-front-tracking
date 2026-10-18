import math
import unittest

import numpy as np

from hetfront import PiecewiseAffine


class ConstructionTests(unittest.TestCase):
    def test_nodal_interpolant_has_cell_average_derivative(self):
        g = lambda x: 2 + np.sin(x)
        nodes = np.linspace(-3, 3, 13)
        h = PiecewiseAffine.from_function(g, nodes)
        np.testing.assert_allclose(h(nodes), g(nodes), atol=1e-15)
        # Slope on each interior cell = average of g' = chord slope.
        chords = np.diff(g(nodes)) / np.diff(nodes)
        np.testing.assert_allclose(h.slope[1:-1], chords, rtol=1e-12)
        self.assertTrue(h.is_well_signed)

    def test_sign_change_inserts_zero_node(self):
        h = PiecewiseAffine([-1.0, 0.5], [-2.0, 1.0])
        np.testing.assert_allclose(h.zeros, [0.0], atol=1e-15)
        self.assertEqual(len(h.nodes), 3)

    def test_true_zeros_located_by_bisection(self):
        g = lambda x: x * (1 - x ** 2) + 0.1
        h = PiecewiseAffine.from_function(g, np.linspace(-2, 2, 7))
        for z in h.zeros:
            self.assertLess(abs(g(z)), 1e-12)
        self.assertEqual(len(h.zeros), 3)

    def test_two_zeros_inside_one_coarse_cell_are_found(self):
        g = lambda x: (x - 0.3) * (x - 0.6)
        h = PiecewiseAffine.from_function(g, np.array([-1.0, 2.0]), oversample=32)
        np.testing.assert_allclose(h.zeros, [0.3, 0.6], atol=1e-12)

    def test_tangential_zeros_rejected(self):
        for g in (lambda x: x ** 2, lambda x: x ** 3, lambda x: (x - 0.013) ** 2,
                  lambda x: -(x - 0.25) ** 4):
            with self.assertRaises(ValueError):
                PiecewiseAffine.from_function(g, np.linspace(-1, 1, 11))

    def test_beta_lower_bound_enforced(self):
        g = lambda x: 1e-3 * x
        PiecewiseAffine.from_function(g, np.linspace(-1, 1, 5), beta=1e-4)
        with self.assertRaises(ValueError):
            PiecewiseAffine.from_function(g, np.linspace(-1, 1, 5), beta=1e-2)

    def test_touching_zero_node_rejected(self):
        with self.assertRaises(ValueError):
            PiecewiseAffine([-1, 0, 1], [1, 0, 1])

    def test_affine_extension_inserts_outer_zero(self):
        h = PiecewiseAffine([1, 2], [1, 2], extend="affine")
        np.testing.assert_allclose(h.zeros, [0.0], atol=1e-15)
        self.assertAlmostEqual(h(-5.0), -5.0)
        self.assertAlmostEqual(h(7.0), 7.0)

    def test_constant_extension(self):
        h = PiecewiseAffine([0, 1], [2, 3])
        self.assertEqual(h(-10.0), 2.0)
        self.assertEqual(h(10.0), 3.0)
        self.assertEqual(h.alpha[0], 0.0)


class QueryTests(unittest.TestCase):
    def test_sigma_matches_sqrt_abs_g(self):
        h = PiecewiseAffine.from_function(lambda x: np.sin(x) - 0.2, np.linspace(-4, 4, 17),
                                          extend="affine")
        xs = np.linspace(-6, 6, 501)
        cells = h.cell_of(xs)
        np.testing.assert_allclose(h.sigma_array(xs, cells), np.sqrt(np.abs(h(xs))), atol=1e-13)
        for x in xs[::25]:
            c = h.cell_of(x)
            self.assertAlmostEqual(h.sigma(c, x), math.sqrt(abs(h(x))), places=13)

    def test_cell_signs_constant(self):
        h = PiecewiseAffine.from_function(lambda x: np.cos(3 * x), np.linspace(-2, 2, 23))
        for c in range(h.n_cells):
            lo, hi = h.left[c], h.right[c]
            lo = lo if np.isfinite(lo) else hi - 1
            hi = hi if np.isfinite(hi) else lo + 1
            inner = np.linspace(lo, hi, 9)[1:-1]
            self.assertTrue(np.all(np.sign(h(inner)) == h.sign[c]))

    def test_sign_intervals_line(self):
        h = PiecewiseAffine([-1, 1], [-1, 1], extend="affine")
        self.assertEqual(h.sign_intervals(), [(-math.inf, 0.0, -1.0), (0.0, math.inf, 1.0)])

    def test_periodic_wrap_and_arcs(self):
        L = 1.0
        h = PiecewiseAffine.from_function(lambda x: np.sin(2 * np.pi * (x - 0.8)),
                                          np.linspace(0, 1, 10, endpoint=False), period=L)
        np.testing.assert_allclose(h.zeros, [0.3, 0.8], atol=1e-12)
        arcs = h.sign_intervals()
        self.assertEqual(len(arcs), 2)
        self.assertGreater(arcs[-1][1], h.origin + L)  # the arc through the seam
        self.assertAlmostEqual(h(1.25), h(0.25))
        self.assertEqual(h.cell_of(0.0, side="left"), h.n_cells - 1)
        self.assertEqual(h.next_cell(h.n_cells - 1, 1), 0)


if __name__ == "__main__":
    unittest.main()
