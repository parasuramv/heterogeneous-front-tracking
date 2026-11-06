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


class AlignedMeshTests(unittest.TestCase):
    """PiecewiseAffine.aligned: zeros are nodes, cells equal between zeros, nested refinement."""

    def check_arcs(self, gd, cells):
        nodes = np.asarray(gd.nodes)
        self.assertEqual(len(nodes) if gd.periodic else len(nodes) - 1, cells)
        ends = np.append(nodes, nodes[0] + gd.period) if gd.periodic else nodes
        widths = np.diff(ends)
        # Split the cells at the zeros; widths are equal within each arc.
        breaks = [i for i, x in enumerate(ends) if np.any(np.abs(np.asarray(gd.zeros) - gd.wrap(x)) < 1e-12)]
        if gd.periodic:
            breaks = breaks[:-1] if breaks and breaks[-1] == len(ends) - 1 else breaks
            breaks = breaks + [breaks[0] + len(widths)]
            widths = np.concatenate((widths, widths))
        else:
            breaks = sorted(set([0] + breaks + [len(ends) - 1]))
        for i, j in zip(breaks[:-1], breaks[1:]):
            arc = widths[i:j]
            self.assertLess(np.ptp(arc), 1e-12 * max(1.0, arc[0] * len(arc)))

    def test_circle_zeros_are_nodes_and_arcs_uniform(self):
        g = lambda x: np.sin(2 * np.pi * x) + 0.3
        for cells in (16, 32, 64, 128):
            gd = PiecewiseAffine.aligned(g, cells, period=1.0)
            self.assertEqual(len(gd.zeros), 2)
            for z in gd.zeros:
                self.assertAlmostEqual(np.sin(2 * np.pi * z), -0.3, places=12)
            self.check_arcs(gd, cells)
            self.assertTrue(0.0 <= gd.nodes[0] < 1 / cells * 1.2)

    def test_dyadic_refinement_is_nested(self):
        g = lambda x: np.sin(2 * np.pi * x) * (1 + 0.3 * np.sin(2 * np.pi * x)) + 0.25
        meshes = [PiecewiseAffine.aligned(g, 16 * 2 ** k, period=1.0) for k in range(4)]
        for coarse, fine in zip(meshes[:-1], meshes[1:]):
            fine_nodes = np.asarray(fine.nodes)
            for x in coarse.nodes:
                self.assertLess(np.min(np.abs(fine_nodes - x)), 1e-13)
            # Each zero keeps its place: the cells beside it halve exactly.
            for z in coarse.zeros:
                c, f = coarse.cell_of(z), fine.cell_of(z)
                self.assertAlmostEqual(coarse.right[c] - coarse.left[c],
                                       2 * (fine.right[f] - fine.left[f]), places=13)

    def test_line_arcs_include_the_interval_ends(self):
        g = lambda x: x * (1 - x * x)
        for cells in (15, 30, 60):
            gd = PiecewiseAffine.aligned(g, cells, interval=(-2, 2))
            np.testing.assert_allclose(gd.zeros, [-1, 0, 1], atol=1e-14)
            self.assertEqual((gd.nodes[0], gd.nodes[-1]), (-2.0, 2.0))
            self.check_arcs(gd, cells)
        # 15 cells over four unit arcs cannot be uniform: 4, 4, 4 and 3.
        widths = np.diff(PiecewiseAffine.aligned(g, 15, interval=(-2, 2)).nodes)
        np.testing.assert_allclose(np.unique(np.round(widths, 12)), [0.25, 1 / 3])

    def test_uniform_mesh_is_kept_when_already_aligned(self):
        g = lambda x: x * (1 - x * x)
        uniform = PiecewiseAffine.from_function(g, np.linspace(-2, 2, 81))
        aligned = PiecewiseAffine.aligned(g, 80, interval=(-2, 2))
        self.assertTrue(np.array_equal(uniform.nodes, aligned.nodes))
        self.assertTrue(np.array_equal(uniform.values, aligned.values))
        well = PiecewiseAffine.aligned(lambda x: 2 + np.sin(x), 12, period=2 * math.pi)
        np.testing.assert_array_equal(well.nodes, np.arange(12) * 2 * math.pi / 12)

    def test_close_zeros_get_two_cells(self):
        g = lambda x: (x - 0.3) * (x - 0.31)
        gd = PiecewiseAffine.aligned(g, 8, interval=(0.0, 1.0), beta=1e-3)
        np.testing.assert_allclose(gd.zeros, [0.3, 0.31], atol=1e-14)
        inside = [x for x in gd.nodes if 0.3 + 1e-12 < x < 0.31 - 1e-12]
        self.assertEqual(len(inside), 1)
        self.assertTrue(np.all(gd.values[[0, -1]] > 0))

    def test_arguments(self):
        g = lambda x: np.sin(x)
        with self.assertRaises(ValueError):
            PiecewiseAffine.aligned(g, 8)
        with self.assertRaises(ValueError):
            PiecewiseAffine.aligned(g, 8, interval=(0, 1), period=1)
        with self.assertRaises(ValueError):
            PiecewiseAffine.aligned(g, 8, interval=(1, 0))
        with self.assertRaises(ValueError):
            PiecewiseAffine.aligned(g, 0, interval=(0, 1))


class GradedMeshTests(unittest.TestCase):
    """PiecewiseAffine.aligned(..., grading=γ): same arcs and counts, cells graded towards zeros."""

    g = staticmethod(lambda x: np.sin(2 * np.pi * x) + 0.3)

    def test_map_is_monotone_c1_and_symmetric(self):
        from hetfront.heterogeneity import _graded
        for gamma in (1.25, 2.0, 3.0):
            for left, right in ((True, True), (True, False), (False, True)):
                w = _graded(64, left, right, gamma)
                self.assertEqual((w[0], w[-1]), (0.0, 1.0))
                self.assertTrue(np.all(np.diff(w) > 0))
            w = _graded(64, True, True, gamma)
            np.testing.assert_allclose(w + w[::-1], 1.0, atol=1e-15)
            d = np.diff(w) * 64
            self.assertAlmostEqual(d[31], d[32], places=12)          # C¹ at s = 1/2
            self.assertLess(abs(d[31] - gamma), 0.1)
        np.testing.assert_array_equal(_graded(8, True, True, 1.0), np.linspace(0, 1, 9))
        np.testing.assert_array_equal(_graded(8, False, False, 3.0), np.linspace(0, 1, 9))

    def test_same_zeros_and_counts_as_aligned(self):
        for cells in (16, 64):
            plain = PiecewiseAffine.aligned(self.g, cells, period=1.0)
            graded = PiecewiseAffine.aligned(self.g, cells, period=1.0, grading=2.0)
            np.testing.assert_allclose(graded.zeros, plain.zeros, atol=1e-15)
            self.assertEqual(len(graded.nodes), len(plain.nodes))
            lo, hi = sorted(plain.zeros)
            count = lambda gd: int(np.sum((gd.nodes > lo) & (gd.nodes < hi)))
            self.assertEqual(count(graded), count(plain))     # same cells on each arc

    def test_touching_cell_shrinks_like_h_to_the_gamma(self):
        for gamma in (1.5, 2.0, 3.0):
            sizes = []
            for cells in (32, 64, 128):
                gd = PiecewiseAffine.aligned(self.g, cells, period=1.0, grading=gamma)
                c = gd.cell_of(gd.zeros[0])
                sizes.append(gd.right[c] - gd.left[c])
            for a, b in zip(sizes, sizes[1:]):
                self.assertAlmostEqual(a / b, 2 ** gamma, places=6)

    def test_graded_refinement_is_nested(self):
        meshes = [PiecewiseAffine.aligned(self.g, 16 * 2 ** k, period=1.0, grading=2.5) for k in range(4)]
        for coarse, fine in zip(meshes[:-1], meshes[1:]):
            for x in coarse.nodes:
                self.assertLess(np.min(np.abs(np.asarray(fine.nodes) - x)), 1e-13)

    def test_line_grades_only_towards_zeros(self):
        g = lambda x: x * (1 - x * x)
        gd = PiecewiseAffine.aligned(g, 16, interval=(-2, 2), grading=2.0)
        widths = np.diff(gd.nodes)
        # Arc [-2, -1]: zero at its right end only, so the cells shrink towards -1.
        self.assertTrue(np.all(np.diff(widths[:4]) < 0))
        self.assertAlmostEqual(widths[3], 1 / 16)
        well = PiecewiseAffine.aligned(lambda x: 2 + np.sin(x), 12, interval=(0, 3), grading=3.0)
        np.testing.assert_allclose(np.diff(well.nodes), 0.25)

    def test_grading_argument(self):
        for bad in (0.5, math.inf, math.nan):
            with self.assertRaises(ValueError):
                PiecewiseAffine.aligned(self.g, 16, period=1.0, grading=bad)


if __name__ == "__main__":
    unittest.main()
