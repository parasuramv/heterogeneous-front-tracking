"""h(u) = |u|^p with p ≠ 2: closed forms, exact solutions, invariants and oracles."""
import bisect
import math
import unittest

import numpy as np

from hetfront import (HeteroFrontTracker, MultiplicativeFlux, PeriodicHeteroFrontTracker,
                      PiecewiseAffine, PowerLaw, trajectory)
from hetfront.homogenisation import PeriodicStraightening
from hetfront.trajectory import divided_power, divided_power_array

from . import _oracles

POWERS = (1.5, 3.0, 4.5)


def xu(power, delta=1.0):
    return MultiplicativeFlux(PiecewiseAffine([-1, 1], [-1, 1], extend="affine"), delta, power)


def rh_speed(flux, kL, kR, x):
    """Physical Rankine–Hugoniot speed [g|u|^p]/[u], computed without the straightening."""
    uL, uR = flux.u(kL, x), flux.u(kR, x)
    gx = flux.g(x)
    return gx * (abs(uR) ** flux.power - abs(uL) ** flux.power) / (uR - uL)


def rk4_front(flux, kL, kR, x0, T, steps=50000):
    """Integrate the physical RH ODE through every corner with RK4 (pure Python for speed).

    g_Δ is evaluated from its nodes, u = sgn(g)(|θ|/|g|)^{1/p} sgn θ, and the
    speed is [g|u|^p]/[u]. Corners make the right side only Lipschitz, so RK4
    converges slowly; 50000 steps resolve the fronts below to about 1e-9.
    """
    g, p, delta = flux.g, flux.power, flux.delta
    nodes, values = g.nodes.tolist(), g.values.tolist()
    left_slope = (values[1] - values[0]) / (nodes[1] - nodes[0]) if g.extend == "affine" else 0.0
    right_slope = (values[-1] - values[-2]) / (nodes[-1] - nodes[-2]) if g.extend == "affine" else 0.0

    def g_at(x):
        j = bisect.bisect_right(nodes, x)
        if j == 0:
            return values[0] + left_slope * (x - nodes[0])
        if j == len(nodes):
            return values[-1] + right_slope * (x - nodes[-1])
        a, b = nodes[j - 1], nodes[j]
        return values[j - 1] + (values[j] - values[j - 1]) * (x - a) / (b - a)

    def speed(x):
        gx = g_at(x)
        u = [math.copysign(1.0, gx) * math.copysign((abs(k) * delta / abs(gx)) ** (1 / p), k) if k else 0.0
             for k in (kL, kR)]
        return gx * (abs(u[1]) ** p - abs(u[0]) ** p) / (u[1] - u[0])

    x, h = x0, T / steps
    for _ in range(steps):
        k1 = speed(x)
        k2 = speed(x + 0.5 * h * k1)
        k3 = speed(x + 0.5 * h * k2)
        k4 = speed(x + h * k3)
        x += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x


class LawTests(unittest.TestCase):
    def test_rejects_invalid_powers(self):
        for p in (1.0, 0.5, -2, np.nan, np.inf):
            with self.subTest(p=p), self.assertRaises(ValueError):
                PowerLaw(p)
        with self.assertRaises(TypeError):
            PowerLaw(True)
        with self.assertRaises(ValueError):
            xu(1.0)

    def test_sigma_is_linear_and_consistent_with_position(self):
        # |g| = σ^m is affine with slope α along the front: σ^m - σ0^m = α (x - x0).
        x0, s0, kappa = 0.3, 1.2, 0.9
        for p in POWERS:
            law = PowerLaw(p)
            for alpha in (-0.7, 0.0, 1.3):
                for tau in (0.0, 0.4, 1.1):
                    with self.subTest(p=p, alpha=alpha, tau=tau):
                        x = law.position(x0, s0, alpha, kappa, tau)
                        s = law.sigma(s0, alpha, kappa, tau)
                        self.assertAlmostEqual(s, s0 + law.q * alpha * kappa * tau, places=14)
                        self.assertAlmostEqual(s ** law.m - s0 ** law.m, alpha * (x - x0), places=13)
                        if alpha == 0.0:
                            # Flat cell: constant speed |g|^{1/p} K in x.
                            self.assertAlmostEqual(x, x0 + s0 ** (law.m - 1) * kappa * tau, places=14)
                        # Travel time back to x recovers tau.
                        if tau:
                            self.assertAlmostEqual(law.travel_time(x0, s0, x, s, kappa), tau, places=13)

    def test_meeting_time_matches_positions(self):
        for p in POWERS:
            law = PowerLaw(p)
            alpha, xa, xb, ka, kb = 0.8, 0.1, 0.6, 1.4, -0.3
            sa = law.sigma_of(2.0 + alpha * xa)
            sb = law.sigma_of(2.0 + alpha * xb)
            tau = law.meeting_time(xa, sa, ka, xb, sb, kb)
            self.assertAlmostEqual(law.position(xa, sa, alpha, ka, tau),
                                   law.position(xb, sb, alpha, kb, tau), places=13)
            self.assertEqual(law.meeting_time(xa, sa, kb, xb, sb, ka), math.inf)

    def test_quadratic_law_is_the_unchanged_module_code(self):
        law = PowerLaw(2)
        self.assertTrue(law.quadratic)
        args = (0.3, 1.2, -0.7, 0.9)
        for tau in (0.0, 0.4, 1.1):
            self.assertEqual(law.position(*args, tau), trajectory.position(*args, tau))
            self.assertEqual(law.sigma(*args[1:], tau), trajectory.sigma(*args[1:], tau))
        self.assertEqual(law.travel_time(0.0, 1.0, 0.7, 1.3, 0.4), trajectory.travel_time(0.0, 1.0, 0.7, 1.3, 0.4))
        self.assertEqual(law.meeting_time(0.0, 1.0, 0.9, 0.5, 1.1, 0.2),
                         trajectory.meeting_time(0.0, 1.0, 0.9, 0.5, 1.1, 0.2))

    def test_laws_accept_time_arrays(self):
        taus = np.array([0.0, 0.4, 1.1, 50.0])
        for p in (1.5, 2.0, 3.0):
            law = PowerLaw(p)
            np.testing.assert_allclose(law.sigma(1.2, -0.7, 0.9, taus),
                                       [law.sigma(1.2, -0.7, 0.9, t) for t in taus], rtol=1e-15)
            np.testing.assert_allclose(law.position(0.3, 1.2, -0.7, 0.9, taus[:3]),
                                       [law.position(0.3, 1.2, -0.7, 0.9, t) for t in taus[:3]], rtol=1e-14)

    def test_divided_power_is_accurate_for_close_and_distant_arguments(self):
        for e in (1 / 3, 2 / 3, 1.5, 3.0):
            for a in (1e-3, 0.7, 40.0):
                with self.subTest(e=e, a=a):
                    self.assertAlmostEqual(divided_power(a, a, e) / (e * a ** (e - 1)), 1.0, places=14)
                    # Close arguments: the derivative at the midpoint to O(h²).
                    h = 1e-9 * a
                    mid = e * (a + h / 2) ** (e - 1)
                    self.assertAlmostEqual(divided_power(a, a + h, e) / mid, 1.0, places=12)
                    b = 3.7 * a
                    self.assertAlmostEqual(divided_power(a, b, e) / ((b ** e - a ** e) / (b - a)), 1.0, places=13)
            self.assertAlmostEqual(divided_power(0.0, 2.0, e), 2.0 ** (e - 1))
            self.assertAlmostEqual(divided_power(1e-200, 1.0, e), 1.0, places=12)
            a = np.array([0.0, 0.5, 0.5, 2.0, 1e-200])
            b = np.array([2.0, 0.5, 0.5 + 1e-10, 0.3, 1.0])
            np.testing.assert_allclose(divided_power_array(a, b, e),
                                       [divided_power(x, y, e) for x, y in zip(a, b)], rtol=1e-14)


class FluxTests(unittest.TestCase):
    def test_kappa_is_the_physical_rankine_hugoniot_speed(self):
        rng = np.random.default_rng(5)
        g = PiecewiseAffine.from_function(lambda x: 2 + np.sin(3 * x), np.linspace(-2, 2, 21))
        neg = PiecewiseAffine.from_function(lambda x: -1.5 - 0.5 * np.cos(2 * x), np.linspace(-2, 2, 21))
        for p in POWERS:
            for het in (g, neg):
                flux = MultiplicativeFlux(het, 0.05, p)
                pairs = [(k, k + 1) for k in (-7, -1, 0, 3, 400)] + [tuple(rng.integers(-30, 31, 2)) for _ in range(20)]
                for kL, kR in pairs:
                    if kL == kR:
                        continue
                    x = float(rng.uniform(-1.9, 1.9))
                    c = het.cell_of(x)
                    speed = abs(het(x)) ** (1 / p) * flux.kappa(kL, kR)
                    with self.subTest(p=p, kL=kL, kR=kR):
                        self.assertAlmostEqual(speed, rh_speed(flux, kL, kR, x), places=11)
                        self.assertAlmostEqual(flux.sigma(c, x), abs(het(x)) ** flux.law.q, places=14)

    def test_fan_steps_separate_and_characteristics_match(self):
        for p in POWERS:
            flux = xu(p, delta=0.1)
            kappas = [flux.kappa(k, k + 1) for k in range(-12, 12)]
            self.assertTrue(np.all(np.diff(kappas) > 0))
            for k in (-5, 0, 7):
                x = 0.37
                expected = abs(x) ** (1 / p) * flux.law.characteristic_kappa(k * flux.delta)
                self.assertAlmostEqual(float(flux.characteristic_speed(k, x)), expected, places=14)
                # h'(u) g at the stationary profile, from the physical variables.
                u = flux.u(k, x)
                self.assertAlmostEqual(expected, x * p * np.sign(u) * abs(u) ** (p - 1), places=12)

    def test_profiles_round_trip_and_blow_up_like_power(self):
        for p in POWERS:
            flux = xu(p, delta=0.1)
            x = np.array([-0.9, -0.2, 0.3, 0.8])
            levels = np.array([4, -3, 7, -2])
            np.testing.assert_allclose(flux.theta_of(x, flux.u(levels, x)), flux.theta(levels), rtol=1e-14)
            near = np.array([1e-4, 1e-6])
            np.testing.assert_allclose(flux.u(1, near) * near ** (1 / p), flux.s(1), rtol=1e-12)


class ExactSolutionTests(unittest.TestCase):
    def test_shock_into_the_zero(self):
        # g = x, θ_L = 0, θ_R = -1 from x0 = 1: K = -1, σ = x^q = 1 - q t, so
        # x(t) = (1 - q t)^m and the front is absorbed at t = 1/q = p/(p-1).
        for p in POWERS:
            law = PowerLaw(p)
            tracker = HeteroFrontTracker(xu(p), [1.0], [0, -1])
            arrival = 1 / law.q
            for t in (0.25 * arrival, 0.6 * arrival, 0.95 * arrival):
                tracker.run(t)
                self.assertAlmostEqual(tracker.fronts[0].x(t), (1 - law.q * t) ** law.m, places=13)
            tracker.run(arrival + 0.5)
            self.assertEqual((tracker.front_count(), tracker.counts["absorption"]), (0, 1))
            # Afterwards the zero releases flux f(0+) = 1 per unit time.
            self.assertAlmostEqual(tracker.mass_absorbed(), -0.5, places=12)

    def test_stationary_profile_integrals(self):
        for p in POWERS:
            q = 1 - 1 / p
            tracker = HeteroFrontTracker(xu(p), [], [-1])
            # u = -x^{-1/p} on x > 0 and |x|^{-1/p} on x < 0.
            self.assertAlmostEqual(tracker.integral(0.0, 4.0), -4.0 ** q / q, places=12)
            self.assertAlmostEqual(tracker.integral(-1.0, 0.0), 1 / q, places=12)
            self.assertAlmostEqual(tracker.integral(0.25, 2.0), -(2.0 ** q - 0.25 ** q) / q, places=12)


class CornerCrossingTests(unittest.TestCase):
    def check(self, g, kL, kR, x0, T, p, delta=0.1):
        flux = MultiplicativeFlux(g, delta, p)
        tracker = HeteroFrontTracker(flux, [x0], [kL, kR])
        self.assertEqual(tracker.front_count(), 1)
        tracker.run(T)
        self.assertGreater(tracker.counts["corner"], 2)
        self.assertAlmostEqual(tracker.fronts[0].x(T), rk4_front(flux, kL, kR, x0, T), places=8)

    def test_shock_through_many_corners(self):
        g = PiecewiseAffine.from_function(lambda x: 2 + np.sin(3 * x), np.linspace(-4, 4, 33))
        for p in (1.5, 3.0):
            self.check(g, 30, 5, -1.0, 2.0, p)

    def test_leftward_front_in_negative_g(self):
        g = PiecewiseAffine.from_function(lambda x: -1.5 - 0.5 * np.cos(2 * x), np.linspace(-4, 4, 41))
        for p in (1.5, 3.0):
            self.check(g, -6, -20, 1.5, 1.5, p)

    def test_front_through_flat_cells(self):
        g = PiecewiseAffine([-2, -1, 0, 1, 2], [1, 1, 2, 2, 3])
        for p in (1.5, 3.0):
            self.check(g, 25, 0, -2.5, 3.0, p)

    def test_mixed_sign_shock(self):
        g = PiecewiseAffine.from_function(lambda x: 2 + np.sin(3 * x), np.linspace(-4, 4, 33))
        for p in (1.5, 3.0):
            self.check(g, 12, -4, -1.0, 2.0, p)


class InvariantTests(unittest.TestCase):
    def test_line_invariants_and_exact_mass(self):
        rng = np.random.default_rng(11)
        for p in POWERS:
            for signed in (False, True):
                nodes = np.linspace(-1, 1, 9)
                values = rng.uniform(-2, 2, len(nodes)) if signed else rng.uniform(.4, 2.5, len(nodes))
                g = PiecewiseAffine(nodes, values)
                positions = np.sort(rng.uniform(-.9, .9, 10))
                levels = rng.integers(-8, 9, 11)
                levels[0] = levels[-1] = 0  # no inflow from infinity
                tracker = HeteroFrontTracker(MultiplicativeFlux(g, .1, p), positions, levels)
                mass, tv, count = tracker.integral(-20, 20), tracker.total_variation(), tracker.front_count()
                for t in (.1, .3, .8, 2.0, 5.0):
                    with self.subTest(p=p, signed=signed, t=t):
                        tracker.run(t)
                        tracker.check_levels()
                        self.assertTrue(np.all(np.diff(tracker.positions()) >= -1e-12))
                        self.assertLessEqual(tracker.total_variation(), tv + 1e-12)
                        self.assertLessEqual(tracker.front_count(), count)
                        self.assertTrue(tracker.interface_admissible())
                        self.assertEqual(tracker.counts["multi_front_collision"], 0)
                        self.assertEqual(tracker.counts["emission"], 0)
                        budget = tracker.integral(-20, 20) + tracker.mass_absorbed() - mass
                        self.assertLess(abs(budget), 1e-12)
                        tv, count = tracker.total_variation(), tracker.front_count()

    def test_l1_contraction(self):
        g = PiecewiseAffine.from_function(lambda x: 2 + np.sin(3 * x), np.linspace(-4, 4, 33))
        for p in (1.5, 3.0):
            flux = MultiplicativeFlux(g, 0.05, p)
            a = HeteroFrontTracker(flux, [-1.0, 0.0, 1.0], [0, 20, -6, 0])
            b = HeteroFrontTracker(flux, [-0.8, 0.3, 1.2], [0, 14, -10, 0])
            previous = a.u_l1_distance(b, -6, 6)
            for t in (.2, .5, 1.0, 2.0):
                a.run(t), b.run(t)
                distance = a.u_l1_distance(b, -6, 6)
                self.assertLessEqual(distance, previous + 1e-11)
                previous = distance

    def test_observed_budget_with_edge_crossings(self):
        from hetfront.visualization import observe_tracker
        g = PiecewiseAffine.from_function(lambda x: x * (1 - x * x), np.linspace(-2, 2, 41))
        for p in (1.5, 3.0):
            tracker = HeteroFrontTracker.from_function(
                MultiplicativeFlux(g, 0.02, p), lambda x: 0.6 * np.sin(2 * x), (-1.8, 1.8), cells=40)
            d = observe_tracker(tracker, np.linspace(-1.3, 1.4, 60), np.linspace(0, 1.5, 4)).diagnostics
            budget = d["mass"] + d["absorbed"] - d["inflow"]
            self.assertLess(np.max(np.abs(budget - budget[0])), 1e-12)
            self.assertGreater(np.ptp(d["edge_flux"]), 0)


class CircleTests(unittest.TestCase):
    def test_exact_conjugacy_with_the_effective_homogeneous_flux(self):
        rng = np.random.default_rng(823)
        for p in (1.5, 3.0):
            for sign in (-1, 1):
                g = PiecewiseAffine(np.arange(12) / 12, sign * (2 + rng.uniform(-.8, .8, 12)), period=1)
                mapping = PeriodicStraightening(g, power=p)
                # Effective coefficient η<|g|^{-1/p}>^{-p}, checked by direct quadrature.
                xs = np.linspace(0, 1, 200001)
                mean = np.trapezoid(np.abs(g(xs)) ** (-1 / p), xs)
                self.assertAlmostEqual(mapping.coefficient, sign / mean ** p, places=8)
                hg = PiecewiseAffine.constant(mapping.coefficient, period=1)
                positions = np.sort(rng.uniform(.02, .98, 8))
                levels = rng.integers(-8, 9, 8)
                het = PeriodicHeteroFrontTracker(MultiplicativeFlux(g, .1, p), positions, levels)
                hom = PeriodicHeteroFrontTracker(MultiplicativeFlux(hg, .1, p),
                                                 mapping.coordinate(positions), levels)
                mass = het.integral()
                for t in (.07, .2, .6, 2.):
                    with self.subTest(p=p, sign=sign, t=t):
                        het.run(t), hom.run(t)
                        het.check_levels()
                        self.assertEqual([(f.kL, f.kR) for f in het.fronts], [(f.kL, f.kR) for f in hom.fronts])
                        np.testing.assert_allclose(mapping.coordinate(het.positions()), hom.positions(), atol=3e-12)
                        self.assertAlmostEqual(het.integral(), mass, places=12)
                        self.assertAlmostEqual(het.integral(), hom.integral(), places=11)

    def test_stationary_corrector_is_exact_for_a_constant_level(self):
        # u = η s(θ)/|g|^{1/p} equals ρ times the homogeneous constant state.
        for p in (1.5, 3.0):
            for sign in (-1, 1):
                g = PiecewiseAffine([0, .3, .5], sign * np.array([1., 4., 2.]), period=1)
                mapping = PeriodicStraightening(g, power=p)
                hg = PiecewiseAffine.constant(mapping.coefficient, period=1)
                het = PeriodicHeteroFrontTracker(MultiplicativeFlux(g, .1, p), [], [5])
                hom = PeriodicHeteroFrontTracker(MultiplicativeFlux(hg, .1, p), [], [5])
                self.assertLess(mapping.corrector_l1(het, hom), 1e-13)
                self.assertAlmostEqual(het.integral(), hom.integral(), places=13)

    def test_corrector_rejects_mismatched_power(self):
        g = PiecewiseAffine([0, .5], [1., 4.], period=1)
        mapping = PeriodicStraightening(g, power=3)
        het = PeriodicHeteroFrontTracker(MultiplicativeFlux(g, .1), [], [2])
        hom = PeriodicHeteroFrontTracker(
            MultiplicativeFlux(PiecewiseAffine.constant(mapping.coefficient, period=1), .1), [], [2])
        with self.assertRaises(ValueError):
            mapping.corrector_l1(het, hom)


@unittest.skipIf(_oracles.fronttrack() is None, "fronttrack (scalar-front-tracking) not available")
class HomogeneousOracleTests(unittest.TestCase):
    """For constant g the scheme is Dafermos front tracking for b|u|^p on the grid U[δZ]."""

    def compare(self, b, p, seed):
        ft = _oracles.fronttrack()
        rng = np.random.default_rng(seed)
        flux = MultiplicativeFlux(PiecewiseAffine.constant(b), 0.05, p)
        ks = np.arange(-40, 41)
        grid = np.sign(b) * flux.s(ks) / abs(b) ** (1 / p)
        order = np.argsort(grid)
        reference_flux = ft.DiscreteFlux(lambda u: b * np.abs(u) ** p, grid[order])
        positions = np.sort(rng.uniform(-3, 3, 25))
        levels = rng.integers(-30, 31, 26)
        indices = np.searchsorted(grid[order], np.sign(b) * flux.s(levels) / abs(b) ** (1 / p))
        ours = HeteroFrontTracker(flux, positions, levels)
        reference = ft.FrontTracker(reference_flux, positions, indices)
        for t in (0.5, 1.0, 2.0, 4.0):
            ours.run(t), reference.run(t)
            self.assertEqual([(f.kL, f.kR) for f in ours.fronts],
                             [(ks[order][f.iL], ks[order][f.iR]) for f in reference.fronts])
            np.testing.assert_allclose(ours.positions(), [f.x for f in reference.fronts], atol=1e-11)

    def test_convex(self):
        for p in (1.5, 3.0):
            with self.subTest(p=p):
                self.compare(1.5, p, 3)

    def test_concave(self):
        for p in (1.5, 3.0):
            with self.subTest(p=p):
                self.compare(-0.7, p, 4)


if __name__ == "__main__":
    unittest.main()
