"""General convex h: nonlinearity numerics and the per-cell ``GeneralLaw`` front laws."""
import math
import unittest

import numpy as np

from hetfront.nonlinearity import AbsPolynomial, Custom, Power, parse
from hetfront.trajectory import FrontCoef, GeneralLaw, PowerLaw

POLYS = {"u2+u4": AbsPolynomial({2: 1.0, 4: 1.0}), "u2+|u|3": AbsPolynomial({2: 1.0, 3: 1.0})}
FRONTS = [(2.0, 0.5), (0.5, 0.6), (-0.4, 1.3), (1.0, -2.0), (-3.0, -0.7), (0.0, -1.0), (0.0, 2.5)]


def secant_speed(h, thL, thR, gamma):
    """Physical [g h(u)]/[u] at |g| = γ > 0 for g > 0, with no use of the law's algebra."""
    uL = math.copysign(h.inverse(abs(thL) / gamma), thL) if thL else 0.0
    uR = math.copysign(h.inverse(abs(thR) / gamma), thR) if thR else 0.0
    return gamma * (h.h(abs(uR)) - h.h(abs(uL))) / (uR - uL)


def rk4(speed, x, T, steps=800):
    dt = T / steps
    for _ in range(steps):
        k1 = speed(x)
        k2 = speed(x + 0.5 * dt * k1)
        k3 = speed(x + 0.5 * dt * k2)
        k4 = speed(x + dt * k3)
        x += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x


class NonlinearityTests(unittest.TestCase):
    def test_osgood_and_inverse_match_closed_forms(self):
        for p in (1.1, 1.5, 3.0, 8.0):
            generic = AbsPolynomial({p: 1.0})  # numerical path, exact answer known
            for u in (1e-4, 0.3, 1.0, 7.0, 1e3):
                self.assertAlmostEqual(generic.osgood(u) / Power(p).osgood(u), 1.0, places=14)
                self.assertAlmostEqual(generic.inverse(generic.h(u)) / u, 1.0, places=14)
        h = POLYS["u2+u4"]
        for u in (0.01, 0.5, 1.0, 3.0):  # ∫ dr/(r²+r⁴) = 1/u - atan(1/u)
            self.assertAlmostEqual(h.osgood(u) / (1 / u - math.atan(1 / u)), 1.0, places=13)

    def test_divided_differences_have_no_cancellation(self):
        for h in POLYS.values():
            for a in (1e-3, 0.7, 40.0):
                for rel in (1e-12, 1e-6, 0.3, 5.0):
                    b = a * (1 + rel)
                    mid = 0.5 * (a + b)
                    self.assertAlmostEqual(h.dd(a, b) / h.dh(mid), 1.0, delta=2 * rel + 1e-13)
                    q = h.q_divided(a, b)
                    exact = (h.q_function(b) - h.q_function(a)) / (b - a) if rel >= 0.3 else h.dh(mid) * h.osgood(mid)
                    self.assertAlmostEqual(q / exact, 1.0, delta=2 * rel + 1e-12)

    def test_validation_rejects_bad_h(self):
        with self.assertRaises(ValueError):  # nonzero minimum: a source term m g'(x)
            Custom(lambda u: u * u + 1, lambda u: 2 * u, tail_exponent=2).validate()
        with self.assertRaises(ValueError):  # Osgood fails for u log(1+u)
            Custom(lambda u: u * np.log1p(u), lambda u: np.log1p(u) + u / (1 + u), tail_exponent=1.0).validate()
        with self.assertRaises(ValueError):  # not convex: h'' < 0 somewhere
            Custom(lambda u: u * u * (2 + np.cos(u)), lambda u: 2 * u * (2 + np.cos(u)) - u * u * np.sin(u),
                   tail_exponent=2).validate()
        for bad in ({1.0: 1.0}, {2.0: -1.0}, {}):
            with self.assertRaises(ValueError):
                AbsPolynomial(bad)

    def test_custom_matches_the_polynomial(self):
        """Custom exercises generic Newton for H, the R table from callables and quadrature tails."""
        custom = Custom(lambda u: u ** 2 + u ** 4, lambda u: 2 * u + 4 * u ** 3, tail_exponent=4, name="u^2+u^4")
        poly = POLYS["u2+u4"]
        custom.validate()
        for u in np.geomspace(1e-9, 1e9, 37):  # inside and outside the table window [1e-6, 1e6]
            self.assertAlmostEqual(custom.osgood(u) / poly.osgood(u), 1.0, delta=5e-13, msg=u)
            self.assertAlmostEqual(custom.inverse(poly.h(u)) / u, 1.0, delta=1e-14, msg=u)
            self.assertAlmostEqual(custom.q_function(u) / poly.q_function(u), 1.0, delta=5e-13, msg=u)
        g = SIGN_CHANGING()
        one = HeteroFrontTracker.from_function(MultiplicativeFlux(g, 0.05, h=poly), _pulse, (-1.5, 1.5), cells=20)
        two = HeteroFrontTracker.from_function(MultiplicativeFlux(g, 0.05, h=custom), _pulse, (-1.5, 1.5), cells=20)
        one.run(1.5), two.run(1.5)
        self.assertEqual(one.counts, two.counts)
        np.testing.assert_allclose(two.positions(), one.positions(), rtol=0, atol=1e-10)
        self.assertAlmostEqual(two.integral(-2.4, 2.4), one.integral(-2.4, 2.4), delta=1e-10)

    def test_parse(self):
        self.assertEqual(parse("u^2+u^4"), POLYS["u2+u4"])
        self.assertEqual(parse("|u|^3 + u^2"), POLYS["u2+|u|3"])
        self.assertEqual(parse("2.5"), Power(2.5))
        self.assertEqual(parse("|u|^3"), Power(3))
        self.assertEqual(parse("0.5u^2 + 2u^4"), AbsPolynomial({2: 0.5, 4: 2.0}))


class GeneralLawTests(unittest.TestCase):
    def test_power_case_reproduces_power_law(self):
        for p in (1.5, 2.0, 3.0):
            P = PowerLaw(p)
            for G in (GeneralLaw(Power(p)), GeneralLaw(AbsPolynomial({p: 1.0}))):
                for thL, thR in FRONTS + [(1e4, 1e4 + 1), (-30.0, -30.001)]:
                    sL = math.copysign(abs(thL) ** (1 / p), thL)
                    sR = math.copysign(abs(thR) ** (1 / p), thR)
                    if p == 2.0:
                        K = sL + sR
                    elif thL * thR >= 0:
                        from hetfront.trajectory import divided_power
                        K = math.copysign(1.0, thL + thR) / divided_power(abs(thL), abs(thR), 1 / p)
                    else:
                        K = (abs(thR) - abs(thL)) / (sR - sL)
                    c = G.coef(thL, thR)
                    for alpha in (0.7, 0.0, -0.4):
                        x0, g0 = 0.3, 1.1
                        for tau in (0.01, 0.5, 2.0):
                            xp = P.position(x0, P.sigma_of(g0), alpha, K, tau)
                            if P.sigma_of(g0) + P.q * alpha * K * tau <= 0.0:
                                xp = x0 - g0 / alpha  # absorbed: PowerLaw's formula is invalid past the zero
                            xg = G.position(x0, g0, alpha, c, tau)
                            self.assertAlmostEqual(xg, xp, delta=1e-12 * (1 + abs(xp)), msg=(p, thL, thR, alpha, tau))

    def test_positions_match_rk4_of_the_physical_speed(self):
        for name, h in POLYS.items():
            G = GeneralLaw(h)
            for thL, thR in FRONTS + [(1.0, 1.0), (-2.0, -2.0)]:
                c = G.coef(thL, thR)
                for alpha, b in ((0.7, 1.1), (0.0, 1.3), (-0.2, 2.0)):
                    def speed(x):
                        gamma = alpha * x + b
                        if thL == thR:  # characteristic: γ h'(H(θ/γ)) sgn θ
                            return math.copysign(gamma * h.dh(h.inverse(abs(thL) / gamma)), thL)
                        return secant_speed(h, thL, thR, gamma)
                    x0, g0 = 0.3, alpha * 0.3 + b
                    zero = G._zero_location(x0, g0, alpha, c.direction)
                    T = 0.6 if zero is None else min(0.6, 0.5 * G.travel_time(x0, g0, zero, 0.0, c))
                    expect = rk4(speed, x0, T)
                    got = G.position(x0, alpha * x0 + b, alpha, c, T)
                    self.assertAlmostEqual(got, expect, delta=1e-11, msg=(name, thL, thR, alpha))
                    back = G.travel_time(x0, alpha * x0 + b, got, alpha * got + b, c)
                    self.assertAlmostEqual(back, T, delta=1e-13)

    def test_characteristic_time_is_a_difference_of_osgood_primitives(self):
        h = POLYS["u2+u4"]
        G = GeneralLaw(h)
        theta, alpha, x0, x1 = 1.7, 0.9, 0.0, 2.0
        g0, g1 = 0.5, 0.5 + alpha * (x1 - x0)
        u0, u1 = h.inverse(theta / g0), h.inverse(theta / g1)
        tau = G.travel_time(x0, g0, x1, g1, G.characteristic_coef(theta))
        self.assertAlmostEqual(alpha * tau, h.osgood(u1) - h.osgood(u0), places=13)

    def test_absorption_into_a_zero_in_finite_time(self):
        for h in POLYS.values():
            G = GeneralLaw(h)
            c = G.coef(0.0, -1.0)  # left-moving, as the manuscript's shock into x = 0 for g = x
            total = G.travel_time(1.0, 1.0, 0.0, 0.0, c)
            self.assertTrue(0 < total < math.inf)
            near = [G.travel_time(1.0, 1.0, e, e, c) for e in (1e-2, 1e-4, 1e-6, 1e-8)]
            gaps = [total - t for t in near]
            self.assertTrue(all(a > b > 0 for a, b in zip(gaps, gaps[1:])))
            mid = 0.37
            self.assertAlmostEqual(G.travel_time(1.0, 1.0, mid, mid, c) + G.travel_time(mid, mid, 0.0, 0.0, c),
                                   total, places=13)
            self.assertEqual(G.position(1.0, 1.0, 1.0, c, total * 1.5), 0.0)
            self.assertAlmostEqual(G.position(1.0, 1.0, 1.0, c, near[1]), 1e-4, delta=1e-15)

    def test_positions_near_a_zero_keep_relative_accuracy(self):
        """Regression: Newton's stopping rule must respect both the displacement and the distance to the zero.

        A front just emitted from a zero, or about to be absorbed, sits ~1e-10
        from it, where |u| ~ 1e4–1e6. The generic numerics must then match
        PowerLaw's closed form relative to that distance, not to |x|.
        """
        from hetfront.trajectory import divided_power
        for p in (1.5, 2.0, 3.0):
            P, G = PowerLaw(p), GeneralLaw(AbsPolynomial({p: 1.0}))
            alpha, z = 2.0, 0.0                       # γ = 2 (x - z): a zero at x = z = 0, so tiny distances are representable
            for thL, thR in ((0.0, 0.8), (-0.8, 0.0)):  # fronts leaving the zero to the right / arriving from the left
                sign = 1.0 if thL + thR > 0 else -1.0
                K = sign / divided_power(abs(thL), abs(thR), 1 / p)
                c = G.coef(thL, thR)
                if c.direction > 0:   # emitted at the zero: anchor at γ = 0
                    for tau in (1e-12, 1e-9, 1e-6):
                        xp = P.position(z, 0.0, alpha, K, tau)
                        xg = G.position(z, 0.0, alpha, c, tau)
                        self.assertAlmostEqual((xg - z) / (xp - z), 1.0, delta=1e-12, msg=(p, tau))
                else:                 # arriving: start at x0 > z, stop just short of absorption
                    x0 = z + 0.25
                    s0 = P.sigma_of(alpha * (x0 - z))
                    total = G.travel_time(x0, alpha * (x0 - z), z, 0.0, c)
                    for gap in (1e-3, 1e-7, 1e-10):
                        tau = total * (1 - gap)
                        # Exact distance to the zero, free of the cancellation in x0 + displacement:
                        # |g| = α (x - z) = σ^m with σ(τ) = σ0 + q α K τ.
                        d = max(s0 + P.q * alpha * K * tau, 0.0) ** P.m / alpha
                        xg = G.position(x0, alpha * (x0 - z), alpha, c, tau)
                        # Rounding of x itself limits the comparison to a few ulps of x0 relative to d.
                        self.assertAlmostEqual((xg - z) / d, 1.0, delta=1e-12 + 8 * math.ulp(x0) / d, msg=(p, gap))

    def test_fan_steps_keep_full_precision(self):
        P, G = PowerLaw(3.0), GeneralLaw(AbsPolynomial({3.0: 1.0}))
        k, delta = 10_000, 1e-3
        from hetfront.trajectory import divided_power
        K = 1 / divided_power(k * delta, (k + 1) * delta, 1 / 3)
        c = G.coef(k * delta, (k + 1) * delta)
        for alpha in (0.5, 0.0, -0.05):
            x = G.position(0.0, 1.0, alpha, c, 0.05)
            self.assertAlmostEqual(x, P.position(0.0, 1.0, alpha, K, 0.05), delta=1e-13 * (1 + abs(x)))

    def test_meeting_time_against_bisection(self):
        h = POLYS["u2+|u|3"]
        G = GeneralLaw(h)
        cases = [((2.0, 1.0), (1.0, 0.2)),      # both right-moving shocks
                 ((0.5, 0.0), (0.0, -1.0)),      # head-on
                 ((-0.2, -1.0), (-1.0, -3.0)),   # both left-moving
                 ((1.0, 1.0), (1.0, 0.0))]       # characteristic catches a shock
        for alpha, b in ((0.6, 1.0), (0.0, 1.5), (-0.3, 2.0)):
            for (la, ra), (lb, rb) in cases:
                ca, cb = G.coef(la, ra), G.coef(lb, rb)
                xa, xb = 0.0, 0.2
                sa, sb = b + alpha * xa, b + alpha * xb
                tau = G.meeting_time(xa, sa, ca, xb, sb, cb, alpha)
                gap = lambda t: G.position(xb, sb, alpha, cb, t) - G.position(xa, sa, alpha, ca, t)
                if math.isinf(tau):
                    self.assertGreater(gap(5.0), 0.0)
                    continue
                lo, hi = 0.0, tau * 2
                for _ in range(64):
                    mid = 0.5 * (lo + hi)
                    lo, hi = (mid, hi) if gap(mid) > 0 else (lo, mid)
                self.assertAlmostEqual(tau, 0.5 * (lo + hi), delta=1e-11, msg=((la, ra), (lb, rb), alpha))
        fan_a, fan_b = G.coef(0.0, 1.0), G.coef(1.0, 2.0)
        self.assertEqual(G.meeting_time(0.0, 1.0, fan_a, 0.2, 1.1, fan_b, 0.5), math.inf)

    def test_speed_order_is_independent_of_gamma(self):
        G = GeneralLaw(POLYS["u2+u4"])
        levels = [-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0]
        for i in range(len(levels) - 2):
            ca, cb = G.coef(levels[i], levels[i + 1]), G.coef(levels[i + 1], levels[i + 2])
            verdicts = {G.closing(ca, cb, g) for g in (1e-3, 0.1, 1.0, 10.0, 1e3)}
            self.assertEqual(len(verdicts), 1)
            # a δ-fan (increasing levels) never closes
            self.assertFalse(verdicts.pop())

    def test_no_common_straightening_for_polynomials(self):
        """Characteristic speed ratios vary with γ: no coordinate makes all fronts straight."""
        G = GeneralLaw(POLYS["u2+u4"])
        ratios = [G.speed(G.characteristic_coef(1.0), g) / G.speed(G.characteristic_coef(4.0), g)
                  for g in (0.5, 1.0, 4.0)]
        self.assertGreater(max(ratios) - min(ratios), 0.02)
        P = GeneralLaw(Power(3.0))
        ratios = [P.speed(P.characteristic_coef(1.0), g) / P.speed(P.characteristic_coef(4.0), g)
                  for g in (0.5, 1.0, 4.0)]
        self.assertAlmostEqual(max(ratios) - min(ratios), 0.0, places=14)

    def test_front_coef_direction(self):
        self.assertEqual(FrontCoef(2.0, 0.5).direction, 1)    # shock of positive levels moves right
        self.assertEqual(FrontCoef(-0.5, -2.0).direction, -1)
        self.assertEqual(FrontCoef(1.0, -1.0).direction, 0)   # stationary
        self.assertEqual(FrontCoef(-1.0, 3.0).direction, 1)
        self.assertEqual(FrontCoef(-1.0, -1.0).direction, -1)  # characteristic
        self.assertTrue(FrontCoef(0.4, 0.4).characteristic)



# ---------------------------------------------------------------------------
# The tracker's law interface (phase 2): GeneralLaw driven through the solver.
# ---------------------------------------------------------------------------
from hetfront import HeteroFrontTracker, MultiplicativeFlux, PeriodicHeteroFrontTracker, PiecewiseAffine  # noqa: E402


class _GeneralLawFlux(MultiplicativeFlux):
    """Test shim: a MultiplicativeFlux whose fronts use GeneralLaw (anchor γ = |g_Δ|).

    Only the tracker-facing pieces are replaced; level maps keep ``power``, so
    the shim is physical only for h = |u|^power or when data are given as levels.
    """

    def __init__(self, g, delta, nonlinearity, power=2.0):
        super().__init__(g, delta, power)
        self.law = GeneralLaw(nonlinearity)
        self.sigma_left, self.sigma_right = g.absg_left, g.absg_right

    def kappa(self, kL, kR, cell=None):
        return self.law.coef(kL * self.delta, kR * self.delta)

    def sigma(self, c, x):
        return self.g.abs_g(c, x)

    def sigma_array(self, x, cells):
        return self.g.abs_g_array(x, cells)


def _theta_data(n=14, seed=3):
    rng = np.random.default_rng(seed)
    positions = np.sort(rng.uniform(-1.4, 1.4, n))
    levels = rng.integers(-6, 7, n + 1)
    levels[0] = levels[-1] = 0
    return positions, levels


class TrackerInterfaceTests(unittest.TestCase):
    COEFFS = {
        "sign-changing": lambda: PiecewiseAffine.from_function(lambda x: x * (1 - x ** 2), np.linspace(-2, 2, 17)),
        "well-signed": lambda: PiecewiseAffine.from_function(lambda x: 2 + np.sin(3 * x), np.linspace(-2, 2, 17)),
    }

    def test_general_law_power_case_reproduces_the_tracker(self):
        positions, levels = _theta_data()
        for p in (1.5, 2.0, 3.0):
            for name, make in self.COEFFS.items():
                g = make()
                ref = HeteroFrontTracker(MultiplicativeFlux(g, 0.05, p), positions, levels).run(1.5)
                gen = HeteroFrontTracker(_GeneralLawFlux(g, 0.05, Power(p), p), positions, levels).run(1.5)
                self.assertEqual(ref.counts, gen.counts, (p, name))
                self.assertEqual([(i.kL, i.kR) for i in ref.items], [(i.kL, i.kR) for i in gen.items])
                np.testing.assert_allclose(gen.positions(), ref.positions(), rtol=0, atol=1e-10)

    def test_polynomial_tracker_invariants(self):
        positions, levels = _theta_data(seed=7)
        for h in POLYS.values():
            for name, make in self.COEFFS.items():
                tr = HeteroFrontTracker(_GeneralLawFlux(make(), 0.05, h), positions, levels)
                tv, count = tr.total_variation(), tr.front_count()
                for t in (0.05, 0.3, 1.0, 2.5):
                    tr.run(t)
                    tr.check_levels()
                    self.assertTrue(np.all(np.diff(tr.positions()) >= -1e-11))
                    self.assertLessEqual(tr.total_variation(), tv + 1e-12)
                    self.assertLessEqual(tr.front_count(), count)
                    self.assertTrue(tr.interface_admissible())
                    self.assertEqual(tr.counts["emission"], 0)
                    self.assertEqual(tr.counts["multi_front_collision"], 0)
                self.assertGreater(tr.counts["collision"], 0, name)

    def test_circle_runs_with_the_general_law(self):
        g = PiecewiseAffine.from_function(lambda x: 2 + np.sin(2 * np.pi * x), np.arange(12) / 12, period=1.0)
        positions = np.array([0.1, 0.35, 0.6, 0.8])
        levels = np.array([3, -2, 5, 0])
        for nl, p in ((Power(3.0), 3.0), (POLYS["u2+u4"], 2.0)):
            tr = PeriodicHeteroFrontTracker(_GeneralLawFlux(g, 0.1, nl, p), positions, levels)
            tv = tr.total_variation()
            tr.run(2.0)
            tr.check_levels()
            self.assertLessEqual(tr.total_variation(), tv + 1e-12)
            self.assertGreater(tr.counts["corner"], 0)
        ref = PeriodicHeteroFrontTracker(MultiplicativeFlux(g, 0.1, 3.0), positions, levels).run(2.0)
        gen = PeriodicHeteroFrontTracker(_GeneralLawFlux(g, 0.1, Power(3.0), 3.0), positions, levels).run(2.0)
        self.assertEqual(ref.counts, gen.counts)
        np.testing.assert_allclose(gen.positions(), ref.positions(), rtol=0, atol=1e-10)



# ---------------------------------------------------------------------------
# The public API (phase 3): MultiplicativeFlux(g, δ, h=...).
# ---------------------------------------------------------------------------
from hetfront.homogenisation import PeriodicStraightening, effective_flux_nodes  # noqa: E402

SIGN_CHANGING = lambda: PiecewiseAffine.from_function(lambda x: x * (1 - x ** 2), np.linspace(-2.5, 2.5, 21))


def _pulse(x):
    return np.where(np.abs(x) < 1.3, 1.2 * np.cos(np.pi * x / 2.6) - 0.3 * np.sin(3 * x), 0.0)


class PublicFluxTests(unittest.TestCase):
    def test_construction(self):
        g = SIGN_CHANGING()
        self.assertFalse(MultiplicativeFlux(g, 0.1, h="|u|^3").general)
        self.assertEqual(MultiplicativeFlux(g, 0.1, h="|u|^3").power, 3.0)
        flux = MultiplicativeFlux(g, 0.1, h="u^2+u^4")
        self.assertTrue(flux.general)
        self.assertIsNone(flux.power)
        self.assertEqual(flux.nonlinearity, POLYS["u2+u4"])
        with self.assertRaises(TypeError):
            flux.s(3)
        with self.assertRaises(ValueError):
            MultiplicativeFlux(g, 0.1, power=3.0, h="u^2+u^4")
        with self.assertRaises(ValueError):
            MultiplicativeFlux(g, 0.1, h="u^2+1")  # not parseable as a convex h with h(0) = 0

    def test_level_maps_invert(self):
        flux = MultiplicativeFlux(SIGN_CHANGING(), 0.01, h="u^2+|u|^3")
        x = np.array([-2.2, -0.6, 0.4, 1.7])
        k = np.array([-40, 17, -3, 250])
        u = flux.u(k, x)
        np.testing.assert_allclose(flux.theta_of(x, u), k * flux.delta, rtol=1e-14)
        np.testing.assert_array_equal(flux.level(flux.theta_of(x, u)), k)
        # f_u = g h'(u)
        np.testing.assert_allclose(flux.characteristic_speed(k, x),
                                   flux.g(x) * np.sign(u) * POLYS["u2+|u|3"].dh(np.abs(u)), rtol=1e-13)

    def test_generic_numerics_reproduce_the_power_tracker(self):
        """AbsPolynomial({p: 1}) runs GeneralLaw's numerical path; PowerLaw is exact."""
        g = SIGN_CHANGING()
        for p in (1.5, 3.0):
            ref = HeteroFrontTracker.from_function(MultiplicativeFlux(g, 0.05, p), _pulse, (-1.5, 1.5), cells=24)
            gen = HeteroFrontTracker.from_function(MultiplicativeFlux(g, 0.05, h=AbsPolynomial({p: 1.0})),
                                                   _pulse, (-1.5, 1.5), cells=24)
            for t in (0.0, 0.4, 1.5):
                ref.run(t), gen.run(t)
                self.assertEqual(ref.counts, gen.counts)
                np.testing.assert_allclose(gen.positions(), ref.positions(), rtol=0, atol=1e-10)
                self.assertAlmostEqual(gen.integral(-2.4, 2.4), ref.integral(-2.4, 2.4), delta=1e-12)
                self.assertAlmostEqual(gen.mass_absorbed(), ref.mass_absorbed(), delta=1e-12)
                xs = np.linspace(-2.4, 2.4, 97)
                np.testing.assert_allclose(gen.primitive(xs), ref.primitive(xs), rtol=0, atol=1e-12)

    def test_mass_budget_through_absorption(self):
        for h in POLYS.values():
            flux = MultiplicativeFlux(SIGN_CHANGING(), 0.05, h=h)
            tr = HeteroFrontTracker.from_function(flux, _pulse, (-1.5, 1.5), cells=30)
            a, b = -2.3, 2.3
            mass0 = tr.integral(a, b)
            times = [0.2, 0.7, 1.5, 3.0]
            for t in times:
                tr.run(t)
            inflow = tr.flux_through(a, [0.0] + times) - tr.flux_through(b, [0.0] + times)
            self.assertGreater(tr.counts["absorption"], 0)
            self.assertGreater(abs(tr.mass_absorbed()), 1e-3)
            residual = tr.integral(a, b) + tr.mass_absorbed() - inflow[-1] - mass0
            self.assertLess(abs(residual), 1e-11, str(h))
            self.assertEqual(tr.counts["emission"], 0)
            self.assertTrue(tr.interface_admissible())

    def test_integral_matches_quadrature_of_the_profile(self):
        flux = MultiplicativeFlux(SIGN_CHANGING(), 0.05, h="u^2+u^4")
        tr = HeteroFrontTracker.from_function(flux, _pulse, (-1.5, 1.5), cells=30).run(0.5)
        a, b = 0.15, 0.85  # inside one sign interval, away from zeros
        edges = np.unique(np.r_[a, b, tr.positions(), flux.g.nodes])
        edges = edges[(edges >= a) & (edges <= b)]
        xs, ws = np.polynomial.legendre.leggauss(30)
        total = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            pts = 0.5 * (lo + hi) + 0.5 * (hi - lo) * xs
            total += 0.5 * (hi - lo) * np.dot(ws, tr.sample_u(pts))
        self.assertAlmostEqual(tr.integral(a, b), total, delta=1e-12)

    def test_l1_contraction(self):
        g = SIGN_CHANGING()
        for h in POLYS.values():
            flux = MultiplicativeFlux(g, 0.05, h=h)
            one = HeteroFrontTracker.from_function(flux, _pulse, (-1.5, 1.5), cells=30)
            two = HeteroFrontTracker.from_function(flux, lambda x: 0.8 * _pulse(x - 0.2), (-1.5, 1.5), cells=30)
            distances = []
            for t in (0.0, 0.3, 0.9, 2.0):
                one.run(t), two.run(t)
                distances.append(one.u_l1_distance(two, -2.4, 2.4))
            self.assertTrue(all(b <= a + 1e-11 for a, b in zip(distances, distances[1:])), distances)

    def test_characteristic_accepts_numpy_levels(self):
        """Regression: notebooks pass θ as numpy floats (np.linspace); direction must not subtract numpy bools."""
        g = PiecewiseAffine.from_function(lambda x: 2 + np.sin(3 * x), np.linspace(-2, 2, 9))
        for flux in (MultiplicativeFlux(g, 0.1), MultiplicativeFlux(g, 0.1, 3.0),
                     MultiplicativeFlux(g, 0.1, h="u^2+u^4"), MultiplicativeFlux(g, 0.1, h="u^2; u^2+u^4")):
            for theta in np.linspace(-0.9, 0.9, 4).tolist() + [np.float64(0.5), np.float64(-0.5), np.float64(0.0)]:
                xs, ts = flux.characteristic(0.0, theta if isinstance(theta, np.float64) else np.float64(theta), 0.5)
                self.assertEqual(ts[-1], 0.5)
                self.assertEqual(np.sign(xs[-1] - xs[0]), np.sign(theta))

    def test_characteristic_reaches_a_zero_at_the_osgood_time(self):
        h = POLYS["u2+u4"]
        flux = MultiplicativeFlux(PiecewiseAffine([-1, 1], [-1, 1], extend="affine"), 0.1, h=h)
        theta = -0.6  # g = x, x0 = 1: left-moving into the zero at 0
        xs, ts = flux.characteristic(1.0, theta, 50.0, samples=8)
        self.assertEqual(xs[-1], 0.0)
        self.assertAlmostEqual(ts[-1], h.osgood(h.inverse(abs(theta))), places=13)

    def test_effective_flux_nodes(self):
        g = PiecewiseAffine.from_function(lambda x: 2 + np.sin(2 * np.pi * x), np.arange(16) / 16, period=1.0)
        levels = np.arange(-6, 7)
        for p in (1.5, 3.0):
            v, F = effective_flux_nodes(g, 0.1, levels, Power(p))
            c = PeriodicStraightening(g, p).coefficient
            np.testing.assert_allclose(F, c * np.abs(v) ** p, rtol=1e-12)
        v, F = effective_flux_nodes(g, 0.1, levels, "u^2+u^4")
        self.assertTrue(np.all(np.diff(v) > 0))
        slopes = np.diff(F) / np.diff(v)  # the homogenised flux F̄(v̄) is even and convex
        self.assertTrue(np.all(np.diff(slopes) > 0))
        np.testing.assert_allclose(F, F[::-1])

    def test_straightening_rejects_general_h(self):
        g = PiecewiseAffine.from_function(lambda x: 2 + np.sin(2 * np.pi * x), np.arange(16) / 16, period=1.0)
        straight = PeriodicStraightening(g, 2.0)
        flux = MultiplicativeFlux(g, 0.1, h="u^2+u^4")
        het = PeriodicHeteroFrontTracker(flux, [0.2, 0.6], [1, -1])
        with self.assertRaises(ValueError):
            straight.corrector_l1(het, het)



# ---------------------------------------------------------------------------
# Examples, studies and command-line entry points (phase 4).
# ---------------------------------------------------------------------------
import csv  # noqa: E402
import json  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402

from hetfront.examples import EXAMPLES  # noqa: E402
from hetfront.studies import PROBLEMS, standard_study  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class EntryPointTests(unittest.TestCase):
    def test_examples_choose_their_nonlinearity(self):
        ex = EXAMPLES["poly-zeros"]
        self.assertTrue(ex.build(delta=0.1, cells=10).flux.general)
        self.assertEqual(ex.build(delta=0.1, cells=10, power=3.0).flux.power, 3.0)
        self.assertEqual(EXAMPLES["two-zeros"].build(delta=0.1, cells=10, h="u^2+|u|^3").flux.nonlinearity,
                         POLYS["u2+|u|3"])
        with self.assertRaises(ValueError):
            ex.build(delta=0.1, cells=10, power=3.0, h="u^2+u^4")
        tr = ex.build(delta=0.1, cells=10).run(ex.final_time)
        self.assertGreater(tr.counts["absorption"], 0)

    def test_study_records_the_nonlinearity(self):
        problem = replace(PROBLEMS["line-sign-changing"], h="u^2+u^4")
        study = standard_study(problem, levels=2, depth=1, coarse_delta=0.2, coarse_nodes=8, samples=41)
        self.assertEqual(study["reference"]["flux"], "|u|^2 + |u|^4")
        self.assertIsNone(study["reference"]["power"])
        for kind in ("delta", "mesh", "joint"):
            self.assertTrue(all(row["flux"] == "|u|^2 + |u|^4" and np.isfinite(row["theta_l1"]) for row in study[kind]))
        mesh = [row["h"] for row in study["mesh"]]  # "h" stays the mesh width
        self.assertGreater(mesh[0], mesh[1])

    def test_convergence_cli_with_flux(self):
        with tempfile.TemporaryDirectory() as output:
            subprocess.run([sys.executable, str(ROOT / "convergence.py"), "--problem", "line-sign-changing",
                            "--flux", "u^2+u^4", "--flux", "|u|^3", "--levels", "2", "--depth", "1",
                            "--samples", "41", "--no-plot", "--output", output],
                           cwd=ROOT, check=True, capture_output=True, text=True)
            names = sorted(p.name for p in Path(output).glob("*.json"))
            self.assertEqual(names, ["convergence_line-sign-changing_h-u2+u4.json",
                                     "convergence_line-sign-changing_p3p0.json"])
            general = json.loads((Path(output) / names[0]).read_text())
            self.assertIsNone(general["power"])
            self.assertEqual(general["flux"], "|u|^2 + |u|^4")

    def test_main_cli_writes_a_general_run(self):
        with tempfile.TemporaryDirectory() as output:
            done = subprocess.run([sys.executable, str(ROOT / "main.py"), "--example", "poly-zeros", "--delta", "0.1",
                                   "--cells", "12", "--frames", "4", "--samples", "101", "--no-plot",
                                   "--output", output], cwd=ROOT, check=True, capture_output=True, text=True)
            self.assertIn("h(u) = |u|^2 + |u|^4", done.stdout)
            residual = float(done.stdout.rsplit(":", 1)[1])
            self.assertLess(abs(residual), 1e-12)
            with open(Path(output) / "poly-zeros_pieces.csv") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(rows and all(r["kappa"] == "nan" for r in rows))
            data = np.load(Path(output) / "poly-zeros.npz")
            self.assertEqual(str(data["h"]), "|u|^2 + |u|^4")
            bad = subprocess.run([sys.executable, str(ROOT / "main.py"), "--example", "poly-zeros", "--flux",
                                  "u log u", "--no-plot", "--output", output], cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("--flux", bad.stderr)



# ---------------------------------------------------------------------------
# Asymmetric h (phase 5): h₊ for u ≥ 0, h₋ for u < 0.
# ---------------------------------------------------------------------------
from hetfront.nonlinearity import Asymmetric  # noqa: E402

ASYM = {"u2|u2+u4": "u^2; u^2+u^4", "u2+u4|u2": "u^2+u^4; u^2", "u3|u1.5": "|u|^3; |u|^1.5"}


def _asym_rk4_speed(h, thL, thR, eta, gamma):
    """Physical [g h(u)]/[u] for an asymmetric h, g = η γ, states from the branch of sgn u = η sgn θ."""
    def state(theta):
        if theta == 0:
            return 0.0
        b = h.branch(eta * theta)
        return eta * math.copysign(b.inverse(abs(theta) / gamma), theta)
    uL, uR = state(thL), state(thR)
    return eta * gamma * (h.h(uR) - h.h(uL)) / (uR - uL)


class AsymmetricTests(unittest.TestCase):
    def test_parse_and_construction(self):
        h = parse("u^2; u^2+u^4")
        self.assertIsInstance(h, Asymmetric)
        self.assertEqual(h.plus, Power(2))
        self.assertEqual(h.minus, POLYS["u2+u4"])
        self.assertEqual(parse(str(h)), h)
        self.assertEqual(parse("u<0: u^2+u^4; u>0: u^2"), h)
        self.assertEqual(parse("u^2+u^4; u^2+u^4"), POLYS["u2+u4"])  # equal branches: even h
        for bad in ("u^2; u^3; u^4", "u>0: u^2; u>0: u^3", "u^2;"):
            with self.assertRaises(ValueError):
                parse(bad)
        with self.assertRaises(ValueError):
            Asymmetric(h, Power(2))  # branches must be even
        with self.assertRaises(TypeError):
            h.inverse(1.0)  # no single inverse: use a branch
        np.testing.assert_allclose(h(np.array([-2.0, -0.5, 0.0, 0.5, 2.0])), [20.0, 0.3125, 0.0, 0.25, 4.0])
        np.testing.assert_allclose(h.dh(np.array([-1.0, 1.0])), [-6.0, 2.0])
        with self.assertRaises(ValueError):
            Asymmetric(Power(2), "u^2+1").validate()

    def test_kappa_needs_the_cell(self):
        flux = MultiplicativeFlux(SIGN_CHANGING(), 0.1, h="u^2; u^2+u^4")
        with self.assertRaises(ValueError):
            flux.kappa(1, 2)
        c_pos, c_neg = flux.g.cell_of(-2.0), flux.g.cell_of(-0.5)  # g > 0 left of -1, g < 0 on (-1, 0)
        self.assertEqual(flux.kappa(1, 2, c_pos).eta, 1.0)
        self.assertEqual(flux.kappa(1, 2, c_neg).eta, -1.0)

    def test_level_maps_choose_the_branch(self):
        flux = MultiplicativeFlux(SIGN_CHANGING(), 0.01, h="u^2; u^2+u^4")
        h = flux.nonlinearity
        x = np.array([-2.2, -2.2, -0.6, -0.6, 0.4, 0.4, 1.7, 1.7])  # each sign of g, each sign of θ
        k = np.array([40, -40, 40, -40, 17, -17, 250, -250])
        u = flux.u(k, x)
        gx = flux.g(x)
        np.testing.assert_array_equal(np.sign(u), np.sign(gx) * np.sign(k))
        for ui, ki, gi in zip(u, k, gx):
            branch = h.plus if ui > 0 else h.minus
            self.assertAlmostEqual(branch.h(abs(ui)) * abs(gi), abs(ki) * flux.delta, places=13)
        np.testing.assert_allclose(flux.theta_of(x, u), k * flux.delta, rtol=1e-14)
        np.testing.assert_allclose(flux.characteristic_speed(k, x), gx * h.dh(u), rtol=1e-13)

    def test_positions_match_rk4_on_both_signs_of_g(self):
        for spec in ASYM.values():
            G = GeneralLaw(spec)
            for eta in (1.0, -1.0):
                for thL, thR in FRONTS + [(1.0, 1.0), (-2.0, -2.0)]:
                    c = G.coef(thL, thR, eta)
                    for alpha, b in ((0.7, 1.1), (0.0, 1.3), (-0.2, 2.0)):
                        def speed(x):
                            gamma = alpha * x + b
                            if thL == thR:
                                br = G.h.branch(eta * thL)
                                return math.copysign(gamma * br.dh(br.inverse(abs(thL) / gamma)), thL)
                            return _asym_rk4_speed(G.h, thL, thR, eta, gamma)
                        x0, g0 = 0.3, alpha * 0.3 + b
                        zero = G._zero_location(x0, g0, alpha, c.direction)
                        T = 0.6 if zero is None else min(0.6, 0.5 * G.travel_time(x0, g0, zero, 0.0, c))
                        got = G.position(x0, g0, alpha, c, T)
                        self.assertAlmostEqual(got, rk4(speed, x0, T), delta=1e-11, msg=(spec, eta, thL, thR, alpha))

    def test_equal_branches_reproduce_the_even_tracker(self):
        even = POLYS["u2+u4"]
        glued = Asymmetric(AbsPolynomial({2: 1.0, 4: 1.0}), AbsPolynomial({2: 1.0, 4: 1.0}))  # bypasses parse
        g = SIGN_CHANGING()
        one = HeteroFrontTracker.from_function(MultiplicativeFlux(g, 0.05, h=even), _pulse, (-1.5, 1.5), cells=24)
        two = HeteroFrontTracker.from_function(MultiplicativeFlux(g, 0.05, h=glued), _pulse, (-1.5, 1.5), cells=24)
        one.run(1.5), two.run(1.5)
        self.assertEqual(one.counts, two.counts)
        np.testing.assert_allclose(two.positions(), one.positions(), rtol=0, atol=1e-13)
        self.assertAlmostEqual(two.integral(-2.4, 2.4), one.integral(-2.4, 2.4), delta=1e-13)

    def test_reflection_symmetry(self):
        """v = -u solves the problem with -g and swapped branches; the θ levels are identical."""
        nodes = np.linspace(-2.5, 2.5, 21)
        g = PiecewiseAffine.from_function(lambda x: x * (1 - x ** 2), nodes)
        minus_g = PiecewiseAffine.from_function(lambda x: -x * (1 - x ** 2), nodes)
        positions, levels = _theta_data(seed=11)
        for plus, minus in (("u^2", "u^2+u^4"), ("|u|^3", "|u|^1.5")):
            a = HeteroFrontTracker(MultiplicativeFlux(g, 0.05, h=f"{plus}; {minus}"), positions, levels)
            b = HeteroFrontTracker(MultiplicativeFlux(minus_g, 0.05, h=f"{minus}; {plus}"), positions, levels)
            xs = np.linspace(-2.3, 2.3, 47) + 1e-3
            for t in (0.3, 1.0, 2.5):
                a.run(t), b.run(t)
                self.assertEqual(a.counts, b.counts)
                np.testing.assert_allclose(b.positions(), a.positions(), rtol=0, atol=1e-11)
                np.testing.assert_array_equal(b.sample_levels(xs), a.sample_levels(xs))
                np.testing.assert_allclose(b.sample_u(xs), -a.sample_u(xs), rtol=1e-12, atol=0)
                self.assertAlmostEqual(b.integral(-2.4, 2.4), -a.integral(-2.4, 2.4), delta=1e-12)
                self.assertAlmostEqual(b.mass_absorbed(), -a.mass_absorbed(), delta=1e-12)
            self.assertGreater(a.counts["absorption"], 0)

    def test_mass_budget_and_single_branch_traces(self):
        g = SIGN_CHANGING()
        for spec in ASYM.values():
            flux = MultiplicativeFlux(g, 0.05, h=spec)
            tr = HeteroFrontTracker.from_function(flux, _pulse, (-1.5, 1.5), cells=30)
            a, b = -2.3, 2.3
            mass0 = tr.integral(a, b)
            times = [0.2, 0.7, 1.5, 3.0]
            for t in times:
                tr.run(t)
                for iface in tr.interfaces:
                    # g' < 0 at the zero (g > 0 on its left): traces on h₊ (u ≥ 0); g' > 0: on h₋ (u ≤ 0).
                    side = 1.0 if iface.left_sign > 0 else -1.0
                    z = iface.x
                    traces = flux.u(np.array([iface.kL, iface.kR]), np.array([z - 1e-6, z + 1e-6]))
                    self.assertTrue(np.all(side * traces >= 0.0), (spec, z, traces))
            inflow = tr.flux_through(a, [0.0] + times) - tr.flux_through(b, [0.0] + times)
            residual = tr.integral(a, b) + tr.mass_absorbed() - inflow[-1] - mass0
            self.assertLess(abs(residual), 1e-11, spec)
            self.assertGreater(tr.counts["absorption"], 0)
            self.assertEqual(tr.counts["emission"], 0)
            self.assertEqual(tr.counts["multi_front_collision"], 0)
            self.assertTrue(tr.interface_admissible())

    def test_l1_contraction_and_invariants(self):
        g = SIGN_CHANGING()
        for spec in ASYM.values():
            flux = MultiplicativeFlux(g, 0.05, h=spec)
            one = HeteroFrontTracker.from_function(flux, _pulse, (-1.5, 1.5), cells=30)
            two = HeteroFrontTracker.from_function(flux, lambda x: 0.8 * _pulse(x - 0.2), (-1.5, 1.5), cells=30)
            tv, count = one.total_variation(), one.front_count()
            distances = []
            for t in (0.0, 0.3, 0.9, 2.0):
                one.run(t), two.run(t)
                one.check_levels()
                self.assertTrue(np.all(np.diff(one.positions()) >= -1e-11))
                self.assertLessEqual(one.total_variation(), tv + 1e-12)
                self.assertLessEqual(one.front_count(), count)
                distances.append(one.u_l1_distance(two, -2.4, 2.4))
            self.assertTrue(all(b <= a + 1e-11 for a, b in zip(distances, distances[1:])), (spec, distances))

    def test_circle_and_effective_flux(self):
        g = PiecewiseAffine.from_function(lambda x: 2 + np.sin(2 * np.pi * x), np.arange(16) / 16, period=1.0)
        tr = PeriodicHeteroFrontTracker(MultiplicativeFlux(g, 0.1, h="u^2; u^2+u^4"), [0.1, 0.35, 0.6, 0.8], [3, -2, 5, 0])
        mass, tv = tr.integral(), tr.total_variation()
        tr.run(2.0)
        tr.check_levels()
        self.assertAlmostEqual(tr.integral(), mass, delta=1e-12)  # well-signed circle: exact conservation
        self.assertLessEqual(tr.total_variation(), tv + 1e-12)
        levels = np.arange(-6, 7)
        v, F = effective_flux_nodes(g, 0.1, levels, "u^2; u^2+u^4")
        v_plus, _ = effective_flux_nodes(g, 0.1, levels, "u^2")
        v_minus, _ = effective_flux_nodes(g, 0.1, levels, "u^2+u^4")
        np.testing.assert_allclose(v[levels > 0], v_plus[levels > 0], rtol=1e-14)  # η = +1: θ > 0 on h₊
        np.testing.assert_allclose(v[levels < 0], v_minus[levels < 0], rtol=1e-14)
        slopes = np.diff(F) / np.diff(v)
        self.assertTrue(np.all(np.diff(slopes) > 0))  # still convex, no longer even

    def test_cli_with_asymmetric_flux(self):
        with tempfile.TemporaryDirectory() as output:
            done = subprocess.run([sys.executable, str(ROOT / "main.py"), "--example", "two-zeros", "--flux",
                                   "u^2; u^2+u^4", "--delta", "0.1", "--cells", "12", "--frames", "3",
                                   "--samples", "101", "--no-plot", "--output", output],
                                  cwd=ROOT, check=True, capture_output=True, text=True)
            self.assertIn("h(u) = u>0: |u|^2; u<0: |u|^2 + |u|^4", done.stdout)
            self.assertLess(abs(float(done.stdout.rsplit(":", 1)[1])), 1e-12)
            subprocess.run([sys.executable, str(ROOT / "convergence.py"), "--problem", "line-sign-changing",
                            "--flux", "u^2; u^2+u^4", "--levels", "2", "--depth", "1", "--samples", "41",
                            "--no-plot", "--output", output], cwd=ROOT, check=True, capture_output=True, text=True)
            record = json.loads((Path(output) / "convergence_line-sign-changing_h-u2_u2+u4.json").read_text())
            self.assertEqual(parse(record["flux"]), parse("u^2; u^2+u^4"))


if __name__ == "__main__":
    unittest.main()
