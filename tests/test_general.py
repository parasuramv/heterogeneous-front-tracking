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


if __name__ == "__main__":
    unittest.main()
