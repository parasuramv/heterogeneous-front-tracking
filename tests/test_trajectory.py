"""Closed-form front laws against direct numerical integration of the RH ODE."""
import math
import unittest

import numpy as np

from hetfront import HeteroFrontTracker, MultiplicativeFlux, PiecewiseAffine, trajectory


def rk4_front(flux, kL, kR, x0, T, steps=20000):
    """Integrate ẋ = g_Δ(x) (u_L(x) + u_R(x)) through every corner with RK4."""
    def speed(x):
        return flux.g(x) * (flux.u(kL, x) + flux.u(kR, x))
    x, h = x0, T / steps
    for _ in range(steps):
        k1 = speed(x)
        k2 = speed(x + 0.5 * h * k1)
        k3 = speed(x + 0.5 * h * k2)
        k4 = speed(x + h * k3)
        x += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x


class FormulaTests(unittest.TestCase):
    def test_sigma_is_linear_and_consistent_with_position(self):
        x0, s0, alpha, kappa = 0.3, 1.2, -0.7, 0.9
        for tau in (0.0, 0.4, 1.1):
            x = trajectory.position(x0, s0, alpha, kappa, tau)
            s = trajectory.sigma(s0, alpha, kappa, tau)
            # |g| affine with slope alpha: s² - s0² = alpha (x - x0).
            self.assertAlmostEqual(s * s - s0 * s0, alpha * (x - x0), places=13)

    def test_travel_and_meeting_times_need_no_division_by_alpha(self):
        # Flat cell: straight lines.
        self.assertAlmostEqual(trajectory.travel_time(0.0, 2.0, 1.0, 2.0, 0.5), 1.0)
        self.assertAlmostEqual(trajectory.meeting_time(0.0, 2.0, 1.0, 1.0, 2.0, 0.5), 1.0)
        self.assertEqual(trajectory.meeting_time(0.0, 1.0, 0.5, 1.0, 1.0, 1.0), math.inf)

    def test_manuscript_shock_curve(self):
        # g = x, θ_L = 0, θ_R = -1: γ(t) = (t - 2)²/4, reaching the zero at t = 2.
        for t in (0.0, 0.5, 1.3, 2.0):
            self.assertAlmostEqual(trajectory.position(1.0, 1.0, 1.0, -1.0, t), (t - 2) ** 2 / 4, places=14)
        self.assertAlmostEqual(trajectory.travel_time(1.0, 1.0, 0.0, 0.0, -1.0), 2.0)

    def test_characteristic_of_xu2(self):
        # Characteristic from q0 with u0 = -1 under g = x: q(t) = q0 (1 - t)².
        flux = MultiplicativeFlux(PiecewiseAffine([-1, 1], [-1, 1], extend="affine"), 1.0)
        for q0 in (0.2, 0.7):
            s = flux.s(-q0)  # θ = sgn(u) g u² = -q0, as a level with δ = 1
            for t in (0.3, 0.8):
                x = trajectory.position(q0, math.sqrt(q0), 1.0, 2 * s, t)
                self.assertAlmostEqual(x, q0 * (1 - t) ** 2, places=14)


class CornerCrossingTests(unittest.TestCase):
    def check_against_rk4(self, g, kL, kR, x0, T, delta=0.1):
        flux = MultiplicativeFlux(g, delta)
        tracker = HeteroFrontTracker(flux, [x0], [kL, kR])
        self.assertEqual(tracker.front_count(), 1)
        tracker.run(T)
        x_closed = tracker.fronts[0].x(T)
        x_rk4 = rk4_front(flux, kL, kR, x0, T)
        self.assertGreater(tracker.counts["corner"], 2)
        self.assertAlmostEqual(x_closed, x_rk4, places=8)
        return tracker

    def test_shock_through_many_corners(self):
        g = PiecewiseAffine.from_function(lambda x: 2 + np.sin(3 * x), np.linspace(-4, 4, 33))
        self.check_against_rk4(g, 30, 5, -1.0, 2.0)

    def test_leftward_front_in_negative_g(self):
        g = PiecewiseAffine.from_function(lambda x: -1.5 - 0.5 * np.cos(2 * x), np.linspace(-4, 4, 41))
        self.check_against_rk4(g, -6, -20, 1.5, 1.5)

    def test_front_through_flat_cells(self):
        g = PiecewiseAffine([-2, -1, 0, 1, 2], [1, 1, 2, 2, 3])
        self.check_against_rk4(g, 25, 0, -2.5, 3.0)


if __name__ == "__main__":
    unittest.main()
