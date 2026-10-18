"""Smoke tests for plots and examples (Agg backend, coarse resolutions)."""
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hetfront.examples import EXAMPLES  # noqa: E402
from hetfront.visualization import (observe_tracker, overview, plot_circle,  # noqa: E402
                                    plot_profile, plot_spacetime, thin_pieces)


class ExampleTests(unittest.TestCase):
    def test_every_example_runs_and_closes_its_budget(self):
        for name, example in EXAMPLES.items():
            with self.subTest(name=name):
                tracker = example.build(delta=0.05, nodes=min(example.defaults["nodes"], 64), cells=30)
                x = np.linspace(*example.window, 201, endpoint=not example.periodic)
                times = np.linspace(0, min(example.final_time, 1.0), 6)
                history = observe_tracker(tracker, x, times)
                tracker.check_levels()
                self.assertTrue(tracker.interface_admissible())
                # Line windows include the exact edge inflow; the circle has none.
                d = history.diagnostics
                budget = d["mass"] + d["absorbed"] - d["inflow"]
                self.assertLess(np.max(np.abs(budget - budget[0])), 1e-10)


class PlotTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_overview_and_profiles(self):
        example = EXAMPLES["xu2-blowup"]
        tracker = example.build(delta=0.05, cells=40)
        history = observe_tracker(tracker, np.linspace(-1.6, 1.6, 201), np.linspace(0, 1.2, 7))
        self.assertFalse(np.isinf(history.u).any())
        overview(history, exact_g=example.g, characteristics=example.characteristics(tracker.flux, 1.2))
        plot_profile(history, 3, "u")
        plot_spacetime(history, "u", max_fronts=None)

    def test_periodic_step_profile_includes_closing_arc(self):
        tracker = EXAMPLES["circle-sine"].build(delta=0.1, cells=10)
        history = observe_tracker(tracker, np.linspace(0, 2*np.pi, 100, endpoint=False), [0.0])
        self.assertEqual(history.steps[0][0][-1], 2*np.pi)

    def test_budget_excludes_interfaces_outside_window(self):
        from hetfront import HeteroFrontTracker, MultiplicativeFlux, PiecewiseAffine
        g = PiecewiseAffine([-1,1], [-1,1], extend="affine")
        tracker = HeteroFrontTracker(MultiplicativeFlux(g,1), [], [-1])
        history = observe_tracker(tracker, np.linspace(1,2,20), [0,.2,.4])
        np.testing.assert_array_equal(history.diagnostics["absorbed"], 0.)
        self.assertNotEqual(tracker.mass_absorbed(), 0.)

    def test_circle_plot(self):
        example = EXAMPLES["circle-zeros"]
        tracker = example.build(delta=0.05, cells=30)
        history = observe_tracker(tracker, np.linspace(0, 1, 200, endpoint=False), np.linspace(0, 0.5, 4))
        plot_circle(history, -1)
        with self.assertRaises(ValueError):
            plot_circle(observe_tracker(EXAMPLES["sine"].build(delta=0.1, cells=10),
                                        np.linspace(-3, 3, 50), np.array([0.0, 0.1])))

    def test_thinning_keeps_whole_paths(self):
        tracker = EXAMPLES["sine"].build(delta=0.005, cells=60).run(1.0)
        pieces = tracker.pieces_until()
        kept = thin_pieces(pieces, 20)
        self.assertLessEqual(len({p.kL for p in kept}), 21)
        kept_levels = {p.kL for p in kept}
        self.assertEqual(len(kept), sum(p.kL in kept_levels for p in pieces))


if __name__ == "__main__":
    unittest.main()
