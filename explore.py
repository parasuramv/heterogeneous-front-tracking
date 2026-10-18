"""Small editable experiment that opens a plot window (or inline in Spyder/Jupyter).

Change g, the mesh, δ, the data and the final time below, then run
``python explore.py``.
"""
import numpy as np
import matplotlib.pyplot as plt

import _srcpath  # noqa: F401
from hetfront import HeteroFrontTracker, MultiplicativeFlux, PiecewiseAffine
from hetfront.visualization import observe_tracker, overview

g_exact = lambda x: x * (1 - x ** 2) + 0.4          # one transverse zero
g = PiecewiseAffine.from_function(g_exact, np.linspace(-2, 2, 41))
flux = MultiplicativeFlux(g, delta=0.01)
u0 = lambda x: np.where(np.abs(x) < 1.5, np.cos(2 * x), 0.0)
tracker = HeteroFrontTracker.from_function(flux, u0, window=(-1.5, 1.5), cells=90)

history = observe_tracker(tracker, np.linspace(-2.5, 2.5, 801), np.linspace(0, 2, 121))
print(tracker.counts)
overview(history, exact_g=g_exact, title="explore.py")
plt.show()
