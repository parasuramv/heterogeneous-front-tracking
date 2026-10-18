"""Regenerate docs/examples.png: θ in space–time for four examples."""
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hetfront.examples import EXAMPLES  # noqa: E402
from hetfront.visualization import observe_tracker, plot_spacetime  # noqa: E402

panels = [("xu2-blowup", "g = x: blow-up at t = 1"),
          ("two-zeros", "g = x(1−x²): three absorbing zeros"),
          ("sine", "well-signed g = 2 + sin 3x"),
          ("circle-zeros", "circle, g = sin 2πx")]
fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
for ax, (name, title) in zip(axes.ravel(), panels):
    example = EXAMPLES[name]
    tracker = example.build()
    x = np.linspace(*example.window, 601, endpoint=not example.periodic)
    history = observe_tracker(tracker, x, np.linspace(0, example.final_time, 121))
    chars = example.characteristics(tracker.flux) if example.characteristic_starts else None
    plot_spacetime(history, ax=ax, characteristics=chars, max_fronts=60)
    ax.set_title(title)
fig.savefig(ROOT / "docs" / "examples.png", dpi=110)
print("saved docs/examples.png")
